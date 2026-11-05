"""Deterministic playback timing plus real socket-worker lifecycle coverage."""
import contextlib
import io
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from inference import protocol
from inference.executor.async_predictor import AsyncPredictor
from inference.executor.executor import FakeCamera, FakeRobot, HAND_ACTION_SCALE, OakCamera, RunLog, parse_args, run


class Clock:
    now = 0.0

    def perf_counter(self):
        return self.now

    monotonic = perf_counter
    time = perf_counter

    def monotonic_ns(self):
        return int(self.now * 1e9)

    def sleep(self, duration):
        self.now += duration


def chunk(length, x=0.01, hand=0.1):
    actions = np.zeros((length, 19))
    actions[:, 0] = x
    actions[:, 6:] = hand
    return actions


class AsyncPlaybackTests(unittest.TestCase):
    def simulate(self, chunks, delays, lead_ms=70, hz=100, pedal_factory=None, alpha=1.0,
                 apply_delay=0.0, substeps=1, error=None, execute_steps=0):
        clock = Clock()
        requests, applied, interpolated = [], [], []
        robot = FakeRobot()
        robot.substeps = substeps

        def apply(action):
            applied.append((clock.now, action.copy()))
            FakeRobot.apply_action(robot, action)
            clock.sleep(apply_delay)

        robot.apply_action = apply
        robot.send_substep = lambda fraction: interpolated.append((clock.now, fraction))
        robot.return_to_start = Mock(wraps=robot.return_to_start)

        class Predictor:
            future = None
            closed = False

            def submit(self, seq, image, state, capture_time):
                if self.future is not None:
                    raise AssertionError("More than one request in flight")
                self.future = seq
                self.ready_at = clock.now + delays[seq]
                requests.append((clock.now, seq, state.copy()))

            def ready(self):
                return self.future is not None and clock.now >= self.ready_at - 1e-12

            def check_error(self):
                if error is not None and self.ready() and self.future == len(chunks) - 1:
                    raise error

            def take(self):
                seq, self.future = self.future, None
                return (seq, chunks[seq], {'infer_ms': delays[seq] * 1000}), delays[seq] * 1000

            def close(self):
                self.closed = True

        predictor = Predictor()
        args = SimpleNamespace(hz=hz, max_cycles=len(chunks), jpeg_quality=85,
                               async_inference=True, inference_latency_ms=lead_ms, camera_timeout=1,
                               execute_steps=execute_steps)
        log = RunLog('/unused')
        pedal = None if pedal_factory is None else pedal_factory(clock, requests, robot)
        output = io.StringIO()
        with patch('inference.executor.executor.time', clock), \
             patch('inference.executor.executor.AsyncPredictor', return_value=predictor), \
             patch('inference.executor.executor.ARM_ACTION_EMA_ALPHA', alpha), \
             contextlib.redirect_stdout(output):
            run(args, robot, FakeCamera(), Mock(), pedal=pedal, log=log)
        self.assertTrue(predictor.closed)
        return SimpleNamespace(requests=requests, applied=applied, log=log, output=output.getvalue(),
                               robot=robot, interpolated=interpolated)

    def test_early_reply_replaces_tail_at_next_action_boundary(self):
        result = self.simulate([chunk(10), chunk(3, .02, .2)], [0, .005], alpha=.4, substeps=4)
        # Chunk 0 lasts 100 ms: request at 30 ms, reply at 35 ms, switch at 40 ms.
        self.assertAlmostEqual(result.requests[1][0], .03)
        self.assertEqual([c['cycle'] for c in result.log.commands], [0] * 4 + [1] * 3)
        self.assertIn('executed=4 dropped=6', result.output)
        times = np.array([t for t, _ in result.applied])
        np.testing.assert_allclose(np.diff(times), .01, atol=1e-12)
        # All substeps finish; no interruption midway through an arm segment.
        self.assertEqual(len(result.interpolated), 7 * 3)
        self.assertEqual([f for _, f in result.interpolated], [.5, .75, 1.] * 7)
        # Keep EMA history across replacement; the new chunk's first action is used.
        expected = .01 * (1 - .6 ** 4)
        self.assertAlmostEqual(result.applied[4][1][0], .4 * .02 + .6 * expected)
        self.assertAlmostEqual(result.applied[4][1][6], .2 * HAND_ACTION_SCALE)

    def test_on_time_and_late_replies_keep_rate_and_hold_when_exhausted(self):
        for latency, next_start in ((.07, .10), (.12, .15)):
            with self.subTest(latency=latency):
                result = self.simulate([chunk(10), chunk(3)], [0, latency])
                self.assertEqual([c['cycle'] for c in result.log.commands], [0] * 10 + [1] * 3)
                self.assertAlmostEqual(result.applied[10][0], next_start, delta=.0021)
                self.assertGreaterEqual(min(np.diff([t for t, _ in result.applied])), .01 - 1e-12)
                if latency > .07:
                    waits = [e for e in result.log.events if e['event'] == 'wait_start']
                    self.assertEqual(len(waits), 1)
                    self.assertAlmostEqual(waits[0]['monotonic_ns'] / 1e9, .1)

    def test_zero_lead_waits_until_end_and_long_lead_does_not_starve_playback(self):
        zero = self.simulate([chunk(10), chunk(2)], [0, .02], lead_ms=0)
        self.assertAlmostEqual(zero.requests[1][0], .1)
        self.assertEqual(len(zero.applied), 12)
        long = self.simulate([chunk(2), chunk(2), chunk(2)], [0, 0, 0], lead_ms=500)
        self.assertEqual([c['cycle'] for c in long.log.commands], [0, 1, 2, 2])
        np.testing.assert_allclose(np.diff([t for t, _ in long.applied]), .01)

    def test_70ms_lead_at_15hz_starts_between_action_and_substep_ticks(self):
        result = self.simulate([chunk(8), chunk(2)], [0, 0], hz=15, substeps=4)
        self.assertAlmostEqual(result.requests[1][0], 8 / 15 - .070)
        self.assertEqual([c['cycle'] for c in result.log.commands], [0] * 7 + [1] * 2)
        np.testing.assert_allclose(np.diff([t for t, _ in result.applied]), 1 / 15)

    def test_selected_prefix_and_frequency_determine_request_time(self):
        for hz in (15, 30):
            for steps in (4, 8):
                with self.subTest(hz=hz, steps=steps):
                    result = self.simulate([chunk(16), chunk(16)], [0, .09], hz=hz, execute_steps=steps)
                    self.assertAlmostEqual(result.requests[1][0], steps / hz - .070)
                    # A late reply holds after the selected prefix; it does not
                    # execute extra predictions from the discarded suffix.
                    self.assertEqual([c['cycle'] for c in result.log.commands], [0] * steps + [1] * steps)
                    self.assertIn(f'planned={steps}', result.output)
                    self.assertIn(f'executed={steps} dropped={16 - steps}', result.output)
                    self.assertEqual(result.log.chunks[0][3].shape, (16, 19))

    def test_selected_prefix_still_drops_remaining_actions_on_early_reply(self):
        result = self.simulate([chunk(16), chunk(16)], [0, .005], execute_steps=10)
        self.assertAlmostEqual(result.requests[1][0], .03)
        self.assertEqual([c['cycle'] for c in result.log.commands], [0] * 4 + [1] * 10)
        self.assertIn('executed=4 dropped=12', result.output)

    def test_prefix_cap_tracks_actual_returned_length_each_cycle(self):
        result = self.simulate([chunk(4), chunk(2), chunk(3)], [0, .07, .07], hz=15, execute_steps=8)
        starts = [e['monotonic_ns'] / 1e9 for e in result.log.events if e['event'] == 'chunk_start']
        for cycle, count in ((0, 4), (1, 2)):
            self.assertAlmostEqual(result.requests[cycle + 1][0] - starts[cycle], count / 15 - .07, places=8)
        self.assertEqual([c['cycle'] for c in result.log.commands], [0] * 4 + [1] * 2 + [2] * 3)

    def test_slow_action_io_shifts_forecast_without_catchup_bursts(self):
        result = self.simulate([chunk(10), chunk(2)], [0, .01], apply_delay=.015)
        self.assertGreaterEqual(min(np.diff([t for t, _ in result.applied])), .025 - 1e-12)
        self.assertEqual(len(result.requests), 2)

    def test_pedal_discards_pending_reply_before_fresh_observation(self):
        def pedal_factory(clock, requests, robot):
            class Pedal:
                pressed = False
                after_return = 0

                def poll(self):
                    if not self.pressed and len(requests) == 2 and clock.now >= .035:
                        self.pressed = True
                        return True
                    if robot.return_to_start.called:
                        self.after_return += 1
                        return self.after_return == 2  # drain, then resume
                    return False
            return Pedal()

        result = self.simulate([chunk(10), chunk(3, .08), chunk(2, .02)], [0, .06, 0],
                               pedal_factory=pedal_factory)
        result.robot.return_to_start.assert_called_once()
        self.assertEqual([c['cycle'] for c in result.log.commands], [0] * 4 + [2] * 2)
        np.testing.assert_array_equal(result.requests[2][2], np.zeros(19))
        self.assertTrue(any(e['event'] == 'chunk_discarded' and e['cycle'] == 1 for e in result.log.events))

    def test_checks_current_command_to_new_chunk_in_addition_to_observation(self):
        old = chunk(10)
        old[:, 0] = np.arange(10) * .08
        # The new chunk starts at its observation (.24), but by arrival the old
        # chunk has moved to .56. Each old step and the new obs->action are valid.
        result = self.simulate([old, chunk(2, .24)], [0, .05])
        self.assertIn('last commanded action', result.output)
        self.assertTrue(all(c['cycle'] == 0 for c in result.log.commands))

    def test_invalid_replacement_is_rejected_before_new_action(self):
        for bad in (np.full((2, 19), np.nan), np.zeros((0, 19)), np.zeros((2, 16))):
            with self.subTest(shape=bad.shape), self.assertRaisesRegex(ValueError, 'finite, nonempty'):
                self.simulate([chunk(10), bad], [0, .01])
        bad = chunk(3)
        bad[-1, 0] = .3
        result = self.simulate([chunk(10), bad], [0, .01])
        self.assertIn('refusing entire chunk', result.output)
        self.assertTrue(all(c['cycle'] == 0 for c in result.log.commands))

    def test_worker_error_stops_during_playback(self):
        with self.assertRaisesRegex(TimeoutError, 'test timeout'):
            self.simulate([chunk(10), chunk(2)], [0, .005], error=TimeoutError('test timeout'))

    def test_cli_defaults_and_latency_validation(self):
        args = parse_args(['--predictor', 'localhost'])
        self.assertFalse(args.async_inference)
        self.assertFalse(args.rtc)
        self.assertEqual(args.inference_latency_ms, 70)
        self.assertEqual(args.execute_steps, 0)
        args = parse_args(['--predictor', 'localhost', '--async-inference', '--inference-latency-ms', '40'])
        self.assertTrue(args.async_inference)
        self.assertEqual(args.inference_latency_ms, 40)
        args = parse_args(['--predictor', 'localhost', '--execute-steps', '8'])
        self.assertEqual(args.execute_steps, 8)
        for value in ('-1', '1.5'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--predictor', 'localhost', '--execute-steps', value])
        for value in ('-1', 'nan', 'inf'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--predictor', 'localhost', '--inference-latency-ms', value])

    def test_execution_modes_are_mutually_exclusive(self):
        args = parse_args(['--predictor', 'localhost', '--rtc'])
        self.assertTrue(args.rtc)
        self.assertFalse(args.async_inference)
        for options in (['--rtc', '--async-inference'], ['--async-inference', '--rtc'],
                        ['--rtc', '--execute-steps', '8']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--predictor', 'localhost', *options])

    def test_no_mode_flag_preserves_synchronous_full_chunk_playback(self):
        args = parse_args(['--predictor', 'localhost', '--fake', '--max-cycles', '2'])
        chunks = [chunk(3, .01, .1), chunk(2, .02, .2)]
        robot, events = FakeRobot(), []
        applied = []

        def apply(action):
            events.append('apply')
            applied.append(action.copy())
            FakeRobot.apply_action(robot, action)

        def receive(sock):
            events.append('reply')
            seq = events.count('reply') - 1
            return seq, chunks[seq], {}

        robot.apply_action = apply
        with patch.object(protocol, 'send_obs', side_effect=lambda *a, **k: events.append('request')) as send, \
             patch.object(protocol, 'recv_action', side_effect=receive), \
             patch('inference.executor.executor.AsyncPredictor') as worker, \
             patch('inference.executor.executor.time.sleep'), \
             contextlib.redirect_stdout(io.StringIO()):
            run(args, robot, FakeCamera(), Mock())
        self.assertEqual(events, ['request', 'reply', 'apply', 'apply', 'apply',
                                  'request', 'reply', 'apply', 'apply'])
        expected = np.concatenate(chunks).astype(np.float64)
        expected[:, 6:] *= HAND_ACTION_SCALE
        np.testing.assert_allclose(applied, expected)
        worker.assert_not_called()
        self.assertTrue(all('rtc' not in call.kwargs for call in send.call_args_list))

    def test_async_camera_reuses_preview_frame_without_waiting_and_rejects_stale_frame(self):
        camera = OakCamera.__new__(OakCamera)
        camera.cv2, camera.flip, camera.timeout = Mock(), 'none', 1.0
        camera.window, camera.latest_frame = None, None
        camera.flip_frame = lambda cv2, frame, flip: frame
        packet = Mock()
        packet.getCvFrame.return_value = np.zeros((4, 4, 3), np.uint8)
        camera.queue = Mock()
        camera.queue.tryGet.side_effect = [packet, None, None, None]
        clock = Clock()
        with patch('inference.executor.executor.time', clock):
            camera.newest_bgr()  # consumed by preview
            self.assertIsNotNone(camera.read_available())
            self.assertEqual(clock.now, 0)
            clock.sleep(1.1)
            with self.assertRaises(TimeoutError):
                camera.read_available()


class AsyncSocketTests(unittest.TestCase):
    def test_real_socket_prediction_overlaps_actions_and_discards_old_tail(self):
        client, server = socket.socketpair()
        client.settimeout(2)
        server.settimeout(2)
        robot = FakeRobot()
        six_actions = threading.Event()
        sent = []
        errors = []

        def apply(action):
            FakeRobot.apply_action(robot, action)
            sent.append(action.copy())
            if len(sent) == 6:
                six_actions.set()

        robot.apply_action = apply

        def serve():
            try:
                for seq in range(2):
                    received, _, _, _ = protocol.recv_obs(server)
                    self.assertEqual(received, seq)
                    if seq == 1:
                        # The controller must continue executing while recv_action
                        # waits; a blocking request/reply loop cannot release this.
                        self.assertLess(len(sent), 6)
                        self.assertTrue(six_actions.wait(1))
                    protocol.send_action(server, seq, chunk(10 if seq == 0 else 2, hand=.1 + seq * .1),
                                         infer_ms=1)
            except BaseException as exc:
                errors.append(exc)
            finally:
                server.close()

        worker = threading.Thread(target=serve)
        worker.start()
        args = SimpleNamespace(hz=50, max_cycles=2, jpeg_quality=85, async_inference=True,
                               inference_latency_ms=150, camera_timeout=1)
        log = RunLog('/unused')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                run(args, robot, FakeCamera(), client, log=log)
        finally:
            client.close()
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        if errors:
            raise errors[0]
        cycles = [c['cycle'] for c in log.commands]
        self.assertGreaterEqual(cycles.count(0), 6)
        self.assertLess(cycles.count(0), 10)
        self.assertEqual(cycles.count(1), 2)
        self.assertAlmostEqual(sent[-1][6], .2 * HAND_ACTION_SCALE)

    def test_worker_exchange_copies_observation_and_rejects_second_request(self):
        client, server = socket.socketpair()
        client.settimeout(1)
        server.settimeout(1)
        predictor = AsyncPredictor(client, 85)
        try:
            state = np.zeros(19, np.float32)
            predictor.submit(7, FakeCamera().read(), state, 123.0)
            state[:] = 1
            seq, image, sent_state, header = protocol.recv_obs(server)
            self.assertEqual(seq, 7)
            self.assertEqual(header['t_capture'], 123)
            np.testing.assert_array_equal(sent_state, np.zeros(19))
            self.assertEqual(image.shape, (240, 320, 3))
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                predictor.submit(8, image, state, 124)
            protocol.send_action(server, 7, chunk(2), infer_ms=5)
            predictor.future.result(timeout=1)
            self.assertTrue(predictor.ready())
            reply, rtt_ms = predictor.take()
            self.assertEqual(reply[0], 7)
            self.assertGreater(rtt_ms, 0)
            self.assertIsNone(predictor.future)
        finally:
            predictor.close()
            client.close()
            server.close()

    def test_shutdown_interrupts_blocked_receive_and_joins_worker(self):
        client, server = socket.socketpair()
        predictor = AsyncPredictor(client, 85)
        try:
            predictor.submit(0, FakeCamera().read(), np.zeros(19), 0)
            protocol.recv_obs(server)
            worker = tuple(predictor.pool._threads)
            start = time.monotonic()
            predictor.close()
            self.assertLess(time.monotonic() - start, 1)
            self.assertTrue(all(not t.is_alive() for t in worker))
        finally:
            predictor.close()
            client.close()
            server.close()

    def test_wrong_sequence_disconnect_and_timeout_propagate(self):
        for mode, error in (('sequence', ValueError), ('disconnect', ConnectionError), ('timeout', TimeoutError)):
            with self.subTest(mode=mode):
                client, server = socket.socketpair()
                client.settimeout(.05)
                predictor = AsyncPredictor(client, 85)
                try:
                    predictor.submit(0, FakeCamera().read(), np.zeros(19), 0)
                    protocol.recv_obs(server)
                    if mode == 'sequence':
                        protocol.send_action(server, 99, chunk(1), infer_ms=1)
                    elif mode == 'disconnect':
                        server.close()
                    with self.assertRaises(error):
                        predictor.future.result(timeout=1)
                    with self.assertRaises(error):
                        predictor.check_error()
                finally:
                    predictor.close()
                    client.close()
                    server.close()


if __name__ == '__main__':
    unittest.main()
