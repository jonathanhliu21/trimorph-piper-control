#!/usr/bin/env python3
"""Passively record attached Quest + Piper into DATASET/raw/episode_NNNNNN.

The foot pedal starts episodes after a short ready delay. During recording,
press/release to save, or hold to discard, matching collect_hand_only.py.
--episode-s records one timed episode; --no-pedal uses Enter/Space instead.
No hand/camera is required and no enable or movement commands are sent.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_collection.collection import config as defaults
from data_collection.collection.episode import episode_paths, next_episode_index
from data_collection.collection.quest_worker import run_quest_worker
from data_collection.collection.utils import utc_stamp, worker_wrapper
from scripts.quest_piper.common import POSE_FIELDS, write_csv, write_json
from scripts.quest.visualize_quest_accuracy_piper import PIPER, PiperHistory
from scipy.spatial.transform import Rotation


@dataclass
class StreamConfig:
    quest_host: str
    quest_port: int
    quest_startup_timeout_s: float
    quest_jump_threshold: float = .05


class Recorder:
    """Minimal recording-session interface used by the existing Quest worker."""
    def __init__(self, root):
        self.root, self.lock, self.session = root, threading.Lock(), None

    def current(self):
        with self.lock:
            return self.session

    def start(self):
        with self.lock:
            paths = episode_paths(self.root, next_episode_index(self.root / 'raw'))
            self.session = Episode(paths)
            return self.session

    def stop(self, reason, error=None):
        with self.lock:
            session, self.session = self.session, None
        if session is not None:
            session.finish(reason, error)
        return session

    def discard(self):
        # Unsaved episode rows live only in memory, so discard without writing files.
        with self.lock:
            session, self.session = self.session, None
        if session is not None:
            with session.lock:
                session.closed = True
            print(f'Episode {session.paths.episode_index:06d} deleted.')


@dataclass
class FootPedalController:
    args: object
    recorder: Recorder
    pedal: object
    pending_start_deadline: float | None = None
    recording_pedal_down: bool = False

    def poll(self):
        pressed = bool(self.pedal.poll())
        if self.recorder.current() is not None:
            hold = self.pedal.current_hold_s()
            if pressed and not self.recording_pedal_down:
                self.recording_pedal_down = True
                print(f'Pedal held. Release to save; hold {self.args.hold_delete_s:g}s to delete.')
            if self.recording_pedal_down:
                if hold is not None and hold >= self.args.hold_delete_s:
                    self.recording_pedal_down = False
                    self.recorder.discard()
                elif hold is None:
                    self.recording_pedal_down = False
                    self.recorder.stop('foot_pedal')
            return
        if self.pending_start_deadline is None:
            if pressed:
                self.pending_start_deadline = time.monotonic() + self.args.ready_seconds
                print(f'Pedal pressed. Recording starts in {self.args.ready_seconds:.1f}s.')
        elif pressed:
            self.pending_start_deadline = None
            print('Start cancelled by second pedal press during ready delay.')
        elif time.monotonic() >= self.pending_start_deadline:
            print(f'Recording {self.recorder.start().paths.episode_dir}')
            self.pending_start_deadline = None


class Episode:
    def __init__(self, paths):
        self.paths, self.lock = paths, threading.Lock()
        self.start_ns, self.started_utc = time.monotonic_ns(), utc_stamp()
        self.quest, self.piper, self.closed = [], [], False

    def elapsed_s(self, stamp):
        return (stamp-self.start_ns)/1e9

    def append_quest(self, row):
        with self.lock:
            if not self.closed:
                self.quest.append(dict(row))

    def append_piper(self, row):
        with self.lock:
            if not self.closed:
                self.piper.append(dict(row, sample_idx=len(self.piper), elapsed_s=self.elapsed_s(row['monotonic_ns'])))

    def finish(self, reason, error=None):
        with self.lock:
            self.closed = True
            stop = time.monotonic_ns()
            write_csv(self.paths.quest_path, self.quest)
            write_csv(self.paths.piper_path, self.piper)
            complete = error is None and len(self.quest) >= 2 and len(self.piper) >= 2
            write_json(self.paths.metadata_path, dict(
                episode_index=self.paths.episode_index, status='complete' if complete else 'error',
                stop_reason=reason, errors=[] if error is None else [str(error)],
                started_utc=self.started_utc, stopped_utc=utc_stamp(),
                started_monotonic_ns=self.start_ns, stopped_monotonic_ns=stop,
                duration_s=(stop-self.start_ns)/1e9, sync_clock='host monotonic_ns',
                devices=['quest', 'piper'], hand={'enabled': False},
                paths=dict(quest_csv='quest.csv', piper_csv='piper.csv', leader_hand_csv=None,
                           follower_csv=None, camera_mp4=None, mocap_csv=None),
                counts=dict(quest_samples=len(self.quest), piper_samples=len(self.piper), leader_hand_samples=0),
                piper=dict(pose_source='joint-feedback FK', units='meters/radians',
                           start_pose=([self.piper[0][k] for k in POSE_FIELDS] if self.piper else None)),
                quest=dict(units='meters', quaternion_order='xyzw')))
            print(f'Saved {self.paths.episode_dir}: Quest={len(self.quest)}, Piper={len(self.piper)}')


def piper_worker(args, shared, recorder, *, init_done):
    from scripts.piper_api.piper_api import PiperAPI
    api = PiperAPI(firmware=args.firmware, interface=args.interface, channel=args.channel,
                   installation_pos=None, payload=None)
    previous = None
    try:
        if not api.connect():
            raise RuntimeError('Piper unavailable on CAN')
        while not shared.shutdown.is_set():
            joints = api.get_joint_state()
            if joints is not None and joints.timestamp != previous:
                previous = joints.timestamp
                pose = np.asarray(api.robot.fk(list(joints.joints)), float)
                if pose.shape != (6,) or not np.isfinite(pose).all():
                    raise ValueError('Invalid Piper FK')
                stamp = time.monotonic_ns()
                shared.set_tracker_pose(PIPER, stamp, tuple(pose[:3]),
                                        tuple(Rotation.from_euler('xyz', pose[3:]).as_quat()))
                state = api.get_state()
                shared.set_feedback(joints.joints, state)
                init_done.set()
                session = recorder.current()
                if session is not None:
                    eef = api.get_eef_state()
                    row = dict(monotonic_ns=stamp, wall_time_s=time.time(),
                               robot_timestamp_s=joints.timestamp, robot_hz=joints.hz,
                               teach_status=str(state.get('teach_status', '')),
                               arm_status=str(state.get('arm_status', '')), err_code=state.get('err_code', 0))
                    row.update(dict(zip(POSE_FIELDS, map(float, pose))))
                    row.update({f'fk_{key}': float(value) for key, value in zip(POSE_FIELDS, pose)})
                    row.update({f'joint_{i+1}': float(q) for i, q in enumerate(joints.joints)})
                    row.update({f'fw_{key}': '' if eef is None else float(eef.pose[i]) for i, key in enumerate(POSE_FIELDS)})
                    session.append_piper(row)
            shared.shutdown.wait(1/args.rate_hz)
    finally:
        api.disconnect()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--calibration-json', '--calibration', type=Path,
                   help='Viewer calibration only; raw streams remain unchanged. Otherwise press C in the view.')
    p.add_argument('--no-view', action='store_true', help='Headless recording')
    p.add_argument('--episode-s', type=float)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--pedal', '--use-foot-pedal', dest='pedal', action='store_true', default=True,
                      help='Foot-pedal episode control (default)')
    mode.add_argument('--no-pedal', dest='pedal', action='store_false', help='Use Enter/Space episode control')
    p.add_argument('--foot-pedal-device')
    p.add_argument('--foot-pedal-key-code', type=int)
    p.add_argument('--foot-pedal-debounce-seconds', type=float, default=defaults.DEFAULT_FOOT_PEDAL_DEBOUNCE_S)
    p.add_argument('--ready-seconds', type=float, default=defaults.DEFAULT_READY_SECONDS)
    p.add_argument('--hold-delete-s', type=float, default=defaults.DEFAULT_HOLD_DELETE_S)
    p.add_argument('--quest-port', type=int, default=5005)
    p.add_argument('--quest-host', default='0.0.0.0')
    p.add_argument('--channel', default=defaults.DEFAULT_PIPER_CHANNEL)
    p.add_argument('--interface', default=defaults.DEFAULT_PIPER_INTERFACE)
    p.add_argument('--firmware', default=defaults.DEFAULT_PIPER_FIRMWARE)
    p.add_argument('--rate-hz', type=float, default=200.)
    p.add_argument('--startup-timeout-s', type=float, default=15.)
    args = p.parse_args(argv)
    for key in ('episode_s', 'rate_hz', 'startup_timeout_s', 'hold_delete_s'):
        value = getattr(args, key)
        if value is not None and (not np.isfinite(value) or value <= 0):
            p.error(f'--{key.replace("_", "-")} must be finite and positive')
    if not 1 <= args.quest_port <= 65535:
        p.error('--quest-port must be 1..65535')
    for key in ('ready_seconds', 'foot_pedal_debounce_seconds'):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            p.error(f'--{key.replace("_", "-")} must be finite and nonnegative')
    if args.episode_s is not None:
        args.pedal = False
    return args


def main(argv=None):
    args = parse_args(argv)
    recorder, shared, errors = Recorder(args.dataset_root), PiperHistory(), queue.Queue()
    threads, ready, pedal, view = [], [], None, None
    pedal_controller = None
    failure = None
    try:
        for name, target, config in (('quest', run_quest_worker, StreamConfig(args.quest_host, args.quest_port, args.startup_timeout_s)),
                                     ('piper', piper_worker, args)):
            initialized = threading.Event()
            thread = threading.Thread(target=worker_wrapper,
                args=(name, errors, shared, initialized, target, config, shared, recorder), daemon=True)
            threads.append(thread)
            ready.append(initialized)
            thread.start()
        deadline = time.monotonic() + args.startup_timeout_s
        while not all(event.is_set() for event in ready):
            if shared.shutdown.wait(.02):
                raise RuntimeError(errors.get_nowait().message if not errors.empty() else 'Worker stopped')
            if time.monotonic() > deadline:
                raise TimeoutError('Timed out waiting for Piper and Quest')
        keys = queue.Queue()
        if not args.no_view:
            from scripts.quest_piper.visualizer import CollectionView
            view = CollectionView(shared, args.calibration_json)
        if args.pedal:
            from scripts.piper_api.foot_pedal import FootPedalTrigger
            pedal = FootPedalTrigger(args.foot_pedal_device, key_code=args.foot_pedal_key_code,
                                     debounce_s=args.foot_pedal_debounce_seconds)
            pedal.start()
            pedal_controller = FootPedalController(args, recorder, pedal)
            print(f'Foot pedal input: {pedal.active_device}')
        if args.episode_s is None:
            def keyboard():
                for line in sys.stdin:
                    keys.put(line.strip().lower())
                if not args.pedal:
                    keys.put('q')
            threading.Thread(target=keyboard, daemon=True).start()
        print('Passive recording: move attached Quest/Piper with built-in drag-teach. '
              + ('Pedal starts; press/release saves; long hold deletes.' if args.pedal else 'Enter/Space toggles recording.'))
        if args.episode_s is not None:
            recorder.start()
        while not shared.shutdown.is_set():
            if view is not None:
                active = recorder.current()
                control = 'Pedal: start / release to save / hold to delete' if args.pedal else 'Space: start/stop episode'
                if pedal_controller is not None and pedal_controller.pending_start_deadline is not None:
                    control = f'Starting in {max(0., pedal_controller.pending_start_deadline-time.monotonic()):.1f}s'
                view.view.banner = (f'RECORDING {active.paths.episode_dir.name}' if active else 'READY') + ' | ' + control
            if view is not None and not view.update():
                break
            session = recorder.current()
            if session is not None and args.episode_s is not None and session.elapsed_s(time.monotonic_ns()) >= args.episode_s:
                recorder.stop('duration')
                break
            if pedal_controller is not None:
                pedal_controller.poll()
            toggle = bool(not args.pedal and view is not None and view.toggle_recording)
            while not keys.empty():
                key = keys.get()
                if key == 'q':
                    shared.shutdown.set()
                elif key == '' and not args.pedal:
                    toggle = True
            if toggle:
                if recorder.current() is None:
                    print(f'Recording {recorder.start().paths.episode_dir}')
                else:
                    recorder.stop('operator')
            shared.shutdown.wait(.02)
        if not errors.empty():
            error = errors.get_nowait()
            raise RuntimeError(f'{error.name}: {error.message}')
    except KeyboardInterrupt:
        pass
    except BaseException as exc:
        failure = exc
        raise
    finally:
        shared.shutdown.set()
        for thread in threads:
            thread.join(timeout=2.)
        if pedal is not None:
            pedal.stop()
        recorder.stop('error' if failure else 'shutdown', failure)
        if view is not None:
            view.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
