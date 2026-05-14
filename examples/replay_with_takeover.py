import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.piper_api import (  # noqa: E402
    DaggerCsvRecorder,
    KeyboardTakeoverController,
    PiperAPI,
    read_motion_csv,
    timestamped_recording_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a Piper motion CSV with keyboard takeover recording."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="recorded_motion.csv",
        help="CSV file to replay. Default: recorded_motion.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Dataset CSV to write. Default: scripts/recordings/dagger_<timestamp>.csv",
    )
    parser.add_argument(
        "--takeover-only",
        action="store_true",
        help="Only record takeover rows. Default records both replay and takeover rows.",
    )
    parser.add_argument(
        "--record-hz",
        type=float,
        default=60.0,
        help="Dataset logging rate in Hz. Default: 60",
    )
    parser.add_argument(
        "--takeover-delay",
        type=float,
        default=1.0,
        help="Seconds to wait after takeover starts before logging takeover rows. Default: 1",
    )
    parser.add_argument(
        "--interface",
        default="socketcan",
        help="python-can interface. Default: socketcan",
    )
    parser.add_argument(
        "--channel",
        default="can0",
        help="CAN channel. Default: can0",
    )
    parser.add_argument(
        "--firmware",
        choices=["default", "v183", "v188"],
        default="default",
        help="Piper firmware driver selection. Default: default",
    )
    parser.add_argument(
        "--speed-percent",
        type=int,
        default=20,
        help="Arm speed percent sent during replay. Default: 20",
    )
    parser.add_argument(
        "--preposition-speed-percent",
        type=int,
        default=20,
        help="Arm speed percent used to move to the first recorded pose. Default: 20",
    )
    parser.add_argument(
        "--preposition-seconds",
        type=float,
        default=3.0,
        help="Seconds to allow for the slow move to the first recorded pose. Default: 3",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="Seconds to pause after prepositioning before replay. Default: 2",
    )
    parser.add_argument(
        "--ready-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait before moving to the first pose. Default: 0",
    )
    parser.add_argument(
        "--keyboard-toggle-debounce-seconds",
        type=float,
        default=1.0,
        help=(
            "Ignore repeated spacebar events for this long after a toggle. "
            "Default: 1"
        ),
    )
    parser.add_argument(
        "--no-enable",
        action="store_true",
        help="Do not send enable before replay or after takeover.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.record_hz <= 0:
        raise ValueError("--record-hz must be greater than 0")
    if args.takeover_delay < 0:
        raise ValueError("--takeover-delay must be 0 or greater")
    if not 0 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent must be between 0 and 100")
    if not 0 <= args.preposition_speed_percent <= 100:
        raise ValueError("--preposition-speed-percent must be between 0 and 100")
    if args.preposition_seconds < 0:
        raise ValueError("--preposition-seconds must be 0 or greater")
    if args.settle_seconds < 0:
        raise ValueError("--settle-seconds must be 0 or greater")
    if args.ready_seconds < 0:
        raise ValueError("--ready-seconds must be 0 or greater")
    if args.keyboard_toggle_debounce_seconds < 0:
        raise ValueError("--keyboard-toggle-debounce-seconds must be 0 or greater")


def wait_with_message(message, seconds):
    if seconds <= 0:
        return
    print(message.format(seconds=seconds))
    time.sleep(seconds)


def move_to_initial_pose(api, first_joints, args):
    wait_with_message(
        "Get ready. Initial positioning starts in {seconds:.1f} seconds.",
        args.ready_seconds,
    )
    print("Moving slowly to the first recorded pose...")
    api.set_speed_percent(args.preposition_speed_percent)
    api.move_joint(first_joints, mode="smooth")
    wait_with_message(
        "Allowing {seconds:.1f} seconds for initial positioning.",
        args.preposition_seconds,
    )
    wait_with_message("Settling for {seconds:.1f} seconds.", args.settle_seconds)


def newest_due_sample(samples, start_index, replay_elapsed_s):
    if start_index >= len(samples):
        return len(samples), None

    index = start_index
    newest = None
    while index < len(samples) and samples[index].elapsed_s <= replay_elapsed_s:
        newest = samples[index]
        index += 1
    return index, newest


def main():
    args = parse_args()
    validate_args(args)

    samples = read_motion_csv(args.csv_path)
    if not samples:
        raise ValueError(f"No motion samples found in {args.csv_path}")

    output_path = timestamped_recording_path(
        output=args.output,
        directory=str(REPO_ROOT / "scripts" / "recordings"),
        prefix="dagger",
    )

    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
    )
    api.connect()

    takeover = KeyboardTakeoverController(
        api,
        takeover_delay_s=args.takeover_delay,
        toggle_debounce_s=args.keyboard_toggle_debounce_seconds,
    )

    try:
        print("Connected OK:", api.is_ok())
        print("Receive FPS:", api.get_fps())

        api.set_speed_percent(args.speed_percent)
        if not args.no_enable:
            print("Enabling robot...")
            if not api.enable():
                raise RuntimeError("Timed out while enabling robot")

        move_to_initial_pose(api, samples[0].joints, args)

        api.set_speed_percent(args.speed_percent)

        print(f"Writing dataset to {output_path}")
        print(
            "Press SPACE to enter takeover; press SPACE again to resume replay."
        )

        replay_start_t = time.monotonic()
        first_elapsed_s = samples[0].elapsed_s
        last_sample_elapsed_s = samples[-1].elapsed_s
        next_sample_index = 0
        next_record_t = time.monotonic()
        record_period_s = 1.0 / args.record_hz
        last_target_joints = None
        last_flush_rows = 0
        mode = "replay"
        trajectory_finished = False

        with DaggerCsvRecorder(output_path) as recorder:
            takeover.start()
            while True:
                now = time.monotonic()
                replay_elapsed_s = first_elapsed_s + (now - replay_start_t)

                for event in takeover.poll():
                    if event.kind == "takeover_start":
                        print(
                            f"Keyboard takeover at replay t={replay_elapsed_s:.3f}s"
                        )
                        print(event.message)
                        mode = "takeover"
                        recorder.write(
                            robot=api,
                            source="event",
                            replay_elapsed_s=replay_elapsed_s,
                            event="takeover_start",
                        )
                    elif event.kind == "takeover_stop":
                        print(
                            f"Keyboard exit takeover at replay t={replay_elapsed_s:.3f}s"
                        )
                        print(event.message)
                        recorder.write(
                            robot=api,
                            source="event",
                            replay_elapsed_s=replay_elapsed_s,
                            event="takeover_stop",
                        )
                        if trajectory_finished:
                            print("Replay trajectory already ended; stopping.")
                            return
                        mode = "replay"
                        api.set_speed_percent(args.speed_percent)
                        if not args.no_enable:
                            print("Re-enabling robot for replay...")
                            if not api.enable():
                                raise RuntimeError(
                                    "Timed out while re-enabling robot after takeover"
                                )

                if mode == "replay":
                    next_sample_index, sample = newest_due_sample(
                        samples, next_sample_index, replay_elapsed_s
                    )
                    if sample is not None:
                        api.move_joint(sample.joints, mode="smooth")
                        last_target_joints = sample.joints

                    if not args.takeover_only and now >= next_record_t:
                        recorder.write(
                            robot=api,
                            source="replay",
                            replay_elapsed_s=replay_elapsed_s,
                            target_joints=last_target_joints,
                        )
                        next_record_t = now + record_period_s

                    if (
                        replay_elapsed_s > last_sample_elapsed_s
                        and next_sample_index >= len(samples)
                    ):
                        print("Replay complete.")
                        return

                elif mode == "takeover":
                    if replay_elapsed_s > last_sample_elapsed_s:
                        trajectory_finished = True
                    if takeover.should_record_takeover():
                        if now >= next_record_t:
                            recorder.write(
                                robot=api,
                                source="takeover",
                                replay_elapsed_s=replay_elapsed_s,
                            )
                            next_record_t = now + record_period_s

                flush_every = max(1, int(args.record_hz))
                if recorder.rows_written - last_flush_rows >= flush_every:
                    recorder.flush()
                    last_flush_rows = recorder.rows_written

                time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nInterrupted. Stopping script without disabling motors.")
    finally:
        takeover.stop()
        api.disconnect()


if __name__ == "__main__":
    main()
