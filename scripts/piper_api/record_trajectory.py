#!/usr/bin/env python3
"""Record a drag-taught Piper end-effector trajectory with a foot pedal."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.piper_api.foot_pedal import (  # noqa: E402
    FootPedalTrigger,
    format_input_devices,
    list_input_devices,
)
from scripts.piper_api.piper_api import PiperAPI  # noqa: E402
from scripts.piper_api.teacher_mode import get_teacher_mode  # noqa: E402


EEF_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
CSV_FIELDS = (
    "sample_idx",
    "monotonic_ns",
    "wall_time_s",
    "elapsed_s",
    "robot_timestamp_s",
    "robot_hz",
    "teach_status",
    *EEF_FIELDS,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record a passively drag-taught Piper EEF trajectory using a foot "
            "pedal."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination CSV path.",
    )
    parser.add_argument("--rate-hz", type=float, default=200.0)
    parser.add_argument(
        "--episode-s",
        type=float,
        help=(
            "Record for this many seconds after the first pedal press, then "
            "stop and save automatically. By default, a second press stops recording."
        ),
    )
    parser.add_argument("--teacher-mode-poll-hz", type=float, default=20.0)
    parser.add_argument("--foot-pedal-device")
    parser.add_argument("--foot-pedal-key-code", type=int)
    parser.add_argument("--foot-pedal-debounce-seconds", type=float, default=0.5)
    parser.add_argument("--list-input-devices", action="store_true")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--channel", default="can0")
    parser.add_argument(
        "--firmware", choices=("default", "v183", "v188"), default="v183"
    )
    parser.add_argument(
        "--installation-pos",
        choices=("unchanged", "horizontal", "left", "right"),
        default="right",
    )
    parser.add_argument(
        "--payload",
        choices=("unchanged", "empty", "half", "full"),
        default="half",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output CSV if it already exists.",
    )
    return parser.parse_args(argv)


def write_trajectory(path: Path, rows: list[dict[str, object]], *, force: bool) -> None:
    if not rows:
        raise RuntimeError("No Piper samples were recorded")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(
            f"Trajectory already exists: {path} (pass --force to replace it)"
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    if args.output is None:
        raise ValueError("--output is required")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be greater than 0")
    if args.episode_s is not None and args.episode_s <= 0:
        raise ValueError("--episode-s must be greater than 0")
    if args.teacher_mode_poll_hz <= 0:
        raise ValueError("--teacher-mode-poll-hz must be greater than 0")
    if args.foot_pedal_debounce_seconds < 0:
        raise ValueError("--foot-pedal-debounce-seconds must be non-negative")

    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Trajectory already exists: {output_path} (pass --force to replace it)"
        )

    pedal = FootPedalTrigger(
        args.foot_pedal_device,
        key_code=args.foot_pedal_key_code,
        debounce_s=args.foot_pedal_debounce_seconds,
    )
    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
        installation_pos=args.installation_pos,
        payload=args.payload,
    )
    rows: list[dict[str, object]] = []
    recording = False
    start_ns = 0
    recording_deadline_ns: int | None = None
    next_sample_ns = 0
    sample_period_ns = int(round(1_000_000_000 / args.rate_hz))
    teacher_period_ns = int(round(1_000_000_000 / args.teacher_mode_poll_hz))
    next_teacher_ns = 0
    teach_status: object = None

    try:
        pedal.start()
        print(f"Foot pedal input: {pedal.active_device}")
        api.connect()
        print(f"Piper connected: {api.is_ok()} (receive FPS: {api.get_fps():.1f})")
        if not api.is_ok():
            raise RuntimeError("Piper is not responding on the CAN bus")
        print(
            "Piper remains passive. Enable its drag-teach mode, then press the "
            "pedal to start recording."
        )

        while True:
            if pedal.poll():
                if recording:
                    if args.episode_s is None:
                        break
                    continue
                recording = True
                start_ns = time.monotonic_ns()
                recording_deadline_ns = (
                    start_ns + max(1, int(round(args.episode_s * 1_000_000_000)))
                    if args.episode_s is not None
                    else None
                )
                next_sample_ns = start_ns
                next_teacher_ns = start_ns
                if args.episode_s is None:
                    print("Recording started. Press the pedal again to stop and save.")
                else:
                    print(
                        f"Recording started for {args.episode_s:g}s; "
                        "it will stop and save automatically."
                    )

            now_ns = time.monotonic_ns()
            if recording and now_ns >= next_teacher_ns:
                _active, teach_status = get_teacher_mode(api)
                next_teacher_ns += teacher_period_ns
                while next_teacher_ns < now_ns:
                    next_teacher_ns += teacher_period_ns

            if recording and now_ns >= next_sample_ns:
                eef = api.get_eef_state()
                sample_ns = time.monotonic_ns()
                if eef is not None:
                    row: dict[str, object] = {
                        "sample_idx": len(rows),
                        "monotonic_ns": sample_ns,
                        "wall_time_s": f"{time.time():.6f}",
                        "elapsed_s": f"{(sample_ns - start_ns) / 1e9:.9f}",
                        "robot_timestamp_s": f"{eef.timestamp:.6f}",
                        "robot_hz": f"{eef.hz:.3f}",
                        "teach_status": str(teach_status),
                    }
                    row.update(
                        {
                            field: f"{float(value):.9f}"
                            for field, value in zip(EEF_FIELDS, eef.pose)
                        }
                    )
                    rows.append(row)
                next_sample_ns += sample_period_ns
                while next_sample_ns < sample_ns:
                    next_sample_ns += sample_period_ns

            if (
                recording_deadline_ns is not None
                and time.monotonic_ns() >= recording_deadline_ns
            ):
                print(f"Reached the {args.episode_s:g}s episode duration.")
                break

            wait_s = 0.005
            if recording:
                next_wake_ns = next_sample_ns
                if recording_deadline_ns is not None:
                    next_wake_ns = min(next_wake_ns, recording_deadline_ns)
                wait_s = max(
                    0.0,
                    min(wait_s, (next_wake_ns - time.monotonic_ns()) / 1e9),
                )
            if wait_s:
                time.sleep(wait_s)
    except KeyboardInterrupt:
        if not recording:
            print("\nCancelled before recording started.")
            return 130
        print("\nRecording interrupted; saving captured samples.")
    finally:
        api.disconnect()
        pedal.stop()

    write_trajectory(output_path, rows, force=args.force)
    duration_s = float(rows[-1]["elapsed_s"])
    print(f"Saved {len(rows)} samples ({duration_s:.3f}s) to {output_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_input_devices:
        print(format_input_devices(list_input_devices()))
        return 0
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
