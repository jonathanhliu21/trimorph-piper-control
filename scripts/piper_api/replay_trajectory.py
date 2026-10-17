#!/usr/bin/env python3
"""Replay a Piper end-effector trajectory recorded by record_trajectory.py."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import sys
import time
from pathlib import Path
from typing import Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.piper_api.piper_api import PiperAPI  # noqa: E402


EEF_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
JOINT_FIELDS = tuple(f"joint_{joint}" for joint in range(1, 7))
STATUS_FIELDS = (
    "is_ok",
    "receive_fps",
    "ctrl_mode",
    "arm_status",
    "mode_feedback",
    "teach_status",
    "motion_status",
    "trajectory_num",
    "err_code",
)
POSITION_MODE_SETTLE_S = 0.5
DEFAULT_MAX_JOINT_STEP_RAD = 2.0
JOINT_START_WARNING_RAD = 0.1
DEFAULT_JOINT_START_TOLERANCE_RAD = 0.15
JOINT_PREPOSITION_RATE_HZ = 20.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a Piper EEF trajectory CSV at a fixed command rate."
    )
    parser.add_argument(
        "trajectory",
        type=Path,
        help="trajectory.csv, piper.csv, or a directory containing either file.",
    )
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument(
        "--joint-level",
        action="store_true",
        help="Replay recorded joint angles with smooth move_j instead of firmware EEF IK.",
    )
    parser.add_argument(
        "--max-joint-step-rad",
        type=float,
        default=DEFAULT_MAX_JOINT_STEP_RAD,
        help="Maximum change in any commanded joint from its measured angle per joint-level command (default: %(default)s rad).",
    )
    parser.add_argument("--no-preposition", action="store_true")
    parser.add_argument("--preposition-seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--preposition-speed-percent", type=int, default=20)
    parser.add_argument("--replay-speed-percent", type=int, default=100)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print controller state and commanded-versus-measured EEF poses.",
    )
    parser.add_argument(
        "--debug-interval-s",
        type=float,
        default=0.5,
        help="Seconds between motion diagnostics when --debug is enabled.",
    )
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
    return parser.parse_args(argv)


def resolve_trajectory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        for filename in ("trajectory.csv", "piper.csv"):
            candidate = path / filename
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No trajectory.csv or piper.csv found in {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory not found: {path}")
    return path


def load_trajectory(
    path: Path, *, joint_level: bool = False
) -> tuple[list[float], list[list[float]]]:
    fields = JOINT_FIELDS if joint_level else EEF_FIELDS
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        required = {"elapsed_s", *fields}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No samples in {path}")

    times = [float(row["elapsed_s"]) for row in rows]
    poses = [[float(row[field]) for field in fields] for row in rows]
    if not all(math.isfinite(value) for value in times):
        raise ValueError(f"Non-finite timestamp in {path}")
    if not all(math.isfinite(value) for pose in poses for value in pose):
        raise ValueError(f"Non-finite pose in {path}")
    if any(right < left for left, right in zip(times, times[1:])):
        raise ValueError(f"Timestamps are not monotonic in {path}")
    return times, poses


def validate_joint_continuity(
    targets: Sequence[Sequence[float]], *, max_step_rad: float = 0.25
) -> None:
    """Reject recording gaps or angle wraps that could command a large sweep."""

    for row_index in range(1, len(targets)):
        previous = targets[row_index - 1]
        current = targets[row_index]
        for joint_index, (before, after) in enumerate(zip(previous, current), start=1):
            delta = abs(float(after) - float(before))
            if delta > max_step_rad:
                raise ValueError(
                    f"Recorded J{joint_index} jumps {delta:.3f} rad between rows "
                    f"{row_index - 1} and {row_index}; joint replay aborted"
                )


def verify_joint_start(
    api: PiperAPI,
    target: Sequence[float],
    *,
    tolerance_rad: float = DEFAULT_JOINT_START_TOLERANCE_RAD,
) -> None:
    """Do not stream a trajectory from an unverified starting configuration."""

    measured = api.get_joint_angles()
    if measured is None or len(measured) != 6 or len(target) != 6:
        raise RuntimeError("Cannot verify Piper joint angles before replay")
    if not all(math.isfinite(float(value)) for value in measured):
        raise RuntimeError("Piper returned non-finite joint angles before replay")
    worst = max(
        abs(float(actual) - float(goal))
        for actual, goal in zip(measured, target)
    )
    if worst > tolerance_rad:
        raise RuntimeError(
            f"Piper is not at the recorded joint start (worst error {worst:.3f} rad; "
            f"allowed {tolerance_rad:.3f} rad). Increase --preposition-seconds "
            "or check the arm state."
        )
    if worst > JOINT_START_WARNING_RAD:
        print(
            f"Warning: Piper is {worst:.3f} rad from the recorded joint start; "
            "replay will continue with bounded joint steps.",
            file=sys.stderr,
            flush=True,
        )


def read_joint_angles(api: PiperAPI) -> list[float]:
    measured = api.get_joint_angles()
    if measured is None or len(measured) != 6:
        raise RuntimeError("Piper joint feedback unavailable; refusing joint command")
    angles = [float(value) for value in measured]
    if not all(math.isfinite(value) for value in angles):
        raise RuntimeError("Piper joint feedback is non-finite; refusing joint command")
    return angles


def interpolate_joint_step(
    current: Sequence[float], target: Sequence[float], max_step_rad: float
) -> list[float]:
    """Take one straight-line joint-space step, bounded on every axis."""

    if max_step_rad <= 0 or not math.isfinite(max_step_rad):
        raise ValueError("--max-joint-step-rad must be finite and greater than 0")
    if len(current) != 6 or len(target) != 6:
        raise ValueError("Joint vectors must have six angles")
    current_values = [float(value) for value in current]
    target_values = [float(value) for value in target]
    if not all(math.isfinite(value) for value in (*current_values, *target_values)):
        raise ValueError("Joint vectors must contain finite angles")
    delta = [goal - actual for actual, goal in zip(current_values, target_values)]
    largest = max(abs(value) for value in delta)
    alpha = min(1.0, max_step_rad / largest) if largest else 1.0
    return [actual + alpha * change for actual, change in zip(current_values, delta)]


def command_bounded_joint_step(
    api: PiperAPI, target: Sequence[float], max_step_rad: float
) -> list[float]:
    """Never command more than one bounded step from measured joint feedback."""

    command = interpolate_joint_step(read_joint_angles(api), target, max_step_rad)
    api.move_joint(command, mode="smooth")
    return command


def preposition_joints_linearly(
    api: PiperAPI,
    target: Sequence[float],
    *,
    duration_s: float,
    max_step_rad: float,
) -> None:
    """Ramp from measured joints to the start without an initial target jump."""

    if max_step_rad <= 0 or not math.isfinite(max_step_rad):
        raise ValueError("--max-joint-step-rad must be finite and greater than 0")
    if duration_s < 0 or not math.isfinite(duration_s):
        raise ValueError("--preposition-seconds must be finite and non-negative")
    start = read_joint_angles(api)
    largest = max(abs(float(goal) - actual) for actual, goal in zip(start, target))
    if largest == 0:
        return
    steps = math.ceil(largest / max_step_rad)
    period_s = 1.0 / JOINT_PREPOSITION_RATE_HZ
    required_s = steps * period_s
    if required_s > duration_s + 1e-9:
        raise RuntimeError(
            f"Joint preposition needs at least {required_s:.2f}s for a "
            f"{largest:.3f} rad gap at {max_step_rad:.3f} rad/step; "
            f"increase --preposition-seconds (currently {duration_s:.2f}s)"
        )
    for step in range(1, steps + 1):
        waypoint = [
            actual + (float(goal) - actual) * step / steps
            for actual, goal in zip(start, target)
        ]
        command_bounded_joint_step(api, waypoint, max_step_rad)
        time.sleep(period_s)
    remaining_s = duration_s - required_s
    if remaining_s > 0:
        time.sleep(remaining_s)


def sample_pose(
    times: list[float], poses: list[list[float]], query_s: float
) -> list[float]:
    if query_s <= times[0]:
        return list(poses[0])
    if query_s >= times[-1]:
        return list(poses[-1])
    right = bisect.bisect_right(times, query_s)
    left = right - 1
    span_s = times[right] - times[left]
    if span_s <= 0:
        return list(poses[right])
    alpha = (query_s - times[left]) / span_s
    return [
        start + alpha * (end - start)
        for start, end in zip(poses[left], poses[right])
    ]


def _pose_from_state(state: dict[str, object]) -> list[float] | None:
    try:
        pose = [float(state[field]) for field in EEF_FIELDS]
    except (KeyError, TypeError, ValueError):
        return None
    return pose if all(math.isfinite(value) for value in pose) else None


def _format_pose(pose: Sequence[float] | None) -> str:
    if pose is None:
        return "unavailable"
    return "[" + ", ".join(f"{float(value):+.6f}" for value in pose) + "]"


def log_debug_state(
    api: PiperAPI,
    label: str,
    *,
    target: Sequence[float] | None = None,
) -> None:
    state = api.get_state()
    status = " ".join(f"{field}={state.get(field)!r}" for field in STATUS_FIELDS)
    actual = _pose_from_state(state)
    print(f"[debug:{label}] {status}")
    print(f"[debug:{label}] measured_eef={_format_pose(actual)}")
    if target is None:
        return
    target_values = [float(value) for value in target]
    print(f"[debug:{label}] commanded_eef={_format_pose(target_values)}")
    if actual is not None:
        xyz_error_m = math.sqrt(
            sum((target_values[i] - actual[i]) ** 2 for i in range(3))
        )
        rpy_error_rad = math.sqrt(
            sum((target_values[i] - actual[i]) ** 2 for i in range(3, 6))
        )
        print(
            f"[debug:{label}] xyz_error_m={xyz_error_m:.6f} "
            f"rpy_error_rad={rpy_error_rad:.6f}"
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be greater than 0")
    if args.preposition_seconds < 0 or args.settle_seconds < 0:
        raise ValueError("Preposition and settle durations must be non-negative")
    if args.debug_interval_s <= 0:
        raise ValueError("--debug-interval-s must be greater than 0")
    if not math.isfinite(args.max_joint_step_rad) or args.max_joint_step_rad <= 0:
        raise ValueError("--max-joint-step-rad must be finite and greater than 0")
    for name in ("preposition_speed_percent", "replay_speed_percent"):
        if not 0 <= getattr(args, name) <= 100:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 100")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    path = resolve_trajectory(args.trajectory)
    times, poses = load_trajectory(path, joint_level=args.joint_level)
    if args.joint_level:
        validate_joint_continuity(poses)
    start_s, end_s = times[0], times[-1]
    debug = args.debug

    if debug:
        ranges = [
            max(pose[index] for pose in poses) - min(pose[index] for pose in poses)
            for index in range(len(JOINT_FIELDS if args.joint_level else EEF_FIELDS))
        ]
        print(
            f"[debug:trajectory] samples={len(poses)} duration_s={end_s - start_s:.6f} "
            f"first={_format_pose(poses[0])} last={_format_pose(poses[-1])}"
        )
        print(
            "[debug:trajectory] ranges="
            + " ".join(
                f"{field}={value:.6f}" for field, value in zip(
                    JOINT_FIELDS if args.joint_level else EEF_FIELDS, ranges
                )
            )
        )
        print(
            f"[debug:config] firmware={args.firmware!r} interface={args.interface!r} "
            f"channel={args.channel!r} installation_pos={args.installation_pos!r} "
            f"payload={args.payload!r}"
        )

    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
        installation_pos=args.installation_pos,
        payload=args.payload,
    )
    try:
        api.connect()
        print(f"Piper connected: {api.is_ok()} (receive FPS: {api.get_fps():.1f})")
        if not api.is_ok():
            raise RuntimeError("Piper is not responding on the CAN bus")
        if debug:
            print(
                f"[debug:connected] software_version="
                f"{api.get_software_version(timeout_s=2.0)!r}"
            )
            log_debug_state(api, "connected")
        if not api.enable():
            raise RuntimeError("Timed out while enabling Piper")
        if debug:
            log_debug_state(api, "enabled")
        api.set_position_mode()
        time.sleep(POSITION_MODE_SETTLE_S)
        if debug:
            log_debug_state(api, "position-mode")

        if not args.no_preposition:
            print("Moving to the first recorded pose...")
            api.set_speed_percent(args.preposition_speed_percent)
            if args.joint_level:
                preposition_joints_linearly(
                    api,
                    poses[0],
                    duration_s=args.preposition_seconds,
                    max_step_rad=args.max_joint_step_rad,
                )
            else:
                deadline = time.monotonic() + args.preposition_seconds
                next_debug_t = time.monotonic()
                while time.monotonic() < deadline:
                    api.move_eef(poses[0])
                    now = time.monotonic()
                    if debug and now >= next_debug_t:
                        log_debug_state(api, "preposition", target=poses[0])
                        next_debug_t = now + args.debug_interval_s
                    time.sleep(0.05)
            if args.settle_seconds:
                time.sleep(args.settle_seconds)
            if debug and not args.joint_level:
                log_debug_state(api, "preposition-complete", target=poses[0])

        if args.joint_level:
            verify_joint_start(api, poses[0])

        api.set_speed_percent(args.replay_speed_percent)
        period_s = 1.0 / args.rate_hz
        duration_s = end_s - start_s
        wall_start = time.monotonic()
        tick = 0
        next_debug_elapsed_s = 0.0
        print(
            f"Replaying {path} ({duration_s:.3f}s) at {args.rate_hz:.1f} Hz. "
            "Press Ctrl-C to stop."
        )
        while True:
            elapsed_s = min(tick * period_s, duration_s)
            target = sample_pose(times, poses, start_s + elapsed_s)
            if args.joint_level:
                command_bounded_joint_step(api, target, args.max_joint_step_rad)
            else:
                api.move_eef(target)
            if debug and not args.joint_level and elapsed_s + 1e-12 >= next_debug_elapsed_s:
                log_debug_state(api, f"replay-{elapsed_s:.3f}s", target=target)
                next_debug_elapsed_s += args.debug_interval_s
            if elapsed_s >= duration_s:
                break
            tick += 1
            sleep_s = wall_start + tick * period_s - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
        if debug and not args.joint_level:
            log_debug_state(api, "replay-complete", target=poses[-1])
        print("Replay complete.")
        return 0
    except KeyboardInterrupt:
        print("\nReplay interrupted.")
        return 130
    finally:
        api.disconnect()


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
