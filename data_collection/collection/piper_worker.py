from __future__ import annotations

import time

from .config import RAW_EEF_FIELDS, CollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .shared_state import SharedState


PIPER_LIMIT_WARNING_INTERVAL_S = 2.0
PIPER_KINEMATIC_LIMIT_STATUSES = (
    "TARGET_POS_EXCEEDS_LIMIT",
    "NO_SOLUTION",
    "SINGULARITY_POINT",
)


def piper_workspace_warning(state: dict[str, object]) -> str | None:
    """Describe Piper workspace/joint-limit feedback, if present."""

    arm_status = str(state.get("arm_status", ""))
    reasons = [
        f"arm_status={arm_status}"
        for status in PIPER_KINEMATIC_LIMIT_STATUSES
        if status in arm_status
    ]
    try:
        err_code = int(state.get("err_code", 0))
    except (TypeError, ValueError):
        err_code = 0
    limited_joints = [
        joint for joint in range(1, 7) if err_code & (1 << (joint + 7))
    ]
    if limited_joints:
        reasons.append(f"joint_angle_limit={limited_joints}")
    if not reasons:
        return None
    return (
        "WARNING: Piper reached a workspace/kinematic limit ("
        + ", ".join(reasons)
        + "). Move the arm away from the boundary before recording/replay."
    )


def run_piper_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    *,
    init_done,
) -> None:
    """Passive Piper logger: no MIT/backdrive/cuRobo control, ever.

    The arm is expected to be moved with its own built-in drag-teach button.
    This worker only connects, polls teach_status + EEF state, and logs —
    it never calls enable()/set_mit_mode()/sends a motion command.
    """
    from scripts.piper_api.piper_api import PiperAPI

    api = PiperAPI(
        firmware=config.piper_firmware,
        interface=config.piper_interface,
        channel=config.piper_channel,
        installation_pos=config.piper_installation_pos,
        payload=config.piper_payload,
    )
    api.connect()
    print("Piper connected:", api.is_ok(), "receive FPS:", api.get_fps())
    if not api.is_ok():
        raise RuntimeError(
            "Piper arm is not responding on the CAN bus (receive FPS=0). "
            "Check that the arm is powered on and the CAN cable is connected."
        )
    print(
        "[piper_worker] Passive logging only — use the arm's built-in "
        "drag-teach button to move it."
    )

    init_done.set()

    sample_period_ns = int(round(1_000_000_000 / config.piper_rate_hz))
    teacher_period_ns = int(round(1_000_000_000 / config.teacher_mode_poll_hz))
    now_ns = time.monotonic_ns()
    next_sample_ns = now_ns
    next_teacher_ns = now_ns
    sample_idx = 0
    sample_session: EpisodeBuffer | None = None
    teach_status: object = None
    last_limit_warning: str | None = None
    last_limit_warning_t = float("-inf")
    try:
        while not shared.shutdown.is_set():
            now_ns = time.monotonic_ns()

            if now_ns >= next_teacher_ns:
                state = api.get_state()
                teach_status = state.get("teach_status")
                limit_warning = piper_workspace_warning(state)
                now_s = now_ns / 1e9
                if limit_warning is not None and (
                    limit_warning != last_limit_warning
                    or now_s - last_limit_warning_t >= PIPER_LIMIT_WARNING_INTERVAL_S
                ):
                    print(limit_warning, flush=True)
                    last_limit_warning = limit_warning
                    last_limit_warning_t = now_s
                elif limit_warning is None:
                    last_limit_warning = None
                next_teacher_ns += teacher_period_ns
                while next_teacher_ns < now_ns:
                    next_teacher_ns += teacher_period_ns

            if now_ns >= next_sample_ns:
                eef = api.get_eef_state()
                sample_ns = time.monotonic_ns()
                session = recorder.current()
                if session is not None and eef is not None:
                    if session is not sample_session:
                        sample_session = session
                        sample_idx = 0
                    row: dict[str, object] = {
                        "sample_idx": sample_idx,
                        "monotonic_ns": sample_ns,
                        "wall_time_s": f"{time.time():.6f}",
                        "elapsed_s": f"{session.elapsed_s(sample_ns):.9f}",
                        "robot_timestamp_s": f"{eef.timestamp:.6f}",
                        "robot_hz": f"{eef.hz:.3f}",
                        "teach_status": str(teach_status),
                    }
                    row.update(
                        {
                            name: f"{float(value):.9f}"
                            for name, value in zip(RAW_EEF_FIELDS, eef.pose)
                        }
                    )
                    session.append_piper(row)
                    sample_idx += 1
                next_sample_ns += sample_period_ns
                while next_sample_ns < sample_ns:
                    next_sample_ns += sample_period_ns

            next_wake_ns = min(next_sample_ns, next_teacher_ns)
            wait_s = (next_wake_ns - time.monotonic_ns()) / 1e9
            if wait_s > 0:
                shared.shutdown.wait(min(wait_s, 0.01))
    finally:
        api.disconnect()
