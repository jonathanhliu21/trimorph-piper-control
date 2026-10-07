from __future__ import annotations

import time

import numpy as np

from .contact_detection import (
    CONTACT_ENTER_THRESHOLD,
    CONTACT_EXIT_THRESHOLD,
    CONTACT_MOTOR_IDS,
    CONTACT_THRESHOLD,
    ContactStateMachine,
    contact_feedback,
)
from .config import (
    FOLLOWER_PROFILE_ACCELERATION_RAD_S2,
    FOLLOWER_PROFILE_VELOCITY_RAD_S,
    CollectionConfig,
    joint_command_scales_for_motor_ids,
)
from .episode import EpisodeBuffer, RecordingManager
from .hand_common import load_follower_config, verify_connected_hand
from .shared_state import SharedState
from .thumb_retargeting import (
    build_follower_target,
    load_thumb_retarget_model,
)


def adjusted_follower_position_error(
    motor_ids,
    target_positions,
    measured_positions,
) -> np.ndarray:
    """Return exit-deadband error for motors 4 and 5 in follower order."""

    motor_ids = tuple(int(motor_id) for motor_id in motor_ids)
    targets = np.asarray(target_positions, dtype=np.float64)
    positions = np.asarray(measured_positions, dtype=np.float64)
    expected_shape = (len(motor_ids),)
    if targets.shape != expected_shape or positions.shape != expected_shape:
        raise ValueError("Follower contact vectors must match the motor ID count")

    adjusted_error = np.zeros(expected_shape, dtype=np.float64)
    contact_indices = [
        index
        for index, motor_id in enumerate(motor_ids)
        if motor_id in CONTACT_MOTOR_IDS
    ]
    if contact_indices:
        position_errors = targets - positions
        adjusted_error[contact_indices] = contact_feedback(
            position_errors[contact_indices],
            thresholds=CONTACT_EXIT_THRESHOLD,
        )
    return adjusted_error


def follower_contact_detection_ready(
    motor_ids,
    target_positions,
    measured_positions,
) -> bool:
    """Return whether motors 4 and 5 have reached the initial target."""

    motor_ids = tuple(int(motor_id) for motor_id in motor_ids)
    targets = np.asarray(target_positions, dtype=np.float64)
    positions = np.asarray(measured_positions, dtype=np.float64)
    expected_shape = (len(motor_ids),)
    if targets.shape != expected_shape or positions.shape != expected_shape:
        raise ValueError("Follower target and position must match the motor ID count")
    contact_indices = [
        index
        for index, motor_id in enumerate(motor_ids)
        if motor_id in CONTACT_MOTOR_IDS
    ]
    return not contact_indices or bool(
        np.all(
            np.abs(targets[contact_indices] - positions[contact_indices])
            <= CONTACT_THRESHOLD
        )
    )


def prepare_follower_hand(hand, *, configure_hardware: bool) -> None:
    if configure_hardware:
        hand.configure(enable_torque=False)
    hand.set_motion_profile(
        profile_velocity_rad_s=FOLLOWER_PROFILE_VELOCITY_RAD_S,
        profile_acceleration_rad_s2=FOLLOWER_PROFILE_ACCELERATION_RAD_S2,
    )


def run_follower_hand_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    *,
    init_done,
) -> None:
    from midas_hand_api import MidasHand

    if config.follower_hand_config is None or config.follower_hand_port is None:
        raise RuntimeError("Follower worker started without follower configuration")
    hand_config = load_follower_config(
        config.follower_hand_config,
        config.follower_hand_port,
        config.active_motors,
    )
    joint_command_scales = np.asarray(
        joint_command_scales_for_motor_ids(
            config.follower_joint_command_scales, hand_config.motor_ids
        ),
        dtype=np.float64,
    )
    if not np.isfinite(config.follower_command_scale) or not np.isfinite(
        joint_command_scales
    ).all():
        raise ValueError("Follower command scales must be finite")
    thumb_model_path = getattr(config, "thumb_retarget_model", None)
    thumb_model = (
        load_thumb_retarget_model(thumb_model_path)
        if thumb_model_path is not None
        else None
    )
    hand = MidasHand(config=hand_config, autoconnect=True)
    verify_connected_hand(hand, role="follower")
    prepare_follower_hand(
        hand, configure_hardware=config.follower_configure_hardware
    )
    last_command = hand.clip_positions(hand.read_pos())
    hand.set_positions(last_command, clip=True)
    hand.enable_torque()
    print(
        f"Follower MIDAS connected on {hand.port}; model={hand.config.motor_model}; "
        f"baud={hand.config.baudrate}; motor_ids={hand.motor_ids}"
    )
    init_done.set()

    read_period_ns = int(round(1_000_000_000 / config.follower_measure_rate_hz))
    command_period_ns = int(round(1_000_000_000 / config.follower_command_rate_hz))
    next_read_ns = time.monotonic_ns()
    next_command_ns = next_read_ns
    commanded_velocities = np.zeros_like(last_command)
    latest_desired_target: np.ndarray | None = None
    contact_detection_armed = False
    contact_indices = [
        index
        for index, motor_id in enumerate(hand.motor_ids)
        if int(motor_id) in CONTACT_MOTOR_IDS
    ]
    contact_detectors = {
        index: ContactStateMachine() for index in contact_indices
    }
    sample_idx = 0
    sample_session: EpisodeBuffer | None = None
    try:
        while not shared.shutdown.is_set():
            now_ns = time.monotonic_ns()
            if now_ns >= next_command_ns:
                target = shared.latest_hand_target()
                commanded_velocities.fill(0.0)
                if target is not None:
                    if target.shape != (len(hand.motor_ids),) or not np.isfinite(
                        target
                    ).all():
                        raise RuntimeError(
                            f"Primary hand target has shape {target.shape}, but "
                            "follower "
                            f"expects ({len(hand.motor_ids)},)"
                        )
                    target = hand.clip_positions(
                        build_follower_target(
                            target,
                            hand.motor_ids,
                            command_scale=config.follower_command_scale,
                            joint_command_scales=joint_command_scales,
                            thumb_model=thumb_model,
                        )
                    )
                    latest_desired_target = np.array(target, copy=True)
                    delta = target - last_command
                    delta *= config.follower_interpolation_alpha
                    if config.follower_max_step_rad > 0:
                        delta = np.clip(
                            delta,
                            -config.follower_max_step_rad,
                            config.follower_max_step_rad,
                        )
                    next_command = hand.clip_positions(last_command + delta)
                    commanded_velocities = (
                        next_command - last_command
                    ) / (command_period_ns / 1e9)
                    last_command = next_command
                    hand.set_positions(last_command, clip=True)
                next_command_ns += command_period_ns
                while next_command_ns < now_ns:
                    next_command_ns += command_period_ns

            if now_ns >= next_read_ns:
                positions, measured_velocities, measured_currents = (
                    hand.read_pos_vel_cur()
                )
                positions = np.asarray(positions, dtype=np.float64)
                measured_velocities = np.asarray(
                    measured_velocities,
                    dtype=np.float64,
                )
                measured_currents = np.asarray(measured_currents, dtype=np.float64)
                sample_ns = time.monotonic_ns()
                if positions.shape != (len(hand.motor_ids),) or not np.isfinite(
                    positions
                ).all():
                    raise RuntimeError("Follower read returned an invalid motor vector")
                if measured_velocities.shape != (
                    len(hand.motor_ids),
                ) or not np.isfinite(measured_velocities).all():
                    raise RuntimeError(
                        "Follower velocity read returned an invalid motor vector"
                    )
                if measured_currents.shape != (
                    len(hand.motor_ids),
                ) or not np.isfinite(measured_currents).all():
                    raise RuntimeError(
                        "Follower current read returned an invalid motor vector"
                    )
                if (
                    not contact_detection_armed
                    and latest_desired_target is not None
                    and follower_contact_detection_ready(
                        hand.motor_ids,
                        latest_desired_target,
                        positions,
                    )
                ):
                    contact_detection_armed = True
                    print(
                        "Follower reached its initial target; contact detection armed"
                    )
                candidate_error = (
                    adjusted_follower_position_error(
                        hand.motor_ids,
                        last_command,
                        positions,
                    )
                    if contact_detection_armed
                    else np.zeros_like(positions)
                )
                adjusted_error = np.zeros_like(candidate_error)
                if contact_detection_armed:
                    sample_time_s = sample_ns / 1e9
                    for index, detector in contact_detectors.items():
                        adjusted_error[index] = detector.update(
                            commanded_velocity_rad_s=float(
                                commanded_velocities[index]
                            ),
                            measured_velocity_rad_s=float(
                                measured_velocities[index]
                            ),
                            measured_current_ma=float(measured_currents[index]),
                            candidate_feedback=float(candidate_error[index]),
                            now_s=sample_time_s,
                        )
                shared.set_follower_position_error(
                    adjusted_error,
                    motor_ids=hand.motor_ids,
                )
                normalized_joint_difference = (
                    np.abs(last_command - positions) / CONTACT_ENTER_THRESHOLD
                )
                shared.set_follower_contact_snapshot(
                    hand.motor_ids,
                    adjusted_error,
                    normalized_joint_difference,
                )
                if contact_indices and np.any(
                    adjusted_error[contact_indices] != 0.0
                ):
                    formatted = "[{}]".format(
                        " ".join(
                            f"{float(adjusted_error[index]):8.3f}"
                            for index in contact_indices
                        )
                    )
                    contact_motor_ids = [
                        int(hand.motor_ids[index]) for index in contact_indices
                    ]
                    print(
                        "Follower contact detected; adjusted position error "
                        f"for motor IDs {contact_motor_ids}: {formatted}"
                    )
                session = recorder.current()
                if session is not None:
                    if session is not sample_session:
                        sample_session = session
                        sample_idx = 0
                    row: dict[str, object] = {
                        "sample_idx": sample_idx,
                        "monotonic_ns": sample_ns,
                        "wall_time_s": f"{time.time():.6f}",
                        "elapsed_s": f"{session.elapsed_s(sample_ns):.9f}",
                    }
                    for motor_id, value in zip(hand.motor_ids, positions):
                        row[f"pos_{int(motor_id)}"] = f"{float(value):.9f}"
                    session.append_follower(row)
                    sample_idx += 1
                next_read_ns += read_period_ns
                while next_read_ns < sample_ns:
                    next_read_ns += read_period_ns

            next_due_ns = min(next_read_ns, next_command_ns)
            wait_s = (next_due_ns - time.monotonic_ns()) / 1e9
            shared.shutdown.wait(max(0.0, min(wait_s, 0.005)))
    finally:
        hand.shutdown()
