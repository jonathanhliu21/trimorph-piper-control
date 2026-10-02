from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from midas_hand_api.actuators import control_table as ct

from .config import CollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .hand_common import load_hand_config, verify_connected_hand
from .shared_state import SharedState


FORCE_FEEDBACK_MOTOR_IDS = (4, 5)
FORCE_FEEDBACK_MAX_CURRENT_MA = 100.0
FORCE_FEEDBACK_FULL_SCALE_ERROR_RAD = 0.01


def force_feedback_current_ma(position_error_rad: float) -> float:
    """Map follower contact error to an opposing, capped leader current."""

    current_ma = (
        -float(position_error_rad)
        * FORCE_FEEDBACK_MAX_CURRENT_MA
        / FORCE_FEEDBACK_FULL_SCALE_ERROR_RAD
    )
    return float(
        np.clip(
            current_ma,
            -FORCE_FEEDBACK_MAX_CURRENT_MA,
            FORCE_FEEDBACK_MAX_CURRENT_MA,
        )
    )


class TrimorphForceFeedbackController:
    """Drive selected TRIMORPH joints in current mode while contact is active."""

    def __init__(self, hand: Any) -> None:
        self.hand = hand
        self.client = hand.dxl_client
        if self.client is None:
            raise RuntimeError("TRIMORPH Dynamixel client is not connected")
        self._motor_indices = {
            int(motor_id): index for index, motor_id in enumerate(hand.motor_ids)
        }
        self._feedback_motor_ids = tuple(
            motor_id
            for motor_id in FORCE_FEEDBACK_MOTOR_IDS
            if motor_id in self._motor_indices
        )
        self._current_mode_ids: set[int] = set()
        self._latest_positions: np.ndarray | None = None

        self._joint_signs = np.asarray(
            hand.config.joint_signs_array, dtype=np.float64
        )
        self._home_offsets = np.asarray(
            hand.config.home_offsets_array, dtype=np.float64
        )
        self._current_unit_ma = float(hand.config.current_unit_ma)
        if self._current_unit_ma <= 0.0:
            raise ValueError("TRIMORPH current unit must be positive")
        if FORCE_FEEDBACK_MAX_CURRENT_MA > float(
            hand.config.max_current_limit_ma
        ):
            raise ValueError(
                f"Force-feedback current {FORCE_FEEDBACK_MAX_CURRENT_MA} mA "
                f"exceeds configured motor limit {hand.config.max_current_limit_ma} mA"
            )

    def update(
        self,
        feedback_by_motor_id: Mapping[int, float],
        positions: Sequence[float],
    ) -> None:
        positions_array = np.asarray(positions, dtype=np.float64)
        if positions_array.shape != (len(self._motor_indices),):
            raise ValueError("Leader positions do not align with leader motor IDs")
        self._latest_positions = positions_array.copy()

        for motor_id in self._feedback_motor_ids:
            current_ma = force_feedback_current_ma(
                feedback_by_motor_id.get(motor_id, 0.0)
            )
            if current_ma == 0.0:
                if motor_id in self._current_mode_ids:
                    self._restore_current_based_position_mode(motor_id)
                continue

            if motor_id not in self._current_mode_ids:
                self._enter_current_mode(motor_id)
            self._write_current_ma(motor_id, current_ma)

    def shutdown(self) -> None:
        """Make all touched joints torque-free and restore operating mode 5."""

        errors: list[Exception] = []
        for motor_id in tuple(self._current_mode_ids):
            try:
                self._restore_current_based_position_mode(motor_id)
            except Exception as exc:  # preserve cleanup attempts for other joints
                errors.append(exc)
        if errors:
            raise RuntimeError(
                "Failed to restore one or more force-feedback joints"
            ) from errors[0]

    def _enter_current_mode(self, motor_id: int) -> None:
        self.hand.disable_torque([motor_id])
        # Mark it before the register write so shutdown still attempts a safe
        # restoration if communication fails after the mode may have changed.
        self._current_mode_ids.add(motor_id)
        self.client.sync_write(
            [motor_id],
            [ct.OPERATING_MODE_CURRENT],
            ct.ADDR_OPERATING_MODE,
            ct.LEN_OPERATING_MODE,
        )
        self._write_current_ma(motor_id, 0.0)
        self.hand.enable_torque([motor_id])
        print(f"TRIMORPH force feedback engaged for motor {motor_id}")

    def _restore_current_based_position_mode(self, motor_id: int) -> None:
        errors: list[Exception] = []
        try:
            self._write_current_ma(motor_id, 0.0)
        except Exception as exc:
            errors.append(exc)

        torque_disabled = False
        try:
            self.hand.disable_torque([motor_id])
            torque_disabled = True
        except Exception as exc:
            errors.append(exc)

        mode_restored = False
        if torque_disabled:
            try:
                self.client.sync_write(
                    [motor_id],
                    [ct.OPERATING_MODE_CURRENT_BASED_POSITION],
                    ct.ADDR_OPERATING_MODE,
                    ct.LEN_OPERATING_MODE,
                )
                mode_restored = True
            except Exception as exc:
                errors.append(exc)

        # Seed the position goal from the current calibrated joint position so a
        # later torque enable cannot pull the finger toward a stale target.
        if self._latest_positions is None:
            raise RuntimeError("Cannot restore force-feedback mode without a position")
        index = self._motor_indices[motor_id]
        raw_position = (
            self._latest_positions[index] * self._joint_signs[index]
            + self._home_offsets[index]
        )
        if mode_restored:
            try:
                self.client.write_desired_pos([motor_id], [raw_position])
            except Exception as exc:
                errors.append(exc)
        current_limit_raw = int(
            FORCE_FEEDBACK_MAX_CURRENT_MA / self._current_unit_ma
        )
        if mode_restored:
            try:
                self.client.sync_write(
                    [motor_id],
                    [current_limit_raw],
                    ct.ADDR_GOAL_CURRENT,
                    ct.LEN_GOAL_CURRENT,
                )
            except Exception as exc:
                errors.append(exc)

        if mode_restored:
            self._current_mode_ids.discard(motor_id)
        if errors:
            raise RuntimeError(
                f"Failed to fully restore force-feedback motor {motor_id}"
            ) from errors[0]
        print(f"TRIMORPH force feedback released for motor {motor_id}")

    def _write_current_ma(self, motor_id: int, calibrated_current_ma: float) -> None:
        index = self._motor_indices[motor_id]
        raw_current = int(
            float(calibrated_current_ma)
            * self._joint_signs[index]
            / self._current_unit_ma
        )
        self.client.sync_write(
            [motor_id],
            [raw_current],
            ct.ADDR_GOAL_CURRENT,
            ct.LEN_GOAL_CURRENT,
        )


def run_leader_hand_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    *,
    init_done,
) -> None:
    from midas_hand_api import MidasHand

    hand_config = load_hand_config(
        config.hand_config, config.hand_port, config.active_motors
    )
    hand = MidasHand(config=hand_config, autoconnect=True)
    verify_connected_hand(hand, role="primary")
    hand.disable_torque()
    force_feedback = TrimorphForceFeedbackController(hand)
    print(
        f"TRIMORPH hand connected on {hand.port}; model={hand.config.motor_model}; "
        f"baud={hand.config.baudrate}; motor_ids={hand.motor_ids}"
    )
    init_done.set()

    period_ns = int(round(1_000_000_000 / config.hand_measure_rate_hz))
    next_read_ns = time.monotonic_ns()
    sample_idx = 0
    sample_session: EpisodeBuffer | None = None
    try:
        while not shared.shutdown.is_set():
            now_ns = time.monotonic_ns()
            if now_ns < next_read_ns:
                shared.shutdown.wait(min((next_read_ns - now_ns) / 1e9, 0.005))
                continue

            positions = np.asarray(hand.read_pos(), dtype=np.float64)
            sample_ns = time.monotonic_ns()
            if positions.shape != (len(hand.motor_ids),):
                raise RuntimeError(
                    f"Hand read returned shape {positions.shape}, "
                    f"expected ({len(hand.motor_ids)},)"
                )
            if not np.isfinite(positions).all():
                raise RuntimeError("Hand read returned NaN or Inf")
            shared.set_hand_target(positions)
            force_feedback.update(
                shared.latest_follower_position_error_by_motor_id(),
                positions,
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
                session.append_leader_hand(row)
                sample_idx += 1

            next_read_ns += period_ns
            while next_read_ns < sample_ns:
                next_read_ns += period_ns
    finally:
        try:
            force_feedback.shutdown()
        finally:
            try:
                hand.disable_torque()
            finally:
                hand.shutdown()
