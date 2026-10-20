from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from midas_hand_api.actuators import control_table as ct

from .config import HandOnlyCollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .hand_common import load_hand_config, verify_connected_hand
from .shared_state import SharedState


def force_feedback_current_ma(
    position_error_rad: float,
    *,
    max_current_ma: float,
    full_scale_error_rad: float,
) -> float:
    """Map follower contact error to an opposing, capped leader current."""

    current_ma = (
        -float(position_error_rad)
        * max_current_ma
        / full_scale_error_rad
    )
    return float(
        np.clip(
            current_ma,
            -max_current_ma,
            max_current_ma,
        )
    )


class TrimorphForceFeedbackController:
    """Keep TRIMORPH joints in current mode and batch their feedback commands."""

    def __init__(
        self,
        hand: Any,
        *,
        feedback_motor_ids: Sequence[int],
        max_current_ma: float,
        full_scale_error_rad: float,
        tactile_current_ma: float,
        tactile_motor_ids: Sequence[Sequence[int]],
        feedback_source_motor_ids: Mapping[int, int] | None = None,
        disabled_motor_ids: Sequence[int] = (),
    ) -> None:
        self.hand = hand
        self.client = hand.dxl_client
        if self.client is None:
            raise RuntimeError("TRIMORPH Dynamixel client is not connected")
        self._motor_indices = {
            int(motor_id): index for index, motor_id in enumerate(hand.motor_ids)
        }
        tactile_motor_ids = tuple(
            tuple(int(motor_id) for motor_id in digit_ids)
            for digit_ids in tactile_motor_ids
        )
        if len(tactile_motor_ids) != 4:
            raise ValueError("Tactile feedback mapping must contain four digits")
        self._tactile_motor_ids = tactile_motor_ids
        eligible_motor_ids = dict.fromkeys(
            int(motor_id) for motor_id in feedback_motor_ids
        )
        for digit_ids in tactile_motor_ids:
            eligible_motor_ids.update((motor_id, None) for motor_id in digit_ids)
        disabled_motor_ids = frozenset(
            int(motor_id) for motor_id in disabled_motor_ids
        )
        self._disabled_motor_ids = tuple(
            motor_id
            for motor_id in self._motor_indices
            if motor_id in disabled_motor_ids
        )
        self._feedback_motor_ids = tuple(
            motor_id
            for motor_id in eligible_motor_ids
            if motor_id in self._motor_indices and motor_id not in disabled_motor_ids
        )
        self._feedback_source_motor_ids = {
            int(trimorph_id): int(midas_id)
            for trimorph_id, midas_id in (feedback_source_motor_ids or {}).items()
        }
        self._current_mode_configured = False
        self._last_raw_currents: tuple[int, ...] | None = None
        self._latest_positions: np.ndarray | None = None

        self._max_current_ma = float(max_current_ma)
        self._full_scale_error_rad = float(full_scale_error_rad)
        self._tactile_current_ma = float(tactile_current_ma)
        if not math.isfinite(self._max_current_ma) or self._max_current_ma <= 0.0:
            raise ValueError(
                "Force-feedback maximum current must be finite and positive"
            )
        if (
            not math.isfinite(self._full_scale_error_rad)
            or self._full_scale_error_rad <= 0.0
        ):
            raise ValueError(
                "Force-feedback full-scale error must be finite and positive"
            )
        if not math.isfinite(self._tactile_current_ma):
            raise ValueError("Tactile force-feedback current must be finite")
        if abs(self._tactile_current_ma) > self._max_current_ma:
            raise ValueError(
                "Tactile force-feedback current cannot exceed the maximum current"
            )

        self._joint_signs = np.asarray(
            hand.config.joint_signs_array, dtype=np.float64
        )
        self._home_offsets = np.asarray(
            hand.config.home_offsets_array, dtype=np.float64
        )
        self._current_unit_ma = float(hand.config.current_unit_ma)
        if self._current_unit_ma <= 0.0:
            raise ValueError("TRIMORPH current unit must be positive")
        if self._max_current_ma > float(hand.config.max_current_limit_ma):
            raise ValueError(
                f"Force-feedback current {self._max_current_ma} mA "
                f"exceeds configured motor limit {hand.config.max_current_limit_ma} mA"
            )

    def start(self, positions: Sequence[float]) -> None:
        """Configure all feedback joints once and leave them in current mode."""

        self._set_latest_positions(positions)
        if self._disabled_motor_ids:
            self.hand.disable_torque(list(self._disabled_motor_ids))
        motor_ids = list(self._feedback_motor_ids)
        if not motor_ids:
            return
        self.hand.disable_torque(motor_ids)
        self.client.sync_write(
            motor_ids,
            [ct.OPERATING_MODE_CURRENT] * len(motor_ids),
            ct.ADDR_OPERATING_MODE,
            ct.LEN_OPERATING_MODE,
        )
        self._current_mode_configured = True
        self._write_raw_currents([0] * len(motor_ids), force=True)
        self.hand.enable_torque(motor_ids)
        print(f"TRIMORPH force feedback ready for motors {motor_ids}")

    def update(
        self,
        feedback_by_motor_id: Mapping[int, float],
        positions: Sequence[float],
        tactile_contacts: Sequence[bool] | None = None,
    ) -> None:
        self._set_latest_positions(positions)
        if self._feedback_motor_ids and not self._current_mode_configured:
            raise RuntimeError("Force feedback must be started before update")

        tactile_feedback_ids: set[int] = set()
        if tactile_contacts is not None:
            if len(tactile_contacts) != len(self._tactile_motor_ids):
                raise ValueError("Tactile contacts must contain four digits")
            tactile_feedback_ids = {
                motor_id
                for contact, digit_ids in zip(
                    tactile_contacts, self._tactile_motor_ids
                )
                if contact
                for motor_id in digit_ids
            }

        calibrated_currents: list[float] = []
        for motor_id in self._feedback_motor_ids:
            if motor_id in tactile_feedback_ids:
                current_ma = self._tactile_current_ma
            else:
                source_motor_id = self._feedback_source_motor_ids.get(
                    motor_id, motor_id
                )
                current_ma = force_feedback_current_ma(
                    feedback_by_motor_id.get(source_motor_id, 0.0),
                    max_current_ma=self._max_current_ma,
                    full_scale_error_rad=self._full_scale_error_rad,
                )
            calibrated_currents.append(current_ma)
        self._write_calibrated_currents(calibrated_currents)

    def shutdown(self) -> None:
        """Make all touched joints torque-free and restore operating mode 5."""

        if not self._current_mode_configured:
            return
        motor_ids = list(self._feedback_motor_ids)
        errors: list[Exception] = []
        try:
            self._write_raw_currents([0] * len(motor_ids), force=True)
        except Exception as exc:
            errors.append(exc)
        try:
            self.hand.disable_torque(motor_ids)
        except Exception as exc:
            errors.append(exc)
        try:
            self.client.sync_write(
                motor_ids,
                [ct.OPERATING_MODE_CURRENT_BASED_POSITION] * len(motor_ids),
                ct.ADDR_OPERATING_MODE,
                ct.LEN_OPERATING_MODE,
            )
            self._current_mode_configured = False
        except Exception as exc:
            errors.append(exc)
        if self._latest_positions is not None:
            indices = [self._motor_indices[motor_id] for motor_id in motor_ids]
            raw_positions = (
                self._latest_positions[indices] * self._joint_signs[indices]
                + self._home_offsets[indices]
            )
            try:
                self.client.write_desired_pos(motor_ids, raw_positions)
            except Exception as exc:
                errors.append(exc)
        try:
            current_limit_raw = int(self._max_current_ma / self._current_unit_ma)
            self.client.sync_write(
                motor_ids,
                [current_limit_raw] * len(motor_ids),
                ct.ADDR_GOAL_CURRENT,
                ct.LEN_GOAL_CURRENT,
            )
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError(
                "Failed to fully restore force-feedback motors"
            ) from errors[0]
        print(f"TRIMORPH force feedback restored for motors {motor_ids}")

    def _set_latest_positions(self, positions: Sequence[float]) -> None:
        positions_array = np.asarray(positions, dtype=np.float64)
        if positions_array.shape != (len(self._motor_indices),):
            raise ValueError("Leader positions do not align with leader motor IDs")
        if not np.isfinite(positions_array).all():
            raise ValueError("Leader positions must be finite")
        self._latest_positions = positions_array.copy()

    def _write_calibrated_currents(self, currents_ma: Sequence[float]) -> None:
        raw_currents = []
        for motor_id, current_ma in zip(self._feedback_motor_ids, currents_ma):
            index = self._motor_indices[motor_id]
            raw_current = (
                float(current_ma)
                * self._joint_signs[index]
                / self._current_unit_ma
            )
            raw_currents.append(int(raw_current))
        self._write_raw_currents(raw_currents)

    def _write_raw_currents(
        self, raw_currents: Sequence[int], *, force: bool = False
    ) -> None:
        values = tuple(int(value) for value in raw_currents)
        if not force and values == self._last_raw_currents:
            return
        self.client.sync_write(
            self._feedback_motor_ids,
            values,
            ct.ADDR_GOAL_CURRENT,
            ct.LEN_GOAL_CURRENT,
        )
        self._last_raw_currents = values


def run_leader_hand_worker(
    config: HandOnlyCollectionConfig,
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
    force_feedback = TrimorphForceFeedbackController(
        hand,
        feedback_motor_ids=config.force_feedback_motor_ids,
        max_current_ma=config.force_feedback_max_current_ma,
        full_scale_error_rad=config.force_feedback_full_scale_error_rad,
        tactile_current_ma=config.tactile_force_feedback_current_ma,
        tactile_motor_ids=config.tactile_force_feedback_motor_ids,
        feedback_source_motor_ids=getattr(
            config, "force_feedback_source_motor_ids", None
        ),
        disabled_motor_ids=getattr(
            config, "disabled_force_feedback_motor_ids", ()
        ),
    )
    period_ns = int(round(1_000_000_000 / config.hand_measure_rate_hz))
    feedback_period_ns = int(
        round(1_000_000_000 / config.force_feedback_rate_hz)
    )
    next_read_ns = time.monotonic_ns()
    next_feedback_ns = next_read_ns
    sample_idx = 0
    sample_session: EpisodeBuffer | None = None
    try:
        initial_positions = np.asarray(hand.read_pos(), dtype=np.float64)
        force_feedback.start(initial_positions)
        print(
            f"TRIMORPH hand connected on {hand.port}; "
            f"model={hand.config.motor_model}; baud={hand.config.baudrate}; "
            f"motor_ids={hand.motor_ids}"
        )
        init_done.set()
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
            if sample_ns >= next_feedback_ns:
                force_feedback.update(
                    shared.latest_follower_position_error_by_motor_id(),
                    positions,
                    shared.latest_tactile_contacts(),
                )
                next_feedback_ns += feedback_period_ns
                while next_feedback_ns < sample_ns:
                    next_feedback_ns += feedback_period_ns

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
