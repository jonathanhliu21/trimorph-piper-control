"""Interactive hand controller and contact visualizer.

Run from any directory with the project environment::

    .venv/bin/python scripts/trimorph_api_test/hand_ui.py [--add-tactile]
        [--paxini-port /dev/cu.usbserial-XXXX]

The hardware is owned by one background thread. Tkinter never performs a bus
operation, so rendering and the Stop controls remain responsive.
"""

from __future__ import annotations

import argparse
import logging
import math
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Literal

import numpy as np
import yaml

# Keep direct script execution able to import the sibling data_collection
# package without loading the detector under a second module name.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_collection.collection.contact_detection import (
    CONTACT_ENTRY_FEEDBACK_THRESHOLD,
    CONTACT_EXIT_PERSISTENCE_TIME_S,
    CONTACT_EXIT_THRESHOLD,
    CONTACT_MAX_VELOCITY_RAD_S,
    CONTACT_MIN_ENGAGED_TIME_S,
    CONTACT_MOTION_CURRENT_THRESHOLD_MA,
    CONTACT_PERSISTENCE_TIME_S,
    CONTACT_SETTLE_TIME_S,
    CONTACT_VIRTUAL_SPRING_STIFFNESS,
    ContactState,
    ContactStateMachine,
    contact_feedback,
)
from data_collection.collection.hand_common import is_compatible_dynamixel_model
from data_collection.collection.tactile_contact_detection import (
    TactileContactDetector,
)
from midas_hand_api import HandConfig, MidasHand, PaxiniConfig, PaxiniHandSensor


DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config" / "midas_hand_config.yaml"
# Physical finger tuples are PIP, MCP, AB/AD, matching the supplied diagram.
# Control arrays use MCP, PIP, AB/AD within each finger; the mapping below keeps
# that internal order independent of the configuration's motor order.
FINGER_MOTOR_IDS = ((4, 5, 6), (7, 8, 9), (10, 11, 12))
FINGER_UI_MOTOR_IDS = tuple(
    motor_id
    for pip_id, mcp_id, ab_ad_id in FINGER_MOTOR_IDS
    for motor_id in (mcp_id, pip_id, ab_ad_id)
)
THUMB_MOTOR_IDS = (0, 1, 2, 3)  # IP, MCP, CMC side, CMC roll
UI_MOTOR_IDS = FINGER_UI_MOTOR_IDS + THUMB_MOTOR_IDS
JOINT_COUNT = len(UI_MOTOR_IDS)
FLEXION_INDICES = (0, 1, 3, 4, 6, 7)
AB_AD_INDICES = (2, 5, 8)
THUMB_INDICES = (9, 10, 11, 12)
CONTROL_RATE_HZ = 50.0
PROFILE_VELOCITY_RAD_S = 3.0
PROFILE_ACCELERATION_RAD_S2 = 50.0
DEFAULT_JOINT_TARGETS_RAD = (0.0,) * JOINT_COUNT
STREAM_OPEN_ANGLE_RAD = 0.0
STREAM_CLOSE_ANGLE_RAD = -1.35
THUMB_OPEN_ANGLES_RAD = (0.0, 0.0, 0.8, 0.0)
THUMB_CLOSE_ANGLES_RAD = (-1.4, -1.4, -0.1, 1.5)
STREAM_PREPOSITION_TOLERANCE_RAD = 0.1
STREAM_PREPOSITION_TIMEOUT_S = 4.0
MIN_STREAM_DURATION_S = 1.0 / CONTROL_RATE_HZ
TACTILE_SENSOR_NAMES = ("index", "middle", "ring", "thumb")
TACTILE_CONTACT_LOGGER_NAME = (
    "data_collection.collection.tactile_contact_detection"
)


def _configure_tactile_debug_log(path: Path) -> None:
    log_path = path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%H:%M:%S")
    )
    tactile_logger = logging.getLogger(TACTILE_CONTACT_LOGGER_NAME)
    tactile_logger.setLevel(logging.DEBUG)
    tactile_logger.addHandler(handler)
    tactile_logger.propagate = False
    tactile_logger.info("TACTILE_CONTACT event=log_started path=%s", log_path)


@dataclass(frozen=True)
class ConnectionSettings:
    config_path: Path


@dataclass(frozen=True)
class ContactDetectorSettings:
    max_velocity_rad_s: float = CONTACT_MAX_VELOCITY_RAD_S
    motion_current_threshold_ma: float = CONTACT_MOTION_CURRENT_THRESHOLD_MA
    settle_time_s: float = CONTACT_SETTLE_TIME_S
    persistence_time_s: float = CONTACT_PERSISTENCE_TIME_S
    entry_feedback_threshold: float = CONTACT_ENTRY_FEEDBACK_THRESHOLD
    min_engaged_time_s: float = CONTACT_MIN_ENGAGED_TIME_S
    exit_persistence_time_s: float = CONTACT_EXIT_PERSISTENCE_TIME_S


@dataclass(frozen=True)
class Telemetry:
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    currents_ma: tuple[float, ...]
    targets: tuple[float, ...]
    errors: tuple[float, ...]
    candidate_feedback: tuple[float, ...]
    contact_feedback: tuple[float, ...]
    normalized_joint_difference: tuple[float, ...]
    contacts: tuple[bool, ...]
    tactile_contacts: tuple[bool, ...] | None
    sequence_active: bool
    sequence_kind: Literal["fingers", "thumb"] | None


@dataclass(frozen=True)
class WorkerEvent:
    kind: Literal["status", "connected", "telemetry", "disconnected", "error"]
    payload: object = None


def _validated_default_targets(
    lower_limits: tuple[float, ...] | list[float],
    upper_limits: tuple[float, ...] | list[float],
) -> np.ndarray:
    """Return the zero defaults after checking both configured joint ranges."""

    lower = np.asarray(lower_limits, dtype=float)
    upper = np.asarray(upper_limits, dtype=float)
    targets = np.asarray(DEFAULT_JOINT_TARGETS_RAD, dtype=float)
    if lower.shape != targets.shape or upper.shape != targets.shape:
        raise ValueError(
            f"Expected exactly {JOINT_COUNT} configured joint limits for the hand"
        )
    invalid = [
        name
        for name, target, minimum, maximum in zip(
            (
                "Finger 1 MCP", "Finger 1 PIP", "Finger 1 AB/AD",
                "Finger 2 MCP", "Finger 2 PIP", "Finger 2 AB/AD",
                "Finger 3 MCP", "Finger 3 PIP", "Finger 3 AB/AD",
                "Thumb IP", "Thumb MCP", "Thumb CMC side", "Thumb CMC roll",
            ),
            targets,
            lower,
            upper,
        )
        if not minimum <= target <= maximum
    ]
    if invalid:
        raise ValueError(
            f"Default 0 rad target is outside the configured limits for {invalid}"
        )
    return targets


def _ui_order_indices(
    motor_ids: tuple[int, ...] | list[int],
    *requested_ids: int,
) -> tuple[int, ...]:
    """Map requested UI motor order to the configured motor order."""

    if not requested_ids:
        requested_ids = UI_MOTOR_IDS
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("All requested motor IDs must be different")
    try:
        return tuple(motor_ids.index(motor_id) for motor_id in requested_ids)
    except ValueError as exc:
        raise ValueError(
            f"Requested motor IDs {requested_ids} are not all in "
            f"{tuple(motor_ids)}"
        ) from exc


def _build_contact_detectors(
    settings: ContactDetectorSettings,
) -> tuple[ContactStateMachine, ...]:
    """Validate tuning and create one state machine for each hand joint."""

    parameters = {
        "Motion speed threshold": settings.max_velocity_rad_s,
        "Motion current threshold": settings.motion_current_threshold_ma,
        "Settle time": settings.settle_time_s,
        "Entry persistence": settings.persistence_time_s,
        "Entry feedback threshold": settings.entry_feedback_threshold,
        "Minimum contact time": settings.min_engaged_time_s,
        "Exit persistence": settings.exit_persistence_time_s,
    }
    for name, value in parameters.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite, non-negative number")
    kwargs = {
        "max_velocity_rad_s": settings.max_velocity_rad_s,
        "motion_current_threshold_ma": settings.motion_current_threshold_ma,
        "settle_time_s": settings.settle_time_s,
        "persistence_time_s": settings.persistence_time_s,
        "entry_feedback_threshold": settings.entry_feedback_threshold,
        "min_engaged_time_s": settings.min_engaged_time_s,
        "exit_persistence_time_s": settings.exit_persistence_time_s,
    }
    return tuple(ContactStateMachine(**kwargs) for _ in range(JOINT_COUNT))


def _contact_circle_radius(contact_feedback_value: float) -> float:
    """Map absolute contact feedback to a bounded on-canvas radius."""

    return min(120.0, 22.0 + 420.0 * abs(float(contact_feedback_value)))


def _contact_palette(contact_feedback_value: float) -> tuple[str, str, str, str]:
    """Return fill, outline, ring, and accent colors for feedback direction."""

    if contact_feedback_value > 0:
        return "#74c0fc", "#1864ab", "#1971c2", "#a5d8ff"
    if contact_feedback_value < 0:
        return "#ff8a65", "#c92a2a", "#e03131", "#ffc9c9"
    return "#b197fc", "#6741d9", "#7048e8", "#d0bfff"


def _contact_direction_label(contact_feedback_value: float) -> str:
    if contact_feedback_value > 0:
        return "CONTACT  +"
    if contact_feedback_value < 0:
        return "CONTACT  −"
    return "CONTACT"


def _threshold_normalized_joint_difference(
    position_errors: np.ndarray,
    settings: ContactDetectorSettings,
) -> np.ndarray:
    """Return |target - measured| divided by the effective entry threshold."""

    entry_error_threshold = (
        CONTACT_EXIT_THRESHOLD
        + settings.entry_feedback_threshold / CONTACT_VIRTUAL_SPRING_STIFFNESS
    )
    return np.abs(np.asarray(position_errors, dtype=float)) / entry_error_threshold


def _stream_sample(
    elapsed_s: float,
    duration_s: float,
    sequence_kind: Literal["fingers", "thumb"] = "fingers",
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return collision-safe finger or thumb open-close-open targets."""

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    open_targets = np.zeros(JOINT_COUNT, dtype=float)
    open_targets[list(THUMB_INDICES)] = THUMB_OPEN_ANGLES_RAD
    close_targets = open_targets.copy()
    if sequence_kind == "fingers":
        close_targets[list(FLEXION_INDICES)] = STREAM_CLOSE_ANGLE_RAD
    elif sequence_kind == "thumb":
        close_targets[list(THUMB_INDICES)] = THUMB_CLOSE_ANGLES_RAD
    else:
        raise ValueError(f"Unknown sequence kind: {sequence_kind}")
    commanded_speeds = np.abs(close_targets - open_targets) / duration_s
    if elapsed_s <= duration_s:
        progress = max(0.0, elapsed_s / duration_s)
        targets = open_targets + progress * (close_targets - open_targets)
        return targets, commanded_speeds, False
    if elapsed_s <= 2.0 * duration_s:
        progress = (elapsed_s - duration_s) / duration_s
        targets = close_targets + progress * (open_targets - close_targets)
        return targets, commanded_speeds, False
    return open_targets, np.zeros(JOINT_COUNT, dtype=float), True


class HandWorker(threading.Thread):
    """Own the hand and execute every serial-bus operation."""

    def __init__(
        self,
        commands: queue.Queue[tuple[str, object]],
        events: queue.Queue[WorkerEvent],
        *,
        add_tactile: bool = False,
        paxini_port: str | None = None,
    ) -> None:
        # Non-daemon is deliberate: Python must not exit before torque disable.
        super().__init__(name="finger-contact-hand-worker", daemon=False)
        self.commands = commands
        self.events = events
        self.add_tactile = add_tactile
        self.paxini_port = paxini_port
        self.hand: MidasHand | None = None
        self.ui_to_hand_indices = tuple(range(JOINT_COUNT))
        self.targets = np.zeros(JOINT_COUNT, dtype=float)
        self.contact_settings = ContactDetectorSettings()
        self.contact_detectors = _build_contact_detectors(self.contact_settings)
        self.tactile_contact_detectors = {
            name: TactileContactDetector(name=name) for name in TACTILE_SENSOR_NAMES
        }
        self.sequence_active = False
        self.sequence_kind: Literal["fingers", "thumb"] | None = None
        self.sequence_phase: Literal["idle", "preposition", "stream"] = "idle"
        self.sequence_duration_s = 2.0
        self.sequence_started_at = 0.0
        self.preposition_started_at = 0.0
        self.commanded_velocity = np.zeros(JOINT_COUNT, dtype=float)
        self.profile_motion_until = np.zeros(JOINT_COUNT, dtype=float)
        self.running = True

    def run(self) -> None:
        next_tick = time.monotonic()
        while self.running:
            timeout = max(0.0, next_tick - time.monotonic()) if self.hand else 0.1
            try:
                name, payload = self.commands.get(timeout=timeout)
                self._handle_command(name, payload)
                while True:
                    name, payload = self.commands.get_nowait()
                    self._handle_command(name, payload)
            except queue.Empty:
                pass
            except Exception as exc:  # keep the UI alive and make hardware safe
                self.events.put(WorkerEvent("error", str(exc)))
                self._disconnect(emit_event=False)

            if self.hand is not None and time.monotonic() >= next_tick:
                try:
                    self._control_tick()
                except Exception as exc:
                    self.events.put(WorkerEvent("error", str(exc)))
                    self._disconnect(emit_event=False)
                next_tick = time.monotonic() + 1.0 / CONTROL_RATE_HZ

        self._disconnect()

    def _handle_command(self, name: str, payload: object) -> None:
        if name == "connect":
            self._connect(payload)  # type: ignore[arg-type]
        elif name == "disconnect":
            self._disconnect()
        elif name == "targets":
            new_targets = np.asarray(payload, dtype=float)
            if new_targets.shape != (JOINT_COUNT,):
                raise ValueError(f"Expected {JOINT_COUNT} joint targets")
            if not self.sequence_active:
                travel_times = np.abs(new_targets - self.targets) / PROFILE_VELOCITY_RAD_S
                self.profile_motion_until = time.monotonic() + travel_times + 0.1
            self.targets = new_targets
        elif name == "contact_settings":
            settings = payload
            if not isinstance(settings, ContactDetectorSettings):
                raise TypeError("Invalid contact detector settings")
            self.contact_detectors = _build_contact_detectors(settings)
            self.contact_settings = settings
            self.events.put(WorkerEvent("status", "Contact detector settings applied"))
        elif name == "start_sequence":
            sequence_kind, duration_value = payload  # type: ignore[misc]
            if sequence_kind not in ("fingers", "thumb"):
                raise ValueError(f"Unknown sequence kind: {sequence_kind}")
            if self.sequence_active:
                raise RuntimeError("Stop the active sequence before starting another")
            duration_s = float(duration_value)
            if not math.isfinite(duration_s) or duration_s < MIN_STREAM_DURATION_S:
                raise ValueError(
                    f"Time per leg must be at least {MIN_STREAM_DURATION_S:g} seconds"
                )
            if self.hand is None:
                raise RuntimeError("Connect the hand before starting the sequence")
            open_targets, _, _ = _stream_sample(0.0, duration_s, sequence_kind)
            close_targets, _, _ = _stream_sample(
                duration_s, duration_s, sequence_kind
            )
            if not np.array_equal(self._clip_ui_targets(open_targets), open_targets):
                raise ValueError("The open target is outside the configured joint limits")
            if not np.array_equal(self._clip_ui_targets(close_targets), close_targets):
                raise ValueError("The close target is outside the configured joint limits")
            self.sequence_duration_s = duration_s
            self.sequence_kind = sequence_kind
            self.sequence_phase = "preposition"
            self.preposition_started_at = time.monotonic()
            self.sequence_active = True
            self._reset_contact_detectors()
            self.events.put(
                WorkerEvent(
                    "status",
                    f"Opening the hand before the {sequence_kind} sequence…",
                )
            )
        elif name == "stop_sequence":
            self.sequence_active = False
            self.sequence_kind = None
            self.sequence_phase = "idle"
            self.commanded_velocity.fill(0.0)
            self._reset_contact_detectors()
            self.events.put(WorkerEvent("status", "Open-close-open sequence stopped"))
        elif name == "stop":
            self.running = False

    def _connect(self, settings: ConnectionSettings) -> None:
        self._disconnect()
        hardware = "motors and tactile sensor" if self.add_tactile else "motors"
        self.events.put(
            WorkerEvent("status", f"Connecting and verifying {hardware}…")
        )
        with settings.config_path.open(encoding="utf-8") as stream:
            raw_config = yaml.safe_load(stream) or {}
        if raw_config.get("placeholders_unreviewed", True):
            raise RuntimeError(
                "Motion refused: review the calibration and joint limits, then set "
                "placeholders_unreviewed: false in the YAML configuration"
            )
        full_config = HandConfig.load(settings.config_path)
        config = full_config.subset(UI_MOTOR_IDS)
        ui_to_hand_indices = _ui_order_indices(list(config.motor_ids))
        lower_limits = tuple(
            config.joint_lower_limits[index] for index in ui_to_hand_indices
        )
        upper_limits = tuple(
            config.joint_upper_limits[index] for index in ui_to_hand_indices
        )
        default_targets = _validated_default_targets(
            lower_limits, upper_limits
        )
        default_targets_hand_order = np.empty(JOINT_COUNT, dtype=float)
        default_targets_hand_order[list(ui_to_hand_indices)] = default_targets
        tactile_sensor = (
            PaxiniHandSensor(PaxiniConfig(port=self.paxini_port))
            if self.add_tactile
            else None
        )
        hand = MidasHand(
            config, autoconnect=False, tactile_sensor=tactile_sensor
        )
        try:
            hand.connect()
            models = hand.ping()
            missing = sorted(set(hand.motor_ids) - set(models))
            if missing:
                raise OSError(f"Configured motors did not respond: {missing}")
            wrong_models = {
                motor_id: model
                for motor_id, model in models.items()
                if not is_compatible_dynamixel_model(
                    model, config.expected_model_number
                )
            }
            if wrong_models:
                raise OSError(f"Unexpected Dynamixel model numbers: {wrong_models}")

            hand.configure(enable_torque=False)
            positions = hand.read_pos()
            hand.set_positions(positions, clip=False)
            hand.set_motion_profile(
                profile_velocity_rad_s=PROFILE_VELOCITY_RAD_S,
                profile_acceleration_rad_s2=PROFILE_ACCELERATION_RAD_S2,
            )
            hand.enable_torque()
            hand.set_positions(default_targets_hand_order, clip=False)
        except Exception as exc:
            tactile_connection_failed = (
                tactile_sensor is not None
                and hand.is_connected
                and not tactile_sensor.is_connected
            )
            if hand.is_connected:
                hand.shutdown()
            if tactile_connection_failed:
                raise RuntimeError(
                    "Tactile sensor requested, but the Paxini "
                    f"sensor could not be found or connected: {exc}"
                ) from exc
            raise

        self.hand = hand
        self.ui_to_hand_indices = ui_to_hand_indices
        self.targets = default_targets.copy()
        self.sequence_active = False
        self.sequence_kind = None
        self.sequence_phase = "idle"
        self.commanded_velocity.fill(0.0)
        positions_ui_order = np.asarray(positions)[list(ui_to_hand_indices)]
        initial_travel_s = (
            np.abs(default_targets - positions_ui_order) / PROFILE_VELOCITY_RAD_S
        )
        self.profile_motion_until = time.monotonic() + initial_travel_s + 0.1
        self._reset_contact_detectors()
        limits = (
            tuple(float(value) for value in lower_limits),
            tuple(float(value) for value in upper_limits),
            DEFAULT_JOINT_TARGETS_RAD,
            hand.port,
        )
        self.events.put(WorkerEvent("connected", limits))
        tactile_status = (
            f"; tactile on {tactile_sensor.port}" if tactile_sensor is not None else ""
        )
        self.events.put(
            WorkerEvent(
                "status",
                f"Connected on {hand.port}{tactile_status}; torque enabled",
            )
        )

    def _disconnect(self, *, emit_event: bool = True) -> None:
        hand, self.hand = self.hand, None
        self.sequence_active = False
        self.sequence_kind = None
        self.sequence_phase = "idle"
        self.commanded_velocity.fill(0.0)
        self.profile_motion_until.fill(0.0)
        self._reset_contact_detectors()
        self._reset_tactile_contact_detectors()
        if hand is not None and hand.is_connected:
            try:
                hand.shutdown()
            except Exception as exc:
                self.events.put(WorkerEvent("error", f"Shutdown error: {exc}"))
        if hand is not None and emit_event:
            self.events.put(WorkerEvent("disconnected"))

    def _to_hand_order(self, values: np.ndarray) -> np.ndarray:
        ordered = np.empty(JOINT_COUNT, dtype=float)
        ordered[list(self.ui_to_hand_indices)] = values
        return ordered

    def _to_ui_order(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)[list(self.ui_to_hand_indices)]

    def _clip_ui_targets(self, targets: np.ndarray) -> np.ndarray:
        assert self.hand is not None
        clipped = self.hand.clip_positions(self._to_hand_order(targets))
        return self._to_ui_order(clipped)

    def _reset_contact_detectors(self) -> None:
        for detector in self.contact_detectors:
            detector.reset()

    def _reset_tactile_contact_detectors(self) -> None:
        for detector in self.tactile_contact_detectors.values():
            detector.reset()

    def _control_tick(self) -> None:
        assert self.hand is not None
        targets = self.targets.copy()
        if self.sequence_phase == "preposition":
            assert self.sequence_kind is not None
            targets, _, _ = _stream_sample(
                0.0, self.sequence_duration_s, self.sequence_kind
            )
            self.commanded_velocity.fill(PROFILE_VELOCITY_RAD_S)
        elif self.sequence_phase == "stream":
            assert self.sequence_kind is not None
            elapsed = time.monotonic() - self.sequence_started_at
            targets, speeds, complete = _stream_sample(
                elapsed, self.sequence_duration_s, self.sequence_kind
            )
            self.commanded_velocity = speeds
            if complete:
                completed_kind = self.sequence_kind
                self.targets = targets.copy()
                self.sequence_active = False
                self.sequence_kind = None
                self.sequence_phase = "idle"
                self._reset_contact_detectors()
                self.events.put(
                    WorkerEvent(
                        "status", f"{completed_kind.title()} sequence complete"
                    )
                )
        else:
            self.commanded_velocity = np.where(
                time.monotonic() < self.profile_motion_until,
                PROFILE_VELOCITY_RAD_S,
                0.0,
            )

        if self.sequence_active:
            # A user-requested stop will hold the last streamed target rather
            # than jumping back to the pre-sequence slider position.
            self.targets = targets.copy()
        self.hand.set_positions(self._to_hand_order(targets), clip=True)
        positions, velocities, currents = self.hand.read_pos_vel_cur()
        positions = self._to_ui_order(positions)
        velocities = self._to_ui_order(velocities)
        currents = self._to_ui_order(currents)
        if self.sequence_phase == "preposition":
            assert self.sequence_kind is not None
            open_targets, _, _ = _stream_sample(
                0.0, self.sequence_duration_s, self.sequence_kind
            )
            if np.all(
                np.abs(positions - open_targets)
                <= STREAM_PREPOSITION_TOLERANCE_RAD
            ):
                self.sequence_phase = "stream"
                self.sequence_started_at = time.monotonic()
                self._reset_contact_detectors()
                self.events.put(
                    WorkerEvent("status", "Streaming open → close → open")
                )
            elif time.monotonic() - self.preposition_started_at > STREAM_PREPOSITION_TIMEOUT_S:
                raise TimeoutError(
                    "Motors did not reach the open start position within "
                    f"{STREAM_PREPOSITION_TIMEOUT_S:g} seconds"
                )
        errors = targets - positions
        candidate_feedback = contact_feedback(
            errors, thresholds=CONTACT_EXIT_THRESHOLD
        )
        normalized_difference = _threshold_normalized_joint_difference(
            errors, self.contact_settings
        )
        now_s = time.monotonic()
        detected_feedback = np.asarray(
            [
                detector.update(
                    commanded_velocity_rad_s=float(self.commanded_velocity[index]),
                    measured_velocity_rad_s=float(velocities[index]),
                    measured_current_ma=float(currents[index]),
                    candidate_feedback=float(candidate_feedback[index]),
                    now_s=now_s,
                )
                for index, detector in enumerate(self.contact_detectors)
            ]
        )
        contacts = np.asarray(
            [detector.state is ContactState.CONTACT for detector in self.contact_detectors]
        )
        tactile_contacts: tuple[bool, ...] | None = None
        if self.add_tactile:
            try:
                tactile_frame = self.hand.read_tactile_fz()
                missing = [
                    name for name in TACTILE_SENSOR_NAMES if name not in tactile_frame
                ]
                if missing:
                    raise RuntimeError(f"Tactile frame is missing sensors: {missing}")
                tactile_contacts = tuple(
                    self.tactile_contact_detectors[name].update(
                        tactile_frame[name], now_s=now_s
                    )
                    for name in TACTILE_SENSOR_NAMES
                )
            except RuntimeError as exc:
                if "No tactile data available yet" not in str(exc):
                    raise
                tactile_contacts = (False,) * len(TACTILE_SENSOR_NAMES)
        telemetry = Telemetry(
            positions=tuple(float(value) for value in positions),
            velocities=tuple(float(value) for value in velocities),
            currents_ma=tuple(float(value) for value in currents),
            targets=tuple(float(value) for value in targets),
            errors=tuple(float(value) for value in errors),
            candidate_feedback=tuple(float(value) for value in candidate_feedback),
            contact_feedback=tuple(float(value) for value in detected_feedback),
            normalized_joint_difference=tuple(
                float(value) for value in normalized_difference
            ),
            contacts=tuple(bool(value) for value in contacts),
            tactile_contacts=tactile_contacts,
            sequence_active=self.sequence_active,
            sequence_kind=self.sequence_kind,
        )
        self.events.put(WorkerEvent("telemetry", telemetry))


class ContactVisualizerUI:
    """Tk interface for direct position control and contact visualization."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        add_tactile: bool = False,
        paxini_port: str | None = None,
    ) -> None:
        self.root = root
        self.root.title("Hand contact visualizer")
        self.root.minsize(1400, 920)
        self.commands: queue.Queue[tuple[str, object]] = queue.Queue()
        self.events: queue.Queue[WorkerEvent] = queue.Queue()
        self.add_tactile = add_tactile
        self.worker = HandWorker(
            self.commands,
            self.events,
            add_tactile=add_tactile,
            paxini_port=paxini_port,
        )
        self.worker.start()
        self.connected = False
        self.sequence_active = False
        self.sequence_kind: Literal["fingers", "thumb"] | None = None
        self.latest_telemetry: Telemetry | None = None
        self._slider_job: str | None = None
        self._closing = False

        self._build_window()
        self._load_limits_from_config(show_error=False)
        self._draw_hand()
        self.root.after(30, self._drain_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_window(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        for column in range(4):
            outer.columnconfigure(column, weight=1)
        outer.rowconfigure(1, weight=1)

        connection = ttk.Frame(outer)
        connection.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        connection.columnconfigure(1, weight=1)
        ttk.Label(connection, text="Config").grid(row=0, column=0, sticky="w")
        self.config_var = tk.StringVar(value=str(DEFAULT_CONFIG_PATH))
        ttk.Entry(connection, textvariable=self.config_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(connection, text="Browse…", command=self._browse_config).grid(
            row=0, column=2
        )
        ttk.Label(
            connection,
            text="Thumb: 0–3  ·  Fingers: 4–6  ·  7–9  ·  10–12",
        ).grid(row=0, column=3, padx=(16, 0))
        self.connect_button = ttk.Button(
            connection, text="Connect + enable torque", command=self._toggle_connection
        )
        self.connect_button.grid(row=0, column=4, padx=(16, 0))

        visual = ttk.LabelFrame(outer, text="Live contact", padding=8)
        visual.grid(row=1, column=0, columnspan=4, sticky="nsew")
        visual.rowconfigure(0, weight=1)
        visual.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            visual, background="#f7f7f5", highlightthickness=0, height=420
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._redraw_latest())

        self.target_vars = [tk.DoubleVar(value=0.0) for _ in range(JOINT_COUNT)]
        self.value_vars = [tk.StringVar(value="0.000 rad") for _ in range(JOINT_COUNT)]
        self.scales: list[tk.Scale] = []
        for finger_index, (pip_id, mcp_id, ab_ad_id) in enumerate(FINGER_MOTOR_IDS):
            module = ttk.LabelFrame(
                outer, text=f"Finger {finger_index + 1} joint targets", padding=10
            )
            module.grid(
                row=2,
                column=finger_index,
                sticky="nsew",
                padx=(0 if finger_index == 0 else 6, 0 if finger_index == 2 else 6),
                pady=(10, 0),
            )
            module.columnconfigure(0, weight=1)
            offset = finger_index * 3
            for joint_offset, (joint_name, motor_id) in enumerate(
                zip(("MCP", "PIP", "AB/AD"), (mcp_id, pip_id, ab_ad_id))
            ):
                self.scales.append(
                    self._add_slider(
                        module,
                        joint_offset * 2,
                        f"{joint_name}  ·  ID {motor_id}",
                        self.target_vars[offset + joint_offset],
                        self.value_vars[offset + joint_offset],
                    )
                )

        thumb_module = ttk.LabelFrame(outer, text="Thumb joint targets", padding=10)
        thumb_module.grid(
            row=2, column=3, sticky="nsew", padx=(6, 0), pady=(10, 0)
        )
        thumb_module.columnconfigure(0, weight=1)
        for joint_offset, (joint_name, motor_id) in enumerate(
            zip(("IP", "MCP", "CMC side", "CMC roll"), THUMB_MOTOR_IDS)
        ):
            target_index = THUMB_INDICES[joint_offset]
            self.scales.append(
                self._add_slider(
                    thumb_module,
                    joint_offset * 2,
                    f"{joint_name}  ·  ID {motor_id}",
                    self.target_vars[target_index],
                    self.value_vars[target_index],
                )
            )

        detector = ttk.LabelFrame(outer, text="Contact detector settings", padding=10)
        detector.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10, 0), padx=(0, 6))
        detector.columnconfigure(1, weight=1)
        self.contact_max_velocity_var = tk.StringVar(
            value=str(CONTACT_MAX_VELOCITY_RAD_S)
        )
        self.contact_settle_time_var = tk.StringVar(value=str(CONTACT_SETTLE_TIME_S))
        self.contact_motion_current_var = tk.StringVar(
            value=str(CONTACT_MOTION_CURRENT_THRESHOLD_MA)
        )
        self.contact_persistence_var = tk.StringVar(
            value=str(CONTACT_PERSISTENCE_TIME_S)
        )
        self.contact_entry_feedback_var = tk.StringVar(
            value=str(CONTACT_ENTRY_FEEDBACK_THRESHOLD)
        )
        self.contact_min_engaged_var = tk.StringVar(
            value=str(CONTACT_MIN_ENGAGED_TIME_S)
        )
        self.contact_exit_persistence_var = tk.StringVar(
            value=str(CONTACT_EXIT_PERSISTENCE_TIME_S)
        )
        self._labeled_entry(
            detector,
            0,
            "Motion speed threshold",
            self.contact_max_velocity_var,
            "rad/s",
        )
        self._labeled_entry(
            detector,
            1,
            "Motion current threshold",
            self.contact_motion_current_var,
            "mA",
        )
        self._labeled_entry(
            detector, 2, "Settle time", self.contact_settle_time_var, "seconds"
        )
        self._labeled_entry(
            detector, 3, "Entry persistence", self.contact_persistence_var, "seconds"
        )
        self._labeled_entry(
            detector,
            4,
            "Entry feedback threshold",
            self.contact_entry_feedback_var,
            "feedback",
        )
        self._labeled_entry(
            detector,
            5,
            "Minimum contact time",
            self.contact_min_engaged_var,
            "seconds",
        )
        self._labeled_entry(
            detector,
            6,
            "Exit persistence",
            self.contact_exit_persistence_var,
            "seconds",
        )
        ttk.Label(
            detector,
            text=(
                "Above the motion speed threshold, current enters contact; "
                f"feedback still uses the {CONTACT_EXIT_THRESHOLD:g} rad "
                "position deadband."
            ),
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(5, 0))
        ttk.Button(
            detector,
            text="Apply contact settings",
            command=self._apply_contact_settings,
        ).grid(row=8, column=0, columnspan=3, sticky="ew", pady=(7, 0))

        self.sequence_duration_vars: dict[str, tk.StringVar] = {}
        self.sequence_buttons: dict[str, ttk.Button] = {}
        for column, kind, title in (
            (2, "fingers", "Finger open-close-open sequence"),
            (3, "thumb", "Thumb open-close-open sequence"),
        ):
            sequence = ttk.LabelFrame(outer, text=title, padding=10)
            sequence.grid(
                row=3, column=column, sticky="nsew", pady=(10, 0), padx=(6, 0)
            )
            sequence.columnconfigure(1, weight=1)
            duration_var = tk.StringVar(value="2.0")
            self.sequence_duration_vars[kind] = duration_var
            self._labeled_entry(
                sequence, 0, "Time per leg", duration_var, "seconds"
            )
            button = ttk.Button(
                sequence,
                text="Run open → close → open",
                command=lambda selected=kind: self._toggle_sequence(selected),
                state="disabled",
            )
            button.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(7, 0))
            self.sequence_buttons[kind] = button

        self.status_var = tk.StringVar(value="Disconnected — torque disabled")
        ttk.Label(outer, textvariable=self.status_var, anchor="w").grid(
            row=4, column=0, columnspan=4, sticky="ew", pady=(10, 0)
        )

    def _add_slider(
        self,
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.DoubleVar,
        display: tk.StringVar,
    ) -> tk.Scale:
        header = ttk.Frame(parent)
        header.grid(row=row, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=label).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=display).grid(row=0, column=1, sticky="e")
        scale = tk.Scale(
            parent,
            from_=-1.5,
            to=0.05,
            orient="horizontal",
            resolution=0.001,
            showvalue=False,
            variable=variable,
            command=lambda _value: self._slider_changed(),
            sliderlength=24,
            highlightthickness=0,
        )
        scale.grid(row=row + 1, column=0, sticky="ew", pady=(2, 8))
        return scale

    @staticmethod
    def _labeled_entry(
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
        units: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable, width=9).grid(
            row=row, column=1, sticky="ew", padx=8, pady=2
        )
        ttk.Label(parent, text=units).grid(row=row, column=2, sticky="w")

    def _browse_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select hand configuration",
            initialfile=Path(self.config_var.get()).name,
            filetypes=(("YAML files", "*.yaml *.yml"), ("All files", "*")),
        )
        if selected:
            self.config_var.set(selected)
            self._load_limits_from_config(show_error=True)

    def _load_limits_from_config(self, *, show_error: bool) -> bool:
        try:
            config = HandConfig.load(Path(self.config_var.get()).expanduser())
            subset = config.subset(UI_MOTOR_IDS)
            indices = _ui_order_indices(list(subset.motor_ids))
            lower = tuple(subset.joint_lower_limits[index] for index in indices)
            upper = tuple(subset.joint_upper_limits[index] for index in indices)
            _validated_default_targets(lower, upper)
            for scale, minimum, maximum in zip(self.scales, lower, upper):
                scale.configure(from_=minimum, to=maximum)
            if not self.connected:
                for target_var, value_var in zip(self.target_vars, self.value_vars):
                    target_var.set(0.0)
                    value_var.set("0.000 rad")
            return True
        except Exception as exc:
            if show_error:
                messagebox.showerror("Invalid configuration", str(exc))
            return False

    def _toggle_connection(self) -> None:
        if self.connected:
            self.commands.put(("disconnect", None))
            self.connect_button.configure(state="disabled")
            self.status_var.set("Disconnecting and disabling torque…")
            return
        try:
            settings = ConnectionSettings(Path(self.config_var.get()).expanduser())
            if not settings.config_path.is_file():
                raise ValueError(f"Configuration does not exist: {settings.config_path}")
        except ValueError as exc:
            messagebox.showerror("Invalid connection settings", str(exc))
            return
        if not self._load_limits_from_config(show_error=True):
            return
        if not messagebox.askokcancel(
            "Enable physical motion",
            "Connecting will configure motors 0–12, enable torque, and move all "
            "hand joints to 0 rad. Keep the hand clear, verify the motor IDs, "
            "and have an emergency stop ready.",
            icon="warning",
        ):
            return
        self.connect_button.configure(state="disabled")
        self.status_var.set("Connecting…")
        self.commands.put(("connect", settings))

    def _slider_changed(self) -> None:
        for target_var, value_var in zip(self.target_vars, self.value_vars):
            value_var.set(f"{target_var.get():.3f} rad")
        # Throttle to the hardware control rate instead of debouncing. Cancelling
        # and restarting this callback on every drag event delays the command
        # until the user releases the slider, which makes control feel laggy.
        if self._slider_job is None:
            interval_ms = max(1, round(1000.0 / CONTROL_RATE_HZ))
            self._slider_job = self.root.after(interval_ms, self._send_targets)

    def _send_targets(self) -> None:
        self._slider_job = None
        if self.connected and not self.sequence_active:
            self.commands.put(("targets", tuple(var.get() for var in self.target_vars)))

    def _apply_contact_settings(self) -> None:
        try:
            settings = ContactDetectorSettings(
                max_velocity_rad_s=float(self.contact_max_velocity_var.get()),
                motion_current_threshold_ma=float(
                    self.contact_motion_current_var.get()
                ),
                settle_time_s=float(self.contact_settle_time_var.get()),
                persistence_time_s=float(self.contact_persistence_var.get()),
                entry_feedback_threshold=float(
                    self.contact_entry_feedback_var.get()
                ),
                min_engaged_time_s=float(self.contact_min_engaged_var.get()),
                exit_persistence_time_s=float(
                    self.contact_exit_persistence_var.get()
                ),
            )
            _build_contact_detectors(settings)
        except ValueError as exc:
            messagebox.showerror("Invalid contact setting", str(exc))
            return
        self.commands.put(("contact_settings", settings))
        if not self.connected:
            self.status_var.set(
                "Contact detector settings saved; they will apply on connection"
            )

    def _toggle_sequence(self, sequence_kind: Literal["fingers", "thumb"]) -> None:
        if self.sequence_active:
            if self.sequence_kind == sequence_kind:
                self.commands.put(("stop_sequence", None))
            return
        try:
            duration_s = float(self.sequence_duration_vars[sequence_kind].get())
            if not math.isfinite(duration_s) or duration_s <= 0:
                raise ValueError("Time per leg must be a positive number of seconds")
            if duration_s < MIN_STREAM_DURATION_S:
                raise ValueError(
                    f"Time per leg must be at least {MIN_STREAM_DURATION_S:g} "
                    f"seconds at the {CONTROL_RATE_HZ:g} Hz control rate"
                )
        except ValueError as exc:
            messagebox.showerror("Invalid sequence time", str(exc))
            return
        if not messagebox.askokcancel(
            f"Run {sequence_kind} open-close-open sequence",
            (
                "All finger MCP and PIP joints will move 0 → -1.35 → 0 rad; "
                "finger AB/AD joints stay at 0 rad and the thumb stays open. "
                if sequence_kind == "fingers"
                else "The thumb will use its configured four-joint open and close "
                "targets while every finger remains open at 0 rad. "
            )
            + f"Each leg takes {duration_s:g} seconds. Make sure the full hand "
            "workspace is clear.",
            icon="warning",
        ):
            return
        self.commands.put(("start_sequence", (sequence_kind, duration_s)))

    def _drain_events(self) -> None:
        latest_telemetry: Telemetry | None = None
        try:
            while True:
                event = self.events.get_nowait()
                if event.kind == "telemetry":
                    latest_telemetry = event.payload  # type: ignore[assignment]
                else:
                    self._handle_event(event)
        except queue.Empty:
            pass
        if latest_telemetry is not None:
            self._show_telemetry(latest_telemetry)
        self.root.after(30, self._drain_events)

    def _handle_event(self, event: WorkerEvent) -> None:
        if event.kind == "status":
            self.status_var.set(str(event.payload))
        elif event.kind == "connected":
            lower, upper, targets, _port = event.payload  # type: ignore[misc]
            self.connected = True
            self.connect_button.configure(text="Disconnect + disable torque", state="normal")
            for button in self.sequence_buttons.values():
                button.configure(state="normal")
            for index, scale in enumerate(self.scales):
                scale.configure(state="normal", from_=lower[index], to=upper[index])
                self.target_vars[index].set(targets[index])
            self._slider_changed()
        elif event.kind == "disconnected":
            self.connected = False
            self.sequence_active = False
            self.sequence_kind = None
            self.connect_button.configure(text="Connect + enable torque", state="normal")
            for button in self.sequence_buttons.values():
                button.configure(text="Run open → close → open", state="disabled")
            for scale in self.scales:
                scale.configure(state="normal")
            self.status_var.set("Disconnected — torque disabled")
            self.latest_telemetry = None
            self._draw_hand()
        elif event.kind == "error":
            self.connected = False
            self.sequence_active = False
            self.sequence_kind = None
            self.connect_button.configure(text="Connect + enable torque", state="normal")
            for button in self.sequence_buttons.values():
                button.configure(text="Run open → close → open", state="disabled")
            for scale in self.scales:
                scale.configure(state="normal")
            self.status_var.set(f"Error: {event.payload}")
            if not self._closing:
                messagebox.showerror("Hand controller error", str(event.payload))

    def _show_telemetry(self, telemetry: Telemetry) -> None:
        self.latest_telemetry = telemetry
        sequence_changed = self.sequence_active != telemetry.sequence_active
        self.sequence_active = telemetry.sequence_active
        self.sequence_kind = telemetry.sequence_kind
        for kind, button in self.sequence_buttons.items():
            is_active_button = telemetry.sequence_active and kind == telemetry.sequence_kind
            button.configure(
                text="Stop sequence" if is_active_button else "Run open → close → open",
                state=(
                    "normal"
                    if not telemetry.sequence_active or is_active_button
                    else "disabled"
                ),
            )
        slider_state = "disabled" if telemetry.sequence_active else "normal"
        for scale in self.scales:
            scale.configure(state=slider_state)
        if sequence_changed and not telemetry.sequence_active:
            for target_var, value_var, target in zip(
                self.target_vars, self.value_vars, telemetry.targets
            ):
                target_var.set(target)
                value_var.set(f"{target:.3f} rad")
        self._draw_hand(
            telemetry.contacts,
            telemetry.contact_feedback,
            telemetry.normalized_joint_difference,
            telemetry.tactile_contacts,
        )

    def _redraw_latest(self) -> None:
        telemetry = self.latest_telemetry
        if telemetry is None:
            self._draw_hand()
        else:
            self._draw_hand(
                telemetry.contacts,
                telemetry.contact_feedback,
                telemetry.normalized_joint_difference,
                telemetry.tactile_contacts,
            )

    def _draw_hand(
        self,
        contacts: tuple[bool, ...] = (False,) * JOINT_COUNT,
        contact_strengths: tuple[float, ...] = (0.0,) * JOINT_COUNT,
        normalized_differences: tuple[float, ...] = (0.0,) * JOINT_COUNT,
        tactile_contacts: tuple[bool, ...] | None = None,
    ) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 600)
        height = max(self.canvas.winfo_height(), 400)
        show_tactile = tactile_contacts is not None
        x_positions = (width * 0.15, width * 0.32, width * 0.49)
        block_width = min(120.0, width * 0.12)
        pip_center_y = height * (0.31 if show_tactile else 0.25)
        mcp_center_y = height * (0.59 if show_tactile else 0.56)
        ab_ad_center_y = height * 0.84
        block_height = min(115.0, height * 0.25)

        self.canvas.create_text(
            22, height / 2, text="MCP", fill="#333333",
            font=("TkDefaultFont", 13, "bold"),
        )
        self.canvas.create_text(
            22, pip_center_y, text="PIP", fill="#333333",
            font=("TkDefaultFont", 13, "bold"),
        )
        self.canvas.create_text(
            28, ab_ad_center_y, text="AB/AD", fill="#333333",
            font=("TkDefaultFont", 13, "bold"),
        )
        self._draw_contact_direction_legend(width)

        # Match the supplied sketch: finger 3 is leftmost and finger 1 rightmost.
        for x, finger_index in zip(x_positions, (2, 1, 0)):
            offset = finger_index * 3
            mcp_index, pip_index, ab_ad_index = offset, offset + 1, offset + 2
            self.canvas.create_text(
                x, 18, text=f"Finger {finger_index + 1}", fill="#333333",
                font=("TkDefaultFont", 13, "bold"),
            )
            if show_tactile:
                assert tactile_contacts is not None
                self._draw_tactile_contact_indicator(
                    np.asarray((x, 55.0)), tactile_contacts[finger_index]
                )
            self._draw_joint_block(
                np.asarray((x, pip_center_y)), block_width, block_height,
                f"PIP  ·  ID {UI_MOTOR_IDS[pip_index]}", contacts[pip_index],
                contact_strengths[pip_index],
            )
            self._draw_joint_block(
                np.asarray((x, mcp_center_y)), block_width, block_height,
                f"MCP  ·  ID {UI_MOTOR_IDS[mcp_index]}", contacts[mcp_index],
                contact_strengths[mcp_index],
            )
            self._draw_round_joint(
                np.asarray((x, ab_ad_center_y)),
                f"AB/AD · ID {UI_MOTOR_IDS[ab_ad_index]}",
                contacts[ab_ad_index],
                contact_strengths[ab_ad_index],
            )
            for index, center in (
                (pip_index, np.asarray((x, pip_center_y))),
                (mcp_index, np.asarray((x, mcp_center_y))),
                (ab_ad_index, np.asarray((x, ab_ad_center_y))),
            ):
                if contacts[index]:
                    self._draw_contact_circle(
                        center, contact_strengths[index], normalized_differences[index]
                    )

        thumb_x = width * 0.76
        thumb_width = min(105.0, width * 0.10)
        thumb_height = min(88.0, height * 0.19)
        thumb_y_fractions = (
            (0.30, 0.50, 0.70, 0.88)
            if show_tactile
            else (0.20, 0.45, 0.68, 0.88)
        )
        thumb_centers = tuple(
            np.asarray((thumb_x + x_offset, height * y_fraction))
            for x_offset, y_fraction in zip(
                (0.0, 0.0, -8.0, -28.0), thumb_y_fractions
            )
        )
        self.canvas.create_text(
            thumb_x, 18, text="Thumb", fill="#333333",
            font=("TkDefaultFont", 13, "bold"),
        )
        if show_tactile:
            assert tactile_contacts is not None
            self._draw_tactile_contact_indicator(
                np.asarray((thumb_x, 55.0)), tactile_contacts[3]
            )
        self._draw_joint_block(
            thumb_centers[0], thumb_width, thumb_height,
            "IP  ·  ID 0", contacts[THUMB_INDICES[0]],
            contact_strengths[THUMB_INDICES[0]],
        )
        self._draw_joint_block(
            thumb_centers[1], thumb_width, thumb_height,
            "MCP  ·  ID 1", contacts[THUMB_INDICES[1]],
            contact_strengths[THUMB_INDICES[1]],
        )
        self._draw_rotated_joint_block(
            thumb_centers[2], thumb_width, thumb_height * 0.78,
            "CMC side  ·  ID 2", contacts[THUMB_INDICES[2]],
            contact_strengths[THUMB_INDICES[2]], -14.0,
        )
        self._draw_round_joint(
            thumb_centers[3], "CMC roll · ID 3", contacts[THUMB_INDICES[3]],
            contact_strengths[THUMB_INDICES[3]],
        )
        for index, center in zip(THUMB_INDICES, thumb_centers):
            if contacts[index]:
                self._draw_contact_circle(
                    center, contact_strengths[index], normalized_differences[index]
                )

    def _draw_joint_block(
        self,
        center: np.ndarray,
        width: float,
        height: float,
        label: str,
        contact: bool,
        contact_feedback_value: float,
    ) -> None:
        x, y = center
        fill, outline, _, _ = _contact_palette(contact_feedback_value)
        self.canvas.create_rectangle(
            x - width / 2,
            y - height / 2,
            x + width / 2,
            y + height / 2,
            fill=fill if contact else "#ffffff",
            outline=outline if contact else "#333333",
            width=4,
        )
        state = _contact_direction_label(contact_feedback_value) if contact else "NO CONTACT"
        self.canvas.create_text(
            x, y, text=f"{label}\n{state}", fill="#1f1f1f",
            justify="center", font=("TkDefaultFont", 11, "bold"),
        )

    def _draw_rotated_joint_block(
        self,
        center: np.ndarray,
        width: float,
        height: float,
        label: str,
        contact: bool,
        contact_feedback_value: float,
        angle_degrees: float,
    ) -> None:
        angle = math.radians(angle_degrees)
        rotation = np.asarray(
            ((math.cos(angle), -math.sin(angle)),
             (math.sin(angle), math.cos(angle)))
        )
        corners = np.asarray(
            (
                (-width / 2, -height / 2),
                (width / 2, -height / 2),
                (width / 2, height / 2),
                (-width / 2, height / 2),
            )
        )
        points = (corners @ rotation.T) + center
        fill, outline, _, _ = _contact_palette(contact_feedback_value)
        self.canvas.create_polygon(
            *points.ravel(),
            fill=fill if contact else "#ffffff",
            outline=outline if contact else "#333333",
            width=4,
        )
        state = _contact_direction_label(contact_feedback_value) if contact else "NO CONTACT"
        self.canvas.create_text(
            center[0], center[1], text=f"{label}\n{state}", fill="#1f1f1f",
            justify="center", angle=angle_degrees,
            font=("TkDefaultFont", 9, "bold"),
        )

    def _draw_round_joint(
        self,
        center: np.ndarray,
        label: str,
        contact: bool,
        contact_feedback_value: float,
    ) -> None:
        x, y = center
        radius = 43.0
        fill, outline, _, _ = _contact_palette(contact_feedback_value)
        self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            fill=fill if contact else "#ffffff",
            outline=outline if contact else "#333333",
            width=4,
        )
        state = _contact_direction_label(contact_feedback_value) if contact else "NO CONTACT"
        self.canvas.create_text(
            x, y, text=f"{label}\n{state}", fill="#1f1f1f",
            justify="center", font=("TkDefaultFont", 10, "bold"),
        )

    def _draw_contact_direction_legend(self, canvas_width: float) -> None:
        x = canvas_width - 185.0
        self.canvas.create_text(
            x, 18, anchor="w", text="Contact direction", fill="#333333",
            font=("TkDefaultFont", 11, "bold"),
        )
        for row, (feedback, label) in enumerate(
            ((1.0, "Positive feedback"), (-1.0, "Negative feedback"))
        ):
            fill, outline, _, _ = _contact_palette(feedback)
            y = 43.0 + row * 27.0
            self.canvas.create_rectangle(
                x, y - 8, x + 20, y + 8, fill=fill, outline=outline, width=2
            )
            self.canvas.create_text(
                x + 29, y, anchor="w", text=label, fill="#333333",
                font=("TkDefaultFont", 10),
            )

    def _draw_contact_circle(
        self, center: np.ndarray, strength: float, normalized_difference: float
    ) -> None:
        """Draw a direction-colored ring whose radius tracks feedback magnitude."""

        radius = _contact_circle_radius(strength)
        _, _, ring, accent = _contact_palette(strength)
        x, y = center
        bounds = (x - radius, y - radius, x + radius, y + radius)
        self.canvas.create_oval(*bounds, outline=ring, width=9)
        self.canvas.create_oval(*bounds, outline=accent, width=3)
        self.canvas.create_text(
            x,
            y + radius + 13,
            text=f"{normalized_difference:.2f}× threshold",
            fill="#9a2f20",
            font=("TkDefaultFont", 11, "bold"),
        )

    def _draw_tactile_contact_indicator(
        self, center: np.ndarray, contact: bool
    ) -> None:
        """Draw one fixed-size binary fingertip contact indicator."""

        x, y = center
        radius = 24.0
        fill = "#ff8a65" if contact else "#ffffff"
        outline = "#c92a2a" if contact else "#666666"
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=fill,
            outline=outline,
            width=3,
        )
        self.canvas.create_text(
            x,
            y,
            text="CONTACT" if contact else "NO\nCONTACT",
            fill="#1f1f1f",
            font=("TkDefaultFont", 8, "bold"),
            justify="center",
        )

    def _on_close(self) -> None:
        if self._closing:
            return
        if self.connected and not messagebox.askyesno(
            "Disable torque and quit?",
            "The hand is connected. Disable torque and close the controller?",
        ):
            return
        self._closing = True
        self.status_var.set("Disabling torque and closing…")
        self.connect_button.configure(state="disabled")
        for button in self.sequence_buttons.values():
            button.configure(state="disabled")
        self.commands.put(("stop", None))
        self._finish_close()

    def _finish_close(self) -> None:
        if self.worker.is_alive():
            self.root.after(50, self._finish_close)
        else:
            self.root.destroy()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--add-tactile",
        action="store_true",
        help=(
            "connect the auto-detected Paxini tactile sensor and show the summed "
            "normal force above each fingertip"
        ),
    )
    parser.add_argument(
        "--paxini-port",
        metavar="PORT",
        help=(
            "Paxini serial port, for example /dev/cu.usbserial-XXXX; providing "
            "this option also enables tactile sensing"
        ),
    )
    parser.add_argument(
        "--tactile-debug-log",
        type=Path,
        metavar="PATH",
        help=(
            "write structured moving-average contact-detector diagnostics to "
            "PATH; providing this option also enables tactile sensing"
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    add_tactile = (
        args.add_tactile
        or args.paxini_port is not None
        or args.tactile_debug_log is not None
    )
    if args.tactile_debug_log is not None:
        _configure_tactile_debug_log(args.tactile_debug_log)
    root = tk.Tk()
    ui = ContactVisualizerUI(
        root,
        add_tactile=add_tactile,
        paxini_port=args.paxini_port,
    )
    try:
        root.mainloop()
    finally:
        if ui.worker.is_alive():
            ui.commands.put(("stop", None))
            ui.worker.join()


if __name__ == "__main__":
    main()
