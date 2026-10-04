"""Interactive two-joint controller and contact visualizer.

Run from any directory with the project environment::

    .venv/bin/python scripts/trimorph_api_test/two_joint_test_ui.py

The hardware is owned by one background thread. Tkinter never performs a bus
operation, so rendering and the Stop controls remain responsive.
"""

from __future__ import annotations

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
    CONTACT_ENTER_THRESHOLD,
    CONTACT_EXIT_PERSISTENCE_TIME_S,
    CONTACT_EXIT_THRESHOLD,
    CONTACT_MAX_VELOCITY_RAD_S,
    CONTACT_MIN_ENGAGED_TIME_S,
    CONTACT_PERSISTENCE_TIME_S,
    CONTACT_SETTLE_TIME_S,
    ContactState,
    ContactStateMachine,
    contact_feedback,
)
from midas_hand_api import HandConfig, MidasHand


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("xl330_test_config.yaml")
DEFAULT_MCP_ID = 5
DEFAULT_PIP_ID = 4
CONTROL_RATE_HZ = 50.0
PROFILE_VELOCITY_RAD_S = 3.0
PROFILE_ACCELERATION_RAD_S2 = 50.0
DEFAULT_JOINT_TARGETS_RAD = (0.0, 0.0)  # order: MCP, PIP
STREAM_OPEN_ANGLE_RAD = 0.0
STREAM_CLOSE_ANGLE_RAD = -1.35
STREAM_PREPOSITION_TOLERANCE_RAD = 0.1
STREAM_PREPOSITION_TIMEOUT_S = 4.0
MIN_STREAM_DURATION_S = 1.0 / CONTROL_RATE_HZ


@dataclass(frozen=True)
class ConnectionSettings:
    config_path: Path
    mcp_id: int
    pip_id: int


@dataclass(frozen=True)
class Telemetry:
    positions: tuple[float, float]
    velocities: tuple[float, float]
    currents_ma: tuple[float, float]
    targets: tuple[float, float]
    errors: tuple[float, float]
    candidate_feedback: tuple[float, float]
    contact_feedback: tuple[float, float]
    contacts: tuple[bool, bool]
    sequence_active: bool


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
        raise ValueError("Expected exactly two configured joint limits for MCP and PIP")
    invalid = [
        name
        for name, target, minimum, maximum in zip(
            ("MCP", "PIP"), targets, lower, upper
        )
        if not minimum <= target <= maximum
    ]
    if invalid:
        raise ValueError(
            f"Default 0 rad target is outside the configured limits for {invalid}"
        )
    return targets


def _ui_order_indices(
    motor_ids: tuple[int, ...] | list[int], mcp_id: int, pip_id: int
) -> tuple[int, int]:
    """Map UI order (MCP, PIP) to an API array's configured motor order."""

    if mcp_id == pip_id:
        raise ValueError("MCP and PIP motor IDs must be different")
    try:
        return motor_ids.index(mcp_id), motor_ids.index(pip_id)
    except ValueError as exc:
        raise ValueError(
            f"MCP/PIP motor IDs {(mcp_id, pip_id)} are not both in {tuple(motor_ids)}"
        ) from exc


def _stream_sample(elapsed_s: float, duration_s: float) -> tuple[np.ndarray, float, bool]:
    """Return the open-close-open target, speed, and completion state."""

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    open_targets = np.full(2, STREAM_OPEN_ANGLE_RAD)
    close_targets = np.full(2, STREAM_CLOSE_ANGLE_RAD)
    commanded_speed = abs(STREAM_CLOSE_ANGLE_RAD - STREAM_OPEN_ANGLE_RAD) / duration_s
    if elapsed_s <= duration_s:
        progress = max(0.0, elapsed_s / duration_s)
        targets = open_targets + progress * (close_targets - open_targets)
        return targets, commanded_speed, False
    if elapsed_s <= 2.0 * duration_s:
        progress = (elapsed_s - duration_s) / duration_s
        targets = close_targets + progress * (open_targets - close_targets)
        return targets, commanded_speed, False
    return open_targets, 0.0, True


class HandWorker(threading.Thread):
    """Own the hand and execute every serial-bus operation."""

    def __init__(
        self,
        commands: queue.Queue[tuple[str, object]],
        events: queue.Queue[WorkerEvent],
    ) -> None:
        # Non-daemon is deliberate: Python must not exit before torque disable.
        super().__init__(name="two-joint-hand-worker", daemon=False)
        self.commands = commands
        self.events = events
        self.hand: MidasHand | None = None
        self.ui_to_hand_indices = (0, 1)
        self.targets = np.zeros(2, dtype=float)  # order: MCP, PIP
        self.contact_detectors = (ContactStateMachine(), ContactStateMachine())
        self.sequence_active = False
        self.sequence_phase: Literal["idle", "preposition", "stream"] = "idle"
        self.sequence_duration_s = 2.0
        self.sequence_started_at = 0.0
        self.preposition_started_at = 0.0
        self.commanded_velocity = np.zeros(2, dtype=float)
        self.profile_motion_until = np.zeros(2, dtype=float)
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
            if not self.sequence_active:
                travel_times = np.abs(new_targets - self.targets) / PROFILE_VELOCITY_RAD_S
                self.profile_motion_until = time.monotonic() + travel_times + 0.1
            self.targets = new_targets
        elif name == "start_sequence":
            duration_s = float(payload)  # type: ignore[arg-type]
            if not math.isfinite(duration_s) or duration_s < MIN_STREAM_DURATION_S:
                raise ValueError(
                    f"Time per leg must be at least {MIN_STREAM_DURATION_S:g} seconds"
                )
            if self.hand is None:
                raise RuntimeError("Connect the hand before starting the sequence")
            open_targets = np.full(2, STREAM_OPEN_ANGLE_RAD)
            close_targets = np.full(2, STREAM_CLOSE_ANGLE_RAD)
            if not np.array_equal(self._clip_ui_targets(open_targets), open_targets):
                raise ValueError("The open target is outside the configured joint limits")
            if not np.array_equal(self._clip_ui_targets(close_targets), close_targets):
                raise ValueError("The close target is outside the configured joint limits")
            self.sequence_duration_s = duration_s
            self.sequence_phase = "preposition"
            self.preposition_started_at = time.monotonic()
            self.sequence_active = True
            self._reset_contact_detectors()
            self.events.put(WorkerEvent("status", "Moving to the open start position…"))
        elif name == "stop_sequence":
            self.sequence_active = False
            self.sequence_phase = "idle"
            self.commanded_velocity.fill(0.0)
            self._reset_contact_detectors()
            self.events.put(WorkerEvent("status", "Open-close-open sequence stopped"))
        elif name == "stop":
            self.running = False

    def _connect(self, settings: ConnectionSettings) -> None:
        self._disconnect()
        self.events.put(WorkerEvent("status", "Connecting and verifying motors…"))
        with settings.config_path.open(encoding="utf-8") as stream:
            raw_config = yaml.safe_load(stream) or {}
        if raw_config.get("placeholders_unreviewed", True):
            raise RuntimeError(
                "Motion refused: review the calibration and joint limits, then set "
                "placeholders_unreviewed: false in the YAML configuration"
            )
        full_config = HandConfig.load(settings.config_path)
        config = full_config.subset((settings.mcp_id, settings.pip_id))
        ui_to_hand_indices = _ui_order_indices(
            list(config.motor_ids), settings.mcp_id, settings.pip_id
        )
        lower_limits = tuple(
            config.joint_lower_limits[index] for index in ui_to_hand_indices
        )
        upper_limits = tuple(
            config.joint_upper_limits[index] for index in ui_to_hand_indices
        )
        default_targets = _validated_default_targets(
            lower_limits, upper_limits
        )
        default_targets_hand_order = np.empty(2, dtype=float)
        default_targets_hand_order[list(ui_to_hand_indices)] = default_targets
        hand = MidasHand(config, autoconnect=False)
        try:
            hand.connect()
            models = hand.ping()
            missing = sorted(set(hand.motor_ids) - set(models))
            if missing:
                raise OSError(f"Configured motors did not respond: {missing}")
            wrong_models = {
                motor_id: model
                for motor_id, model in models.items()
                if model != config.expected_model_number
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
        except Exception:
            if hand.is_connected:
                hand.shutdown()
            raise

        self.hand = hand
        self.ui_to_hand_indices = ui_to_hand_indices
        self.targets = default_targets.copy()
        self.sequence_active = False
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
        self.events.put(WorkerEvent("status", f"Connected on {hand.port}; torque enabled"))

    def _disconnect(self, *, emit_event: bool = True) -> None:
        hand, self.hand = self.hand, None
        self.sequence_active = False
        self.sequence_phase = "idle"
        self.commanded_velocity.fill(0.0)
        self.profile_motion_until.fill(0.0)
        self._reset_contact_detectors()
        if hand is not None and hand.is_connected:
            try:
                hand.shutdown()
            except Exception as exc:
                self.events.put(WorkerEvent("error", f"Shutdown error: {exc}"))
        if hand is not None and emit_event:
            self.events.put(WorkerEvent("disconnected"))

    def _to_hand_order(self, values: np.ndarray) -> np.ndarray:
        ordered = np.empty(2, dtype=float)
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

    def _control_tick(self) -> None:
        assert self.hand is not None
        targets = self.targets.copy()
        if self.sequence_phase == "preposition":
            targets = np.full(2, STREAM_OPEN_ANGLE_RAD)
            self.commanded_velocity.fill(PROFILE_VELOCITY_RAD_S)
        elif self.sequence_phase == "stream":
            elapsed = time.monotonic() - self.sequence_started_at
            targets, speed, complete = _stream_sample(
                elapsed, self.sequence_duration_s
            )
            self.commanded_velocity.fill(speed)
            if complete:
                self.targets = targets.copy()
                self.sequence_active = False
                self.sequence_phase = "idle"
                self._reset_contact_detectors()
                self.events.put(
                    WorkerEvent("status", "Open-close-open sequence complete")
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
            if np.all(
                np.abs(positions - STREAM_OPEN_ANGLE_RAD)
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
        now_s = time.monotonic()
        detected_feedback = np.asarray(
            [
                detector.update(
                    commanded_velocity_rad_s=float(self.commanded_velocity[index]),
                    measured_velocity_rad_s=float(velocities[index]),
                    candidate_feedback=float(candidate_feedback[index]),
                    now_s=now_s,
                )
                for index, detector in enumerate(self.contact_detectors)
            ]
        )
        contacts = np.asarray(
            [detector.state is ContactState.CONTACT for detector in self.contact_detectors]
        )
        telemetry = Telemetry(
            positions=tuple(float(value) for value in positions),
            velocities=tuple(float(value) for value in velocities),
            currents_ma=tuple(float(value) for value in currents),
            targets=tuple(float(value) for value in targets),
            errors=tuple(float(value) for value in errors),
            candidate_feedback=tuple(float(value) for value in candidate_feedback),
            contact_feedback=tuple(float(value) for value in detected_feedback),
            contacts=tuple(bool(value) for value in contacts),
            sequence_active=self.sequence_active,
        )
        self.events.put(WorkerEvent("telemetry", telemetry))


class ContactVisualizerUI:
    """Tk interface for direct position control and contact visualization."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Two-joint contact visualizer")
        self.root.minsize(980, 680)
        self.commands: queue.Queue[tuple[str, object]] = queue.Queue()
        self.events: queue.Queue[WorkerEvent] = queue.Queue()
        self.worker = HandWorker(self.commands, self.events)
        self.worker.start()
        self.connected = False
        self.sequence_active = False
        self.latest_telemetry: Telemetry | None = None
        self._slider_job: str | None = None
        self._closing = False

        self._build_window()
        self._load_limits_from_config(show_error=False)
        self._draw_finger(0.0, 0.0, False, False)
        self.root.after(30, self._drain_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_window(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(1, weight=1)

        connection = ttk.Frame(outer)
        connection.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        connection.columnconfigure(1, weight=1)
        ttk.Label(connection, text="Config").grid(row=0, column=0, sticky="w")
        self.config_var = tk.StringVar(value=str(DEFAULT_CONFIG_PATH))
        ttk.Entry(connection, textvariable=self.config_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(connection, text="Browse…", command=self._browse_config).grid(
            row=0, column=2
        )
        ttk.Label(connection, text="MCP ID").grid(row=0, column=3, padx=(16, 4))
        self.mcp_id_var = tk.StringVar(value=str(DEFAULT_MCP_ID))
        ttk.Entry(connection, textvariable=self.mcp_id_var, width=5).grid(row=0, column=4)
        ttk.Label(connection, text="PIP ID").grid(row=0, column=5, padx=(12, 4))
        self.pip_id_var = tk.StringVar(value=str(DEFAULT_PIP_ID))
        ttk.Entry(connection, textvariable=self.pip_id_var, width=5).grid(row=0, column=6)
        self.connect_button = ttk.Button(
            connection, text="Connect + enable torque", command=self._toggle_connection
        )
        self.connect_button.grid(row=0, column=7, padx=(16, 0))

        visual = ttk.LabelFrame(outer, text="Live finger", padding=8)
        visual.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        visual.rowconfigure(0, weight=1)
        visual.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            visual, background="#10151c", highlightthickness=0, width=560, height=500
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._redraw_latest())

        controls = ttk.Frame(outer)
        controls.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        controls.columnconfigure(0, weight=1)

        contact_frame = ttk.LabelFrame(controls, text="Contact state", padding=10)
        contact_frame.grid(row=0, column=0, sticky="ew")
        contact_frame.columnconfigure(0, weight=1)
        contact_frame.columnconfigure(1, weight=1)
        self.mcp_contact_label = tk.Label(
            contact_frame, text="MCP\nNO CONTACT", bg="#25313d", fg="white",
            font=("TkDefaultFont", 13, "bold"), padx=12, pady=12,
        )
        self.mcp_contact_label.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.pip_contact_label = tk.Label(
            contact_frame, text="PIP\nNO CONTACT", bg="#25313d", fg="white",
            font=("TkDefaultFont", 13, "bold"), padx=12, pady=12,
        )
        self.pip_contact_label.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        sliders = ttk.LabelFrame(controls, text="Joint targets", padding=10)
        sliders.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        sliders.columnconfigure(0, weight=1)
        self.mcp_target_var = tk.DoubleVar(value=0.0)
        self.pip_target_var = tk.DoubleVar(value=0.0)
        self.mcp_value_var = tk.StringVar(value="0.000 rad")
        self.pip_value_var = tk.StringVar(value="0.000 rad")
        self.mcp_scale = self._add_slider(
            sliders, 0, "MCP", self.mcp_target_var, self.mcp_value_var
        )
        self.pip_scale = self._add_slider(
            sliders, 2, "PIP", self.pip_target_var, self.pip_value_var
        )

        detector = ttk.LabelFrame(controls, text="Contact detector", padding=10)
        detector.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        detector_summary = (
            f"Entry / exit error: {CONTACT_ENTER_THRESHOLD:g} / "
            f"{CONTACT_EXIT_THRESHOLD:g} rad\n"
            f"Motion suppression: > {CONTACT_MAX_VELOCITY_RAD_S:g} rad/s\n"
            f"Settle / entry persistence: {CONTACT_SETTLE_TIME_S:g} / "
            f"{CONTACT_PERSISTENCE_TIME_S:g} s\n"
            f"Minimum contact / exit persistence: {CONTACT_MIN_ENGAGED_TIME_S:g} / "
            f"{CONTACT_EXIT_PERSISTENCE_TIME_S:g} s"
        )
        ttk.Label(detector, text=detector_summary, justify="left").grid(
            row=0, column=0, sticky="w"
        )

        sequence = ttk.LabelFrame(
            controls, text="Streamed open-close-open sequence", padding=10
        )
        sequence.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        sequence.columnconfigure(1, weight=1)
        self.sequence_duration_var = tk.StringVar(value="2.0")
        self._labeled_entry(
            sequence, 0, "Time per leg", self.sequence_duration_var, "seconds"
        )
        self.sequence_button = ttk.Button(
            sequence,
            text="Run open → close → open",
            command=self._toggle_sequence,
            state="disabled",
        )
        self.sequence_button.grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(7, 0)
        )

        readings = ttk.LabelFrame(controls, text="Live readings (MCP / PIP)", padding=10)
        readings.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        self.readings_var = tk.StringVar(
            value=(
                "Position:  — / — rad\nError:     — / — rad\n"
                "Candidate: — / —\nFeedback:  — / —\nCurrent:   — / — mA"
            )
        )
        ttk.Label(readings, textvariable=self.readings_var, font="TkFixedFont").grid(
            row=0, column=0, sticky="w"
        )

        self.status_var = tk.StringVar(value="Disconnected — torque disabled")
        ttk.Label(outer, textvariable=self.status_var, anchor="w").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0)
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
            ids = (int(self.mcp_id_var.get()), int(self.pip_id_var.get()))
            subset = config.subset(ids)
            indices = _ui_order_indices(list(subset.motor_ids), *ids)
            lower = tuple(subset.joint_lower_limits[index] for index in indices)
            upper = tuple(subset.joint_upper_limits[index] for index in indices)
            _validated_default_targets(lower, upper)
            self.mcp_scale.configure(from_=lower[0], to=upper[0])
            self.pip_scale.configure(from_=lower[1], to=upper[1])
            if not self.connected:
                self.mcp_target_var.set(DEFAULT_JOINT_TARGETS_RAD[0])
                self.pip_target_var.set(DEFAULT_JOINT_TARGETS_RAD[1])
                self.mcp_value_var.set("0.000 rad")
                self.pip_value_var.set("0.000 rad")
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
            settings = ConnectionSettings(
                Path(self.config_var.get()).expanduser(),
                int(self.mcp_id_var.get()),
                int(self.pip_id_var.get()),
            )
            if not settings.config_path.is_file():
                raise ValueError(f"Configuration does not exist: {settings.config_path}")
        except ValueError as exc:
            messagebox.showerror("Invalid connection settings", str(exc))
            return
        if not self._load_limits_from_config(show_error=True):
            return
        if not messagebox.askokcancel(
            "Enable physical motion",
            "Connecting will configure both motors, enable torque, and move the "
            "MCP and PIP targets to 0 rad. Keep the finger clear, verify the motor "
            "IDs, and have an emergency stop ready.",
            icon="warning",
        ):
            return
        self.connect_button.configure(state="disabled")
        self.status_var.set("Connecting…")
        self.commands.put(("connect", settings))

    def _slider_changed(self) -> None:
        self.mcp_value_var.set(f"{self.mcp_target_var.get():.3f} rad")
        self.pip_value_var.set(f"{self.pip_target_var.get():.3f} rad")
        # Throttle to the hardware control rate instead of debouncing. Cancelling
        # and restarting this callback on every drag event delays the command
        # until the user releases the slider, which makes control feel laggy.
        if self._slider_job is None:
            interval_ms = max(1, round(1000.0 / CONTROL_RATE_HZ))
            self._slider_job = self.root.after(interval_ms, self._send_targets)

    def _send_targets(self) -> None:
        self._slider_job = None
        if self.connected and not self.sequence_active:
            self.commands.put(
                ("targets", (self.mcp_target_var.get(), self.pip_target_var.get()))
            )

    def _toggle_sequence(self) -> None:
        if self.sequence_active:
            self.commands.put(("stop_sequence", None))
            return
        try:
            duration_s = float(self.sequence_duration_var.get())
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
            "Run open-close-open sequence",
            "Both selected joints will move to 0 rad, stream to -1.35 rad, and "
            f"stream back to 0 rad. Each leg takes {duration_s:g} seconds. "
            "Make sure the full finger workspace is clear.",
            icon="warning",
        ):
            return
        self.commands.put(("start_sequence", duration_s))

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
            self.sequence_button.configure(state="normal")
            self.mcp_scale.configure(state="normal")
            self.pip_scale.configure(state="normal")
            self.mcp_scale.configure(from_=lower[0], to=upper[0])
            self.pip_scale.configure(from_=lower[1], to=upper[1])
            self.mcp_target_var.set(targets[0])
            self.pip_target_var.set(targets[1])
            self._slider_changed()
        elif event.kind == "disconnected":
            self.connected = False
            self.sequence_active = False
            self.connect_button.configure(text="Connect + enable torque", state="normal")
            self.sequence_button.configure(
                text="Run open → close → open", state="disabled"
            )
            self.mcp_scale.configure(state="normal")
            self.pip_scale.configure(state="normal")
            self.status_var.set("Disconnected — torque disabled")
            self._set_contact_labels(False, False)
        elif event.kind == "error":
            self.connected = False
            self.sequence_active = False
            self.connect_button.configure(text="Connect + enable torque", state="normal")
            self.sequence_button.configure(
                text="Run open → close → open", state="disabled"
            )
            self.mcp_scale.configure(state="normal")
            self.pip_scale.configure(state="normal")
            self.status_var.set(f"Error: {event.payload}")
            if not self._closing:
                messagebox.showerror("Hand controller error", str(event.payload))

    def _show_telemetry(self, telemetry: Telemetry) -> None:
        self.latest_telemetry = telemetry
        sequence_changed = self.sequence_active != telemetry.sequence_active
        self.sequence_active = telemetry.sequence_active
        self.sequence_button.configure(
            text=(
                "Stop sequence"
                if telemetry.sequence_active
                else "Run open → close → open"
            )
        )
        slider_state = "disabled" if telemetry.sequence_active else "normal"
        self.mcp_scale.configure(state=slider_state)
        self.pip_scale.configure(state=slider_state)
        if sequence_changed and not telemetry.sequence_active:
            self.mcp_target_var.set(telemetry.targets[0])
            self.pip_target_var.set(telemetry.targets[1])
            self.mcp_value_var.set(f"{telemetry.targets[0]:.3f} rad")
            self.pip_value_var.set(f"{telemetry.targets[1]:.3f} rad")
        mcp_contact, pip_contact = telemetry.contacts
        self._set_contact_labels(mcp_contact, pip_contact)
        self._draw_finger(
            telemetry.positions[0], telemetry.positions[1], mcp_contact, pip_contact
        )
        self.readings_var.set(
            f"Position:  {telemetry.positions[0]:7.3f} / {telemetry.positions[1]:7.3f} rad\n"
            f"Error:     {telemetry.errors[0]:7.3f} / {telemetry.errors[1]:7.3f} rad\n"
            f"Candidate: {telemetry.candidate_feedback[0]:7.3f} / "
            f"{telemetry.candidate_feedback[1]:7.3f}\n"
            f"Feedback:  {telemetry.contact_feedback[0]:7.3f} / "
            f"{telemetry.contact_feedback[1]:7.3f}\n"
            f"Current:   {telemetry.currents_ma[0]:7.1f} / {telemetry.currents_ma[1]:7.1f} mA"
        )

    def _set_contact_labels(self, mcp_contact: bool, pip_contact: bool) -> None:
        for label, name, contact in (
            (self.mcp_contact_label, "MCP", mcp_contact),
            (self.pip_contact_label, "PIP", pip_contact),
        ):
            label.configure(
                text=f"{name}\n{'CONTACT' if contact else 'NO CONTACT'}",
                bg="#ff3b30" if contact else "#25313d",
                fg="white",
            )

    def _redraw_latest(self) -> None:
        telemetry = self.latest_telemetry
        if telemetry is None:
            self._draw_finger(0.0, 0.0, False, False)
        else:
            self._draw_finger(
                telemetry.positions[0], telemetry.positions[1],
                telemetry.contacts[0], telemetry.contacts[1],
            )

    def _draw_finger(
        self,
        mcp_angle: float,
        pip_angle: float,
        mcp_contact: bool,
        pip_contact: bool,
    ) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 300)
        height = max(self.canvas.winfo_height(), 300)
        base = np.array((width * 0.24, height * 0.80))
        proximal_angle = -math.pi / 2.0 - mcp_angle
        distal_angle = proximal_angle - pip_angle
        proximal_length = min(width * 0.36, height * 0.39)
        distal_length = proximal_length * 0.82
        joint = base + proximal_length * np.array(
            (math.cos(proximal_angle), math.sin(proximal_angle))
        )
        tip = joint + distal_length * np.array(
            (math.cos(distal_angle), math.sin(distal_angle))
        )

        self.canvas.create_text(
            16, 16, anchor="nw", fill="#8fa3b8", text="Measured joint pose",
            font=("TkDefaultFont", 11, "bold"),
        )
        self._draw_segment(base, joint, "MCP", mcp_contact)
        self._draw_segment(joint, tip, "PIP", pip_contact)
        for point in (base, joint):
            x, y = point
            self.canvas.create_oval(
                x - 10, y - 10, x + 10, y + 10,
                fill="#d8e0e8", outline="#10151c", width=3,
            )
        x, y = tip
        self.canvas.create_oval(
            x - 7, y - 7, x + 7, y + 7,
            fill="#d8e0e8", outline="#10151c", width=2,
        )

    def _draw_segment(
        self, start: np.ndarray, end: np.ndarray, label: str, contact: bool
    ) -> None:
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length == 0:
            return
        normal = np.array((-delta[1], delta[0])) / length
        half_width = 24.0
        corners = (
            start + normal * half_width,
            end + normal * half_width,
            end - normal * half_width,
            start - normal * half_width,
        )
        points = [coordinate for point in corners for coordinate in point]
        if contact:
            self.canvas.create_polygon(
                points, fill="", outline="#ff9f0a", width=12, joinstyle="round"
            )
        self.canvas.create_polygon(
            points,
            fill="#ff453a" if contact else "#3a86a8",
            outline="#ffd166" if contact else "#83c5d8",
            width=4,
            joinstyle="round",
        )
        midpoint = (start + end) / 2.0
        self.canvas.create_text(
            midpoint[0], midpoint[1], text=label, fill="white",
            font=("TkDefaultFont", 12, "bold"),
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
        self.sequence_button.configure(state="disabled")
        self.commands.put(("stop", None))
        self._finish_close()

    def _finish_close(self) -> None:
        if self.worker.is_alive():
            self.root.after(50, self._finish_close)
        else:
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ui = ContactVisualizerUI(root)
    try:
        root.mainloop()
    finally:
        if ui.worker.is_alive():
            ui.commands.put(("stop", None))
            ui.worker.join()


if __name__ == "__main__":
    main()
