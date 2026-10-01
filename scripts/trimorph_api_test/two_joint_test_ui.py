"""Tkinter front end for all commands in :mod:`two_joint_test`.

Run this file from the repository root (or from any other directory):

    .venv/bin/python scripts/trimorph_api_test/two_joint_test_ui.py

The UI deliberately runs the existing CLI in a subprocess instead of
duplicating its hardware logic.  This keeps validation and shutdown behavior
in one place and allows long-running record/measurement commands to be stopped
from the UI.
"""

from __future__ import annotations

import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable


SCRIPT_PATH = Path(__file__).resolve().with_name("two_joint_test.py")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("xl330_test_config.yaml")
MOVEMENT_COMMANDS = {
    "home",
    "profile_move",
    "profile_move_sequential",
    "stream_move",
    "measure_current",
    "measure_position_error",
    "detect_contact",
}

COMMAND_LABELS = {
    "Home": "home",
    "Profile move": "profile_move",
    "Profile move (sequential)": "profile_move_sequential",
    "Stream move": "stream_move",
    "Record joint state": "record",
    "Measure current": "measure_current",
    "Measure position error": "measure_position_error",
    "Detect contact": "detect_contact",
}


class TwoJointTestUI:
    """Single-window launcher for the two-joint hardware test commands."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Trimorph two-joint test")
        self.root.minsize(780, 650)
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str | None] = queue.Queue()
        self.fields: dict[str, tk.StringVar] = {}
        self.checks: dict[str, tk.BooleanVar] = {}

        self._build_window()
        self._show_command_fields()
        self.root.after(75, self._drain_output)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_window(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(3, weight=1)

        ttk.Label(outer, text="Configuration").grid(row=0, column=0, sticky="w")
        self.config_var = tk.StringVar(value=str(DEFAULT_CONFIG_PATH))
        ttk.Entry(outer, textvariable=self.config_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 6)
        )
        ttk.Button(outer, text="Browse…", command=self._browse_config).grid(
            row=0, column=2, sticky="ew"
        )

        ttk.Label(outer, text="Fixture motor IDs").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.fixture_ids_var = tk.StringVar(value="4,5,6")
        ttk.Entry(outer, textvariable=self.fixture_ids_var).grid(
            row=1, column=1, sticky="ew", padx=(8, 6), pady=(8, 0)
        )
        ttk.Label(outer, text="comma-separated, exactly 3").grid(
            row=1, column=2, sticky="w", pady=(8, 0)
        )

        selector = ttk.Frame(outer)
        selector.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(14, 8))
        selector.columnconfigure(1, weight=1)
        ttk.Label(selector, text="Command").grid(row=0, column=0, sticky="w")
        self.command_label_var = tk.StringVar(value="Home")
        command_box = ttk.Combobox(
            selector,
            textvariable=self.command_label_var,
            values=list(COMMAND_LABELS),
            state="readonly",
        )
        command_box.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        command_box.bind("<<ComboboxSelected>>", self._show_command_fields)

        body = ttk.Panedwindow(outer, orient="vertical")
        body.grid(row=3, column=0, columnspan=3, sticky="nsew")

        self.options_frame = ttk.LabelFrame(body, text="Command options", padding=10)
        self.options_frame.columnconfigure(1, weight=1)
        body.add(self.options_frame, weight=1)

        output_frame = ttk.LabelFrame(body, text="Output", padding=6)
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        self.output = tk.Text(
            output_frame, height=15, wrap="word", state="disabled", font="TkFixedFont"
        )
        scroll = ttk.Scrollbar(output_frame, command=self.output.yview)
        self.output.configure(yscrollcommand=scroll.set)
        self.output.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        body.add(output_frame, weight=2)

        controls = ttk.Frame(outer)
        controls.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        controls.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Button(controls, text="Clear output", command=self._clear_output).grid(
            row=0, column=1, padx=6
        )
        self.stop_button = ttk.Button(
            controls, text="Stop", command=self._stop_process, state="disabled"
        )
        self.stop_button.grid(row=0, column=2, padx=6)
        self.run_button = ttk.Button(controls, text="Run", command=self._run)
        self.run_button.grid(row=0, column=3)

    @property
    def command(self) -> str:
        return COMMAND_LABELS[self.command_label_var.get()]

    def _show_command_fields(self, _event: object | None = None) -> None:
        for child in self.options_frame.winfo_children():
            child.destroy()
        self.fields.clear()
        self.checks.clear()

        command = self.command
        row = 0

        def entry(key: str, label: str, default: str, hint: str = "") -> None:
            nonlocal row
            value = tk.StringVar(value=default)
            self.fields[key] = value
            ttk.Label(self.options_frame, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=value).grid(
                row=row, column=1, sticky="ew", padx=8, pady=3
            )
            ttk.Label(self.options_frame, text=hint).grid(row=row, column=2, sticky="w")
            row += 1

        def check(key: str, label: str, default: bool = False) -> None:
            nonlocal row
            value = tk.BooleanVar(value=default)
            self.checks[key] = value
            ttk.Checkbutton(self.options_frame, text=label, variable=value).grid(
                row=row, column=0, columnspan=3, sticky="w", pady=3
            )
            row += 1

        if command == "home":
            entry("joints", "Joint indices", "", "blank = all; space-separated")
            entry("current-threshold-ma", "Current threshold", "200.0", "mA")
            entry("velocity-rad-s", "Homing velocity", "1.5", "rad/s")
        elif command != "profile_move_sequential":
            entry("motors", "Motor IDs", "", "blank = all; comma-separated")

        if command in ("profile_move", "profile_move_sequential"):
            entry("profile-velocity-rad-s", "Profile velocity", "3.0", "rad/s")
            entry("profile-acceleration-rad-s2", "Profile acceleration", "50.0", "rad/s²")
            entry("timeout-s", "Target timeout", "4.0", "seconds")
            entry("pause-s", "Pause at target", "0.5", "seconds")
            entry("trials", "Trials", "1")
            entry("output-current", "Telemetry CSV", "", "blank = disabled")
            self._add_browse_button(row - 1, "output-current", save=True)
            check("force", "Overwrite an existing telemetry CSV")
            check("skip-target-safety", "Continue when a target times out")
        elif command == "stream_move":
            entry("rate-hz", "Stream rate", "30.0", "Hz")
            entry("duration-s", "Duration per leg", "2.0", "seconds")
            entry("preposition-timeout-s", "Pre-position timeout", "4.0", "seconds")
            entry("settle-s", "Settle time", "0.5", "seconds")
            entry("trials", "Trials", "1")
            entry("output-current", "Telemetry CSV", "", "blank = disabled")
            self._add_browse_button(row - 1, "output-current", save=True)
            check("motion-profile", "Enable Dynamixel motion profile", True)
            check("force", "Overwrite an existing telemetry CSV")
            check("skip-target-safety", "Continue when a target times out")
        elif command == "record":
            entry("output", "Output CSV", "two_joint_angles.csv")
            self._add_browse_button(row - 1, "output", save=True)
            entry("rate-hz", "Sample rate", "20.0", "Hz")
            entry("duration-s", "Duration", "", "seconds; blank = until stopped")
            entry("print-every", "Print every", "10", "samples; 0 = never")
            check("force", "Overwrite an existing output CSV")
        elif command in ("measure_current", "measure_position_error", "detect_contact"):
            entry("timeout-s", "Target timeout", "4.0", "seconds")
            entry("rate-hz", "Sample rate", "20.0", "Hz")

    def _add_browse_button(self, row: int, key: str, *, save: bool) -> None:
        callback: Callable[[], None]
        callback = lambda: self._browse_path(key, save=save)
        ttk.Button(self.options_frame, text="Browse…", command=callback).grid(
            row=row, column=2, sticky="e"
        )

    def _browse_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select motor configuration",
            initialfile=Path(self.config_var.get()).name,
            filetypes=(("YAML files", "*.yaml *.yml"), ("All files", "*")),
        )
        if selected:
            self.config_var.set(selected)

    def _browse_path(self, key: str, *, save: bool) -> None:
        current = self.fields[key].get()
        selected = filedialog.asksaveasfilename(
            title="Select CSV output",
            initialfile=Path(current).name if current else "current.csv",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*")),
        ) if save else filedialog.askopenfilename()
        if selected:
            self.fields[key].set(selected)

    def _build_command(self) -> list[str]:
        command = [
            sys.executable,
            "-u",
            str(SCRIPT_PATH),
            "--config",
            self.config_var.get().strip(),
            "--motor-ids",
            self.fixture_ids_var.get().strip(),
            self.command,
        ]
        for key, variable in self.fields.items():
            value = variable.get().strip()
            if not value:
                continue
            if key == "joints":
                command.append("--joints")
                command.extend(value.replace(",", " ").split())
            else:
                command.extend((f"--{key}", value))
        for key, variable in self.checks.items():
            enabled = variable.get()
            if key == "motion-profile":
                command.append("--motion-profile" if enabled else "--no-motion-profile")
            elif enabled:
                command.append(f"--{key}")
        if self.command in MOVEMENT_COMMANDS:
            command.append("--yes")
        return command

    def _run(self) -> None:
        if self.process is not None:
            return
        if not self.config_var.get().strip() or not self.fixture_ids_var.get().strip():
            messagebox.showerror(
                "Missing setting",
                "Configuration and fixture motor IDs are required.",
            )
            return

        command = self._build_command()
        if self.command in MOVEMENT_COMMANDS:
            confirmed = messagebox.askokcancel(
                "Confirm physical motion",
                "This command will move physical hardware. Ensure the mechanism "
                "is clear and an emergency stop is available.\n\nRun command?",
                icon="warning",
            )
            if not confirmed:
                return

        self._append_output(f"$ {shlex.join(command)}\n")
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=(os.name != "nt"),
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
        except OSError as exc:
            self.process = None
            messagebox.showerror("Could not start command", str(exc))
            return

        self.status_var.set(f"Running {self.command}…")
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        threading.Thread(target=self._read_process_output, daemon=True).start()

    def _read_process_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.output_queue.put(line)
        return_code = process.wait()
        self.output_queue.put(f"\nProcess exited with code {return_code}.\n")
        self.output_queue.put(None)

    def _drain_output(self) -> None:
        try:
            while True:
                item = self.output_queue.get_nowait()
                if item is None:
                    self.process = None
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status_var.set("Ready")
                else:
                    self._append_output(item)
        except queue.Empty:
            pass
        self.root.after(75, self._drain_output)

    def _append_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.configure(state="disabled")

    def _clear_output(self) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")

    def _stop_process(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        self.status_var.set("Stopping…")
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGINT)
        except (OSError, ProcessLookupError):
            process.terminate()

    def _on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno(
                "Command still running", "Stop the running command and close the UI?"
            ):
                return
            self._stop_process()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TwoJointTestUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
