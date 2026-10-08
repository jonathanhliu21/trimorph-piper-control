from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Sequence
from tkinter import ttk

import numpy as np


# Physical finger tuples are PIP, MCP, AB/AD. The visual layout uses MCP, PIP,
# AB/AD within each finger, independently of the configuration's motor order.
FINGER_MOTOR_IDS = ((4, 5, 6), (7, 8, 9), (10, 11, 12))
FINGER_UI_MOTOR_IDS = tuple(
    motor_id
    for pip_id, mcp_id, ab_ad_id in FINGER_MOTOR_IDS
    for motor_id in (mcp_id, pip_id, ab_ad_id)
)
THUMB_MOTOR_IDS = (0, 1, 2, 3)
UI_MOTOR_IDS = FINGER_UI_MOTOR_IDS + THUMB_MOTOR_IDS
JOINT_COUNT = len(UI_MOTOR_IDS)
THUMB_INDICES = (9, 10, 11, 12)


def contact_circle_radius(contact_feedback_value: float) -> float:
    """Map absolute contact feedback to a bounded on-canvas radius."""

    return min(120.0, 22.0 + 420.0 * abs(float(contact_feedback_value)))


def contact_palette(contact_feedback_value: float) -> tuple[str, str, str, str]:
    """Return fill, outline, ring, and accent colors for feedback direction."""

    if contact_feedback_value > 0:
        return "#74c0fc", "#1864ab", "#1971c2", "#a5d8ff"
    if contact_feedback_value < 0:
        return "#ff8a65", "#c92a2a", "#e03131", "#ffc9c9"
    return "#b197fc", "#6741d9", "#7048e8", "#d0bfff"


def contact_direction_label(contact_feedback_value: float) -> str:
    if contact_feedback_value > 0:
        return "CONTACT  +"
    if contact_feedback_value < 0:
        return "CONTACT  −"
    return "CONTACT"


def contact_vectors_by_motor_id(
    motor_ids: Sequence[int],
    contact_feedback: Sequence[float],
    normalized_joint_difference: Sequence[float],
) -> tuple[tuple[bool, ...], tuple[float, ...], tuple[float, ...]]:
    """Map follower-order telemetry into the fixed visual motor order."""

    if not (
        len(motor_ids)
        == len(contact_feedback)
        == len(normalized_joint_difference)
    ):
        raise ValueError("Follower contact telemetry vectors must align")
    if len(set(int(motor_id) for motor_id in motor_ids)) != len(motor_ids):
        raise ValueError("Follower contact telemetry motor IDs must be unique")

    by_id = {
        int(motor_id): (float(feedback), float(normalized))
        for motor_id, feedback, normalized in zip(
            motor_ids, contact_feedback, normalized_joint_difference
        )
    }
    strengths = tuple(by_id.get(motor_id, (0.0, 0.0))[0] for motor_id in UI_MOTOR_IDS)
    normalized = tuple(by_id.get(motor_id, (0.0, 0.0))[1] for motor_id in UI_MOTOR_IDS)
    return tuple(value != 0.0 for value in strengths), strengths, normalized


class LiveContactWidget(ttk.LabelFrame):
    """Read-only hand contact diagram shared by controller and collection UIs."""

    def __init__(self, parent: tk.Misc, **kwargs: object) -> None:
        super().__init__(parent, text="Live contact", padding=8, **kwargs)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            self, background="#f7f7f5", highlightthickness=0, height=420
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._contacts = (False,) * JOINT_COUNT
        self._contact_strengths = (0.0,) * JOINT_COUNT
        self._normalized_differences = (0.0,) * JOINT_COUNT
        self._tactile_contacts: tuple[bool, ...] | None = None
        self.canvas.bind("<Configure>", lambda _event: self._draw())
        self._draw()

    def show(
        self,
        contacts: Sequence[bool] = (False,) * JOINT_COUNT,
        contact_strengths: Sequence[float] = (0.0,) * JOINT_COUNT,
        normalized_differences: Sequence[float] = (0.0,) * JOINT_COUNT,
        tactile_contacts: Sequence[bool] | None = None,
    ) -> None:
        if not (
            len(contacts)
            == len(contact_strengths)
            == len(normalized_differences)
            == JOINT_COUNT
        ):
            raise ValueError(f"Live contact vectors must contain {JOINT_COUNT} joints")
        if tactile_contacts is not None and len(tactile_contacts) != 4:
            raise ValueError("Tactile contacts must contain three fingers and a thumb")
        self._contacts = tuple(bool(value) for value in contacts)
        self._contact_strengths = tuple(float(value) for value in contact_strengths)
        self._normalized_differences = tuple(
            float(value) for value in normalized_differences
        )
        self._tactile_contacts = (
            None
            if tactile_contacts is None
            else tuple(bool(value) for value in tactile_contacts)
        )
        self._draw()

    def show_by_motor_id(
        self,
        motor_ids: Sequence[int],
        contact_feedback: Sequence[float],
        normalized_joint_difference: Sequence[float],
        tactile_contacts: Sequence[bool] | None = None,
    ) -> None:
        self.show(
            *contact_vectors_by_motor_id(
                motor_ids, contact_feedback, normalized_joint_difference
            ),
            tactile_contacts=tactile_contacts,
        )

    def _draw(self) -> None:
        contacts = self._contacts
        contact_strengths = self._contact_strengths
        normalized_differences = self._normalized_differences
        tactile_contacts = self._tactile_contacts
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

        for x, y, label in (
            (22, height / 2, "MCP"),
            (22, pip_center_y, "PIP"),
            (28, ab_ad_center_y, "AB/AD"),
        ):
            self.canvas.create_text(
                x, y, text=label, fill="#333333",
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
            if tactile_contacts is not None:
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
                contacts[ab_ad_index], contact_strengths[ab_ad_index],
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
            if show_tactile else (0.20, 0.45, 0.68, 0.88)
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
        if tactile_contacts is not None:
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
        self, center: np.ndarray, width: float, height: float, label: str,
        contact: bool, contact_feedback_value: float,
    ) -> None:
        x, y = center
        fill, outline, _, _ = contact_palette(contact_feedback_value)
        self.canvas.create_rectangle(
            x - width / 2, y - height / 2, x + width / 2, y + height / 2,
            fill=fill if contact else "#ffffff",
            outline=outline if contact else "#333333", width=4,
        )
        state = contact_direction_label(contact_feedback_value) if contact else "NO CONTACT"
        self.canvas.create_text(
            x, y, text=f"{label}\n{state}", fill="#1f1f1f",
            justify="center", font=("TkDefaultFont", 11, "bold"),
        )

    def _draw_rotated_joint_block(
        self, center: np.ndarray, width: float, height: float, label: str,
        contact: bool, contact_feedback_value: float, angle_degrees: float,
    ) -> None:
        angle = math.radians(angle_degrees)
        rotation = np.asarray(
            ((math.cos(angle), -math.sin(angle)),
             (math.sin(angle), math.cos(angle)))
        )
        corners = np.asarray((
            (-width / 2, -height / 2), (width / 2, -height / 2),
            (width / 2, height / 2), (-width / 2, height / 2),
        ))
        points = (corners @ rotation.T) + center
        fill, outline, _, _ = contact_palette(contact_feedback_value)
        self.canvas.create_polygon(
            *points.ravel(), fill=fill if contact else "#ffffff",
            outline=outline if contact else "#333333", width=4,
        )
        state = contact_direction_label(contact_feedback_value) if contact else "NO CONTACT"
        self.canvas.create_text(
            center[0], center[1], text=f"{label}\n{state}", fill="#1f1f1f",
            justify="center", angle=angle_degrees,
            font=("TkDefaultFont", 9, "bold"),
        )

    def _draw_round_joint(
        self, center: np.ndarray, label: str, contact: bool,
        contact_feedback_value: float,
    ) -> None:
        x, y = center
        radius = 43.0
        fill, outline, _, _ = contact_palette(contact_feedback_value)
        self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            fill=fill if contact else "#ffffff",
            outline=outline if contact else "#333333", width=4,
        )
        state = contact_direction_label(contact_feedback_value) if contact else "NO CONTACT"
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
            fill, outline, _, _ = contact_palette(feedback)
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
        radius = contact_circle_radius(strength)
        _, _, ring, accent = contact_palette(strength)
        x, y = center
        bounds = (x - radius, y - radius, x + radius, y + radius)
        self.canvas.create_oval(*bounds, outline=ring, width=9)
        self.canvas.create_oval(*bounds, outline=accent, width=3)
        self.canvas.create_text(
            x, y + radius + 13,
            text=f"{normalized_difference:.2f}× threshold", fill="#9a2f20",
            font=("TkDefaultFont", 11, "bold"),
        )

    def _draw_tactile_contact_indicator(
        self, center: np.ndarray, contact: bool
    ) -> None:
        x, y = center
        radius = 24.0
        self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            fill="#ff8a65" if contact else "#ffffff",
            outline="#c92a2a" if contact else "#666666", width=3,
        )
        self.canvas.create_text(
            x, y, text="CONTACT" if contact else "NO\nCONTACT", fill="#1f1f1f",
            font=("TkDefaultFont", 8, "bold"), justify="center",
        )
