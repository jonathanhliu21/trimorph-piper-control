"""Binary tactile contact detection from baseline-relative normal force."""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

import numpy as np


logger = logging.getLogger(__name__)


TACTILE_BASELINE_WINDOW_S = 0.50
TACTILE_MEDIAN_WINDOW_SAMPLES = 5
TACTILE_ENTRY_THRESHOLD_N = 2.5
TACTILE_RELEASE_THRESHOLD_N = 1.0
TACTILE_ENTRY_PERSISTENCE_S = 0.06
TACTILE_RELEASE_PERSISTENCE_S = 0.10
TACTILE_BASELINE_ADAPTATION_TAU_S = 10.0
TACTILE_MIN_BASELINE_SAMPLES = 5


@dataclass(frozen=True)
class TactileContactConfig:
    """Tuning for one baseline-relative total-force contact detector.

    Thresholds are measured above the learned no-contact baseline. Keeping the
    release threshold below the entry threshold provides hysteresis, so the
    Boolean state cannot chatter around one threshold.
    """

    baseline_window_s: float = TACTILE_BASELINE_WINDOW_S
    median_window_samples: int = TACTILE_MEDIAN_WINDOW_SAMPLES
    entry_threshold_n: float = TACTILE_ENTRY_THRESHOLD_N
    release_threshold_n: float = TACTILE_RELEASE_THRESHOLD_N
    entry_persistence_s: float = TACTILE_ENTRY_PERSISTENCE_S
    release_persistence_s: float = TACTILE_RELEASE_PERSISTENCE_S
    baseline_adaptation_tau_s: float = TACTILE_BASELINE_ADAPTATION_TAU_S
    min_baseline_samples: int = TACTILE_MIN_BASELINE_SAMPLES

    def validate(self) -> None:
        positive = {
            "baseline_window_s": self.baseline_window_s,
            "entry_threshold_n": self.entry_threshold_n,
            "release_threshold_n": self.release_threshold_n,
            "baseline_adaptation_tau_s": self.baseline_adaptation_tau_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.release_threshold_n >= self.entry_threshold_n:
            raise ValueError(
                "release_threshold_n must be lower than entry_threshold_n "
                "to provide contact hysteresis"
            )
        for name, value in (
            ("entry_persistence_s", self.entry_persistence_s),
            ("release_persistence_s", self.release_persistence_s),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in (
            ("median_window_samples", self.median_window_samples),
            ("min_baseline_samples", self.min_baseline_samples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class TactileContactDetector:
    """Detect contact from median-filtered force above a learned baseline.

    Normal force is summed across all taxels and passed through a short rolling
    median. During startup, the median of a no-contact baseline window is
    learned. Contact enters above a high threshold and releases below a lower
    threshold, each with independent persistence. While unlatched and below
    the entry threshold, the baseline follows slow drift.
    """

    def __init__(
        self,
        config: TactileContactConfig | None = None,
        *,
        name: str = "unnamed",
    ) -> None:
        self.config = config or TactileContactConfig()
        self.config.validate()
        self.name = name
        self.contact = False
        self.ready = False
        self.raw_total_force_n = 0.0
        self.filtered_total_force_n = 0.0
        self.baseline_force_n = 0.0
        self.baseline_delta_n = 0.0
        self._taxel_count: int | None = None
        self._history_started_s: float | None = None
        self._last_update_s: float | None = None
        self._entry_started_s: float | None = None
        self._release_started_s: float | None = None
        self._median_samples: deque[float] = deque(
            maxlen=self.config.median_window_samples
        )
        self._baseline_samples: list[float] = []

    def reset(self) -> None:
        if self.ready or self.contact or self._median_samples:
            logger.info(
                "TACTILE_CONTACT sensor=%s event=reset contact=%d "
                "baseline=%.3f filtered_force=%.3f",
                self.name,
                self.contact,
                self.baseline_force_n,
                self.filtered_total_force_n,
            )
        self.contact = False
        self.ready = False
        self.raw_total_force_n = 0.0
        self.filtered_total_force_n = 0.0
        self.baseline_force_n = 0.0
        self.baseline_delta_n = 0.0
        self._taxel_count = None
        self._history_started_s = None
        self._last_update_s = None
        self._entry_started_s = None
        self._release_started_s = None
        self._median_samples.clear()
        self._baseline_samples.clear()

    def _debug_sample(self, *, event: str, now_s: float) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        entry_elapsed_s = (
            0.0
            if self._entry_started_s is None
            else now_s - self._entry_started_s
        )
        release_elapsed_s = (
            0.0
            if self._release_started_s is None
            else now_s - self._release_started_s
        )
        logger.debug(
            "TACTILE_CONTACT sensor=%s t=%.6f event=%s contact=%d ready=%d "
            "taxels=%d raw_force=%.3f median_force=%.3f baseline=%.3f "
            "baseline_delta=%.3f entry_threshold=%.3f "
            "release_threshold=%.3f median_samples=%d baseline_samples=%d "
            "entry_elapsed=%.3f/%.3f release_elapsed=%.3f/%.3f",
            self.name,
            now_s,
            event,
            self.contact,
            self.ready,
            self._taxel_count or 0,
            self.raw_total_force_n,
            self.filtered_total_force_n,
            self.baseline_force_n,
            self.baseline_delta_n,
            self.config.entry_threshold_n,
            self.config.release_threshold_n,
            len(self._median_samples),
            len(self._baseline_samples),
            entry_elapsed_s,
            self.config.entry_persistence_s,
            release_elapsed_s,
            self.config.release_persistence_s,
        )

    def update(self, forces_n: np.ndarray, *, now_s: float) -> bool:
        values = np.asarray(forces_n, dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("forces_n must contain finite taxel readings")
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        if self._last_update_s is not None and now_s <= self._last_update_s:
            raise ValueError("now_s must increase on every update")
        if self._taxel_count is None:
            self._taxel_count = len(values)
            self._history_started_s = now_s
        elif len(values) != self._taxel_count:
            raise ValueError(
                f"Taxel count changed from {self._taxel_count} to {len(values)}"
            )

        previous_update_s = self._last_update_s
        self._last_update_s = now_s
        self.raw_total_force_n = float(np.sum(values))
        self._median_samples.append(self.raw_total_force_n)
        self.filtered_total_force_n = float(np.median(self._median_samples))

        if not self.ready:
            self._baseline_samples.append(self.filtered_total_force_n)
            assert self._history_started_s is not None
            baseline_elapsed_s = now_s - self._history_started_s
            baseline_complete = (
                baseline_elapsed_s >= self.config.baseline_window_s - 1e-9
                and len(self._baseline_samples) >= self.config.min_baseline_samples
                and len(self._median_samples) == self.config.median_window_samples
            )
            self.baseline_force_n = float(np.median(self._baseline_samples))
            self.baseline_delta_n = (
                self.filtered_total_force_n - self.baseline_force_n
            )
            if baseline_complete:
                self.ready = True
                event = "baseline_ready"
                logger.info(
                    "TACTILE_CONTACT sensor=%s event=baseline_ready t=%.6f "
                    "baseline=%.3f samples=%d",
                    self.name,
                    now_s,
                    self.baseline_force_n,
                    len(self._baseline_samples),
                )
            else:
                event = "baseline_learning"
            self._debug_sample(event=event, now_s=now_s)
            return False

        self.baseline_delta_n = self.filtered_total_force_n - self.baseline_force_n

        if not self.contact:
            self._release_started_s = None
            if self.baseline_delta_n < self.config.entry_threshold_n:
                event = (
                    "entry_cancel"
                    if self._entry_started_s is not None
                    else "no_contact"
                )
                self._entry_started_s = None
                if previous_update_s is not None:
                    elapsed_s = now_s - previous_update_s
                    alpha = 1.0 - math.exp(
                        -elapsed_s / self.config.baseline_adaptation_tau_s
                    )
                    self.baseline_force_n += alpha * (
                        self.filtered_total_force_n - self.baseline_force_n
                    )
                    self.baseline_delta_n = (
                        self.filtered_total_force_n - self.baseline_force_n
                    )
                self._debug_sample(event=event, now_s=now_s)
                return False

            if self._entry_started_s is None:
                self._entry_started_s = now_s
                event = "entry_start"
            else:
                event = "entry_pending"
            if now_s - self._entry_started_s >= self.config.entry_persistence_s:
                self.contact = True
                event = "contact_enter"
                logger.info(
                    "TACTILE_CONTACT sensor=%s event=contact_enter t=%.6f "
                    "baseline_delta=%.3f entry_threshold=%.3f",
                    self.name,
                    now_s,
                    self.baseline_delta_n,
                    self.config.entry_threshold_n,
                )
            self._debug_sample(event=event, now_s=now_s)
            if self.contact:
                self._entry_started_s = None
            return self.contact

        self._entry_started_s = None
        if self.baseline_delta_n > self.config.release_threshold_n:
            event = (
                "release_cancel"
                if self._release_started_s is not None
                else "contact_latched"
            )
            self._release_started_s = None
            self._debug_sample(event=event, now_s=now_s)
            return True

        if self._release_started_s is None:
            self._release_started_s = now_s
            event = "release_start"
        else:
            event = "release_pending"
        if now_s - self._release_started_s >= self.config.release_persistence_s:
            self.contact = False
            event = "contact_release"
            logger.info(
                "TACTILE_CONTACT sensor=%s event=contact_release t=%.6f "
                "baseline_delta=%.3f release_threshold=%.3f",
                self.name,
                now_s,
                self.baseline_delta_n,
                self.config.release_threshold_n,
            )
        self._debug_sample(event=event, now_s=now_s)
        if not self.contact:
            self._release_started_s = None
        return self.contact
