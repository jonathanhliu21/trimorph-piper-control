import csv
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

from .robot_helpers import collect_robot_state


FIELDNAMES = [
    "elapsed_s",
    "wall_time_s",
    "replay_elapsed_s",
    "source",
    "event",
    "status_timestamp",
    "status_hz",
    "ctrl_mode",
    "arm_status",
    "mode_feedback",
    "teach_status",
    "motion_status",
    "trajectory_num",
    "err_code",
    "is_ok",
    "receive_fps",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "target_joint_1",
    "target_joint_2",
    "target_joint_3",
    "target_joint_4",
    "target_joint_5",
    "target_joint_6",
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
]


def _collect_state(robot) -> Dict[str, object]:
    get_state = getattr(robot, "get_state", None)
    if callable(get_state):
        return get_state()
    return collect_robot_state(robot)


class DaggerCsvRecorder:
    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None
        self._start_t = None
        self.rows_written = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def open(self) -> None:
        if self._file is not None:
            return
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        self._start_t = time.monotonic()

    def close(self) -> None:
        if self._file is None:
            return
        self._file.flush()
        self._file.close()
        self._file = None
        self._writer = None

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def write(
        self,
        *,
        robot,
        source: str,
        replay_elapsed_s: Optional[float],
        target_joints: Optional[Iterable[float]] = None,
        event: str = "",
    ) -> None:
        if self._writer is None or self._start_t is None:
            raise RuntimeError("Recorder is not open")

        now = time.monotonic()
        row: Dict[str, object] = {field: "" for field in FIELDNAMES}
        row.update(
            {
                "elapsed_s": f"{now - self._start_t:.6f}",
                "wall_time_s": f"{time.time():.6f}",
                "replay_elapsed_s": (
                    "" if replay_elapsed_s is None else f"{replay_elapsed_s:.6f}"
                ),
                "source": source,
                "event": event,
            }
        )
        row.update(_collect_state(robot))

        if target_joints is not None:
            for i, value in enumerate(target_joints, start=1):
                row[f"target_joint_{i}"] = f"{value:.9f}"

        self._writer.writerow(row)
        self.rows_written += 1
