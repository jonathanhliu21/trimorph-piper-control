import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List


JOINT_FIELDS = [f"joint_{i}" for i in range(1, 7)]


@dataclass(frozen=True)
class MotionSample:
    elapsed_s: float
    joints: List[float]


def read_motion_csv(csv_path) -> List[MotionSample]:
    path = Path(csv_path).expanduser()
    samples: List[MotionSample] = []

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = {"elapsed_s", *JOINT_FIELDS} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            try:
                elapsed_s = float(row["elapsed_s"])
                joints = [float(row[field]) for field in JOINT_FIELDS]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {line_number}") from exc
            samples.append(MotionSample(elapsed_s=elapsed_s, joints=joints))

    samples.sort(key=lambda sample: sample.elapsed_s)
    return samples
