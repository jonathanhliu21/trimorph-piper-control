#!/usr/bin/env bash
# Generate current and position plots beside a telemetry CSV.
set -euo pipefail

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    echo "Usage: $0 CSV"
    echo "Creates <CSV stem>_all.png and <CSV stem>_positions.png beside CSV."
    exit 0
fi
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 CSV" >&2
    exit 2
fi
if [[ ! -f "$1" ]]; then
    echo "CSV file does not exist: $1" >&2
    exit 1
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${MPLCONFIGDIR:-}" ]]; then
    graph_cache="$(mktemp -d /tmp/trimorph-graphs.XXXXXX)"
    export MPLCONFIGDIR="$graph_cache"
    trap 'rm -rf -- "$graph_cache"' EXIT
fi

python3 - "$1" "$repo_root" <<'PY'
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, sys.argv[2])
from scripts.trimorph_api_test.gen_graphs_from_csvs import (
    FINGER_MOTOR_IDS,
    MOTOR_LABELS,
    _load_telemetry,
    _sequential_trials,
)

source = Path(sys.argv[1]).resolve()
trials, times, motors = _load_telemetry(source)
missing = sorted(set(range(13)) - set(motors))
if missing:
    raise SystemExit(f"CSV is missing motor IDs: {missing}")
times, motors = _sequential_trials(trials, times, motors)

for metric, suffix, title, ylabel in (
    ("current", "all", "Motor currents", "Current (mA)"),
    ("position", "positions", "Expected vs actual position", "Position (rad)"),
):
    figure, axes = plt.subplots(
        4, 4, figsize=(22, 13), sharex=True, squeeze=False
    )
    for column, (finger, motor_ids) in enumerate(FINGER_MOTOR_IDS.items()):
        for row in range(4):
            axis = axes[row, column]
            if row >= len(motor_ids):
                axis.set_visible(False)
                continue
            motor_id = motor_ids[row]
            values = motors[motor_id]
            if metric == "current":
                axis.plot(times, values["current"], color=f"C{column}", linewidth=1)
            else:
                axis.plot(
                    times, values["target"], "--", color="C1", linewidth=1.4,
                    label="Expected (commanded)",
                )
                axis.plot(
                    times, values["actual"], color="C0", linewidth=1,
                    label="Actual",
                )
                axis.legend(fontsize=8, loc="best")
            axis.set_title(
                f"{finger} — ID {motor_id}: {MOTOR_LABELS[motor_id]}", fontsize=11
            )
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            axis.tick_params(axis="x", labelbottom=True)
            if row == len(motor_ids) - 1:
                axis.set_xlabel("Time (s)")
    figure.suptitle(f"{source.stem} — {title}", fontsize=18)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    destination = source.with_name(f"{source.stem}_{suffix}.png")
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    print(f"Saved {destination}")
PY
