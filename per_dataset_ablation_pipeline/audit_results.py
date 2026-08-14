#!/usr/bin/env python3
"""Audit completed folders and build one compact comparison CSV."""

from __future__ import annotations

import csv
import os
from pathlib import Path


OUTPUT = Path(
    os.environ.get("RESULTS_ROOT") or os.environ["PIPELINE_OUTPUT"]
).expanduser().resolve()
SETUP_META = {
    "steps500_eps2": (500, "2/255"),
    "steps500_eps4": (500, "4/255"),
    "steps800_eps2": (800, "2/255"),
    "steps800_eps4": (800, "4/255"),
}
MODES = ("fixed_0_5", "image_f1", "clean_pixel_f1")
KEEP = (
    "source_dataset",
    "target_dataset",
    "direction",
    "loss_mode",
    "category",
    "clean_i_auroc",
    "adversarial_i_auroc",
    "clean_i_ap",
    "adversarial_i_ap",
    "clean_p_auroc",
    "adversarial_p_auroc",
    "clean_aupro",
    "adversarial_aupro",
    "targeted_attack_success_rate",
    "pixel_decision_threshold",
    "clean_pixel_f1_threshold",
    "target_region_pixel_flip_rate",
    "target_region_pixel_attack_success_rate",
    "mean_actual_linf",
)


def main() -> None:
    combined: list[dict[str, str | int]] = []
    for setup_id, (steps, epsilon) in SETUP_META.items():
        for mode in MODES:
            path = OUTPUT / "setups" / setup_id / "evaluation" / mode / "numerical" / "summary.csv"
            if not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    combined.append(
                        {
                            "setup": setup_id,
                            "steps": steps,
                            "epsilon": epsilon,
                            "pixel_threshold_mode": mode,
                            **{field: row.get(field, "") for field in KEEP},
                        }
                    )
    if not combined:
        print("No completed evaluation summaries yet.")
        return
    destination = OUTPUT / "ablation_high_level_summary.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(f"AUDIT: collected {len(combined)} macro rows")
    print(f"Comparison: {destination}")


if __name__ == "__main__":
    main()
