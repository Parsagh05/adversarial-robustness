"""Create a detailed, cohort-checked comparison of legacy and v2 attacks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OLD_ROOT = ROOT / "results" / "anomalyclip" / "anomalyclip_per_dataset"
V2_ROOT = ROOT / "results" / "v2_results" / "anomalyclip_v2_mvtec_results"
OUTPUT = ROOT / "results" / "anomalyclip_old_vs_v2_detailed_comparison.csv"
HIGH_LEVEL_OUTPUT = ROOT / "results" / "anomalyclip_old_vs_v2_high_level_summary.csv"
HIGH_LEVEL_MARKDOWN = ROOT / "results" / "ANOMALYCLIP_OLD_VS_V2_SUMMARY.md"
FOUR_METRICS_OUTPUT = ROOT / "results" / "anomalyclip_old_vs_v2_four_metrics_comparison.csv"

IDENTITY = [
    "source_dataset",
    "target_dataset",
    "direction",
    "loss_mode",
    "scope",
    "category",
]
NON_METRICS = {"model", "condition", *IDENTITY}
ATTACK_EFFECT_METRICS = [
    "mean_directional_score_shift",
    "mean_directional_map_shift",
    "mean_directional_map_pixel_fraction",
    "delta_i_auroc",
    "delta_i_ap",
    "delta_i_f1_max",
    "delta_p_auroc",
    "delta_p_f1_max",
    "delta_aupro",
    "attack_flip_rate",
    "targeted_attack_success_rate",
]


def load_manifest(path: Path) -> dict[tuple[str, str, str, str], dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (
            str(row["source_dataset"]),
            str(row["target_dataset"]),
            str(row["direction"]),
            str(row["loss_mode"]),
        ): row
        for row in rows
    }


def mvtec_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[
        frame["source_dataset"].eq("mvtec")
        & frame["target_dataset"].eq("mvtec")
        & frame["scope"].eq("per_dataset")
    ].copy()


def winner(old: float, v2: float, tolerance: float = 1e-9) -> str:
    if pd.isna(old) or pd.isna(v2):
        return "not_available"
    difference = float(v2) - float(old)
    if abs(difference) <= tolerance:
        return "tie"
    return "v2" if difference > 0 else "old"


def compare_level(
    old_path: Path,
    v2_path: Path,
    aggregation_level: str,
    old_manifest: dict,
    v2_manifest: dict,
) -> pd.DataFrame:
    old = mvtec_rows(old_path)
    v2 = mvtec_rows(v2_path)
    if aggregation_level == "macro":
        old["category"] = "macro"
        v2["category"] = "macro"

    merged = old.merge(v2, on=IDENTITY, suffixes=("_old", "_v2"), validate="one_to_one")
    if len(merged) != len(v2) or len(old) != len(v2):
        raise RuntimeError(
            f"Unmatched {aggregation_level} rows: old={len(old)}, v2={len(v2)}, "
            f"matched={len(merged)}"
        )

    old_numeric = {
        column for column in old.columns
        if column not in NON_METRICS and pd.api.types.is_numeric_dtype(old[column])
    }
    v2_numeric = {
        column for column in v2.columns
        if column not in NON_METRICS and pd.api.types.is_numeric_dtype(v2[column])
    }
    numeric = [column for column in old.columns if column in old_numeric & v2_numeric]

    records = []
    for _, row in merged.iterrows():
        condition_key = (
            row["source_dataset"],
            row["target_dataset"],
            row["direction"],
            row["loss_mode"],
        )
        old_meta = old_manifest[condition_key]
        v2_meta = v2_manifest[condition_key]
        old_ids = old_meta.get("target_evaluation_all_sample_ids", [])
        v2_ids = v2_meta.get("target_evaluation_all_sample_ids", [])
        old_attacked_ids = old_meta.get("target_attacked_sample_ids", [])
        v2_attacked_ids = v2_meta.get("target_attacked_sample_ids", [])
        record = {
            "aggregation_level": aggregation_level,
            **{column: row[column] for column in IDENTITY},
            "comparison_is_same_cohort": old_ids == v2_ids,
            "comparison_has_same_attacked_ids": old_attacked_ids == v2_attacked_ids,
            "old_protocol_split_sha256": old_meta.get("protocol_split_sha256", ""),
            "v2_protocol_split_sha256": v2_meta.get("protocol_split_sha256", ""),
            "old_artifact_sha256": old_meta.get("artifact_sha256", ""),
            "v2_artifact_sha256": v2_meta.get("artifact_sha256", ""),
            "old_optimization_steps": old_meta.get("optimization_steps", ""),
            "v2_optimization_steps": v2_meta.get("optimization_steps", ""),
            "old_step_size": old_meta.get("step_size", ""),
            "v2_step_size": v2_meta.get("step_size", ""),
            "old_local_objective": old_meta.get("local_objective", "not_recorded_legacy"),
            "v2_local_objective": v2_meta.get("local_objective", ""),
            "old_normal_local_target": old_meta.get("normal_local_target", "not_recorded_legacy"),
            "v2_normal_local_target": v2_meta.get("normal_local_target", ""),
        }
        for metric in numeric:
            old_value = row[f"{metric}_old"]
            v2_value = row[f"{metric}_v2"]
            record[f"{metric}_old"] = old_value
            record[f"{metric}_v2"] = v2_value
            record[f"{metric}_v2_minus_old"] = v2_value - old_value

        clean_metrics = [
            metric for metric in numeric
            if metric.startswith("clean_") or metric in {"sample_count", "attacked_count"}
        ]
        record["clean_and_count_values_match"] = all(
            pd.isna(row[f"{metric}_old"]) and pd.isna(row[f"{metric}_v2"])
            or np.isclose(row[f"{metric}_old"], row[f"{metric}_v2"], equal_nan=True)
            for metric in clean_metrics
        )
        wins = [
            winner(row[f"{metric}_old"], row[f"{metric}_v2"])
            for metric in ATTACK_EFFECT_METRICS
            if metric in numeric
        ]
        record["attack_effect_metrics_v2_wins"] = wins.count("v2")
        record["attack_effect_metrics_old_wins"] = wins.count("old")
        record["attack_effect_metrics_ties"] = wins.count("tie")
        record["attack_effect_metrics_unavailable"] = wins.count("not_available")
        if wins.count("v2") > wins.count("old"):
            record["majority_attack_effect_winner"] = "v2"
        elif wins.count("old") > wins.count("v2"):
            record["majority_attack_effect_winner"] = "old"
        else:
            record["majority_attack_effect_winner"] = "tie"
        record["primary_metric"] = "targeted_attack_success_rate"
        record["primary_metric_winner"] = winner(
            row["targeted_attack_success_rate_old"],
            row["targeted_attack_success_rate_v2"],
        )
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    old_manifest = load_manifest(OLD_ROOT / "manifest_snapshot.json")
    v2_manifest = load_manifest(V2_ROOT / "manifest_snapshot.json")
    macro = compare_level(
        OLD_ROOT / "summary.csv",
        V2_ROOT / "summary.csv",
        "macro",
        old_manifest,
        v2_manifest,
    )
    categories = compare_level(
        OLD_ROOT / "category_metrics.csv",
        V2_ROOT / "category_metrics.csv",
        "category",
        old_manifest,
        v2_manifest,
    )
    comparison = pd.concat([macro, categories], ignore_index=True, sort=False)
    if not comparison["comparison_is_same_cohort"].all():
        raise RuntimeError("Old and v2 evaluation cohorts differ")
    if not comparison["comparison_has_same_attacked_ids"].all():
        raise RuntimeError("Old and v2 attacked-image cohorts differ")
    comparison.to_csv(OUTPUT, index=False)
    high_level_columns = [
        "scope",
        "source_dataset",
        "target_dataset",
        "direction",
        "loss_mode",
        "targeted_attack_success_rate_old",
        "targeted_attack_success_rate_v2",
        "targeted_attack_success_rate_v2_minus_old",
        "attack_flip_rate_old",
        "attack_flip_rate_v2",
        "attack_flip_rate_v2_minus_old",
        "clean_i_auroc_old",
        "adversarial_i_auroc_old",
        "adversarial_i_auroc_v2",
        "adversarial_i_auroc_v2_minus_old",
        "delta_i_auroc_old",
        "delta_i_auroc_v2",
        "delta_i_auroc_v2_minus_old",
        "clean_i_ap_old",
        "adversarial_i_ap_old",
        "adversarial_i_ap_v2",
        "adversarial_i_ap_v2_minus_old",
        "delta_i_ap_old",
        "delta_i_ap_v2",
        "delta_i_ap_v2_minus_old",
        "clean_p_auroc_old",
        "adversarial_p_auroc_old",
        "adversarial_p_auroc_v2",
        "adversarial_p_auroc_v2_minus_old",
        "delta_p_auroc_old",
        "delta_p_auroc_v2",
        "delta_p_auroc_v2_minus_old",
        "clean_aupro_old",
        "adversarial_aupro_old",
        "adversarial_aupro_v2",
        "adversarial_aupro_v2_minus_old",
        "delta_aupro_old",
        "delta_aupro_v2",
        "delta_aupro_v2_minus_old",
        "primary_metric_winner",
        "majority_attack_effect_winner",
    ]
    high_level = macro[high_level_columns].copy()
    numeric_columns = high_level.select_dtypes(include=[np.number]).columns
    high_level[numeric_columns] = high_level[numeric_columns].round(4)
    high_level.to_csv(HIGH_LEVEL_OUTPUT, index=False)

    four_metrics_columns = [
        "scope",
        "source_dataset",
        "target_dataset",
        "direction",
        "loss_mode",
    ]
    for metric in ("i_auroc", "i_ap", "p_auroc", "aupro"):
        four_metrics_columns.extend(
            [
                f"clean_{metric}_old",
                f"adversarial_{metric}_old",
                f"adversarial_{metric}_v2",
                f"adversarial_{metric}_v2_minus_old",
                f"delta_{metric}_old",
                f"delta_{metric}_v2",
                f"delta_{metric}_v2_minus_old",
            ]
        )
    four_metrics = macro[four_metrics_columns].copy()
    four_metrics = four_metrics.rename(
        columns={
            "clean_i_auroc_old": "shared_clean_i_auroc",
            "clean_i_ap_old": "shared_clean_image_ap",
            "clean_p_auroc_old": "shared_clean_p_auroc",
            "clean_aupro_old": "shared_clean_aupro",
        }
    )
    four_numeric = four_metrics.select_dtypes(include=[np.number]).columns
    four_metrics[four_numeric] = four_metrics[four_numeric].round(4)
    four_metrics.to_csv(FOUR_METRICS_OUTPUT, index=False)

    headings = [
        "Direction",
        "Loss",
        "Success old",
        "Success v2",
        "Change",
        "Image-AUROC drop old",
        "Image-AUROC drop v2",
        "Pixel-AUROC drop old",
        "Pixel-AUROC drop v2",
        "AUPRO drop old",
        "AUPRO drop v2",
        "Overall",
    ]
    lines = [
        "# AnomalyCLIP old versus v2 attack comparison",
        "",
        "Both runs use exactly the same 861 MVTec evaluation images and attacked-image IDs. "
        "Higher targeted success, flip rate, and metric drop indicate a stronger attack.",
        "",
        "| " + " | ".join(headings) + " |",
        "|" + "|".join(["---"] * 2 + ["---:"] * 9 + ["---"]) + "|",
    ]
    for _, row in high_level.iterrows():
        values = [
            str(row["direction"]).replace("_", " "),
            str(row["loss_mode"]),
            f'{row["targeted_attack_success_rate_old"]:.2f}%',
            f'{row["targeted_attack_success_rate_v2"]:.2f}%',
            f'{row["targeted_attack_success_rate_v2_minus_old"]:+.2f} pp',
            f'{row["delta_i_auroc_old"]:.2f}',
            f'{row["delta_i_auroc_v2"]:.2f}',
            f'{row["delta_p_auroc_old"]:.2f}',
            f'{row["delta_p_auroc_v2"]:.2f}',
            f'{row["delta_aupro_old"]:.2f}',
            f'{row["delta_aupro_v2"]:.2f}',
            str(row["majority_attack_effect_winner"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Brief conclusion",
            "",
            "V2 is stronger across the majority of attack-effect metrics in five of six "
            "conditions. The largest improvements are abnormal-to-normal global and combined. "
            "Normal-to-abnormal global remains saturated at 100% targeted success, while v2 "
            "improves the combined and local targeted-success rates. The exception is "
            "abnormal-to-normal local, where v2 is clearly weaker than the legacy attack.",
            "",
        ]
    )
    HIGH_LEVEL_MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(comparison)} rows x {len(comparison.columns)} columns: {OUTPUT}")
    print(f"Wrote {len(high_level)} high-level rows: {HIGH_LEVEL_OUTPUT}")
    print(f"Wrote four-metric comparison: {FOUR_METRICS_OUTPUT}")
    print(f"Wrote high-level report: {HIGH_LEVEL_MARKDOWN}")


if __name__ == "__main__":
    main()
