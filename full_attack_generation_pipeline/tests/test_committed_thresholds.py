from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_ROOT = ROOT / "thresholds"
SPLIT_ROOT = ROOT / "splits"
MODES = ("mvtec", "visa", "both")
EXPECTED_CHECKPOINT_DIR = {
    "mvtec": "9_12_4_multiscale",
    "visa": "9_12_4_multiscale_visa",
    "both": "9_12_4_multiscale",
}


class CommittedThresholdArtifactTests(unittest.TestCase):
    def _split_categories(self, mode: str) -> set[str]:
        path = SPLIT_ROOT / f"{mode}_matched_test_per_category_v1_seed111.json"
        return set(json.loads(path.read_text(encoding="utf-8"))["categories"])

    def test_metadata_scores_categories_and_quantiles(self) -> None:
        for mode in MODES:
            with self.subTest(mode=mode):
                folder = THRESHOLD_ROOT / mode
                threshold_path = folder / "category_thresholds.json"
                score_path = folder / "normal_train_scores.npz"
                config_path = folder / "threshold_config.json"
                self.assertTrue(threshold_path.is_file(), threshold_path)
                self.assertTrue(score_path.is_file(), score_path)
                self.assertTrue(config_path.is_file(), config_path)

                payload = json.loads(threshold_path.read_text(encoding="utf-8"))
                config = json.loads(config_path.read_text(encoding="utf-8"))
                records = payload["categories"]
                self.assertEqual(payload["dataset"], mode)
                self.assertEqual(config["dataset"], mode)
                self.assertEqual(payload["threshold_mode"], "normal_train_quantile")
                self.assertEqual(payload["threshold_quantile"], 0.95)
                self.assertEqual(payload["image_size"], 518)
                self.assertEqual(set(records), self._split_categories(mode))
                self.assertEqual(
                    Path(config["anomalyclip_checkpoint"]).parent.name,
                    EXPECTED_CHECKPOINT_DIR[mode],
                )

                with np.load(score_path, allow_pickle=False) as scores:
                    self.assertEqual(
                        set(scores.files),
                        {"protocol_ids", "datasets", "categories", "scores"},
                    )
                    values = scores["scores"]
                    categories = scores["categories"]
                    datasets = set(scores["datasets"].tolist())
                    self.assertTrue(np.isfinite(values).all())
                    self.assertEqual(len(values), len(set(scores["protocol_ids"].tolist())))
                    self.assertEqual(set(categories.tolist()), set(records))
                    self.assertEqual(
                        datasets,
                        {"mvtec", "visa"} if mode == "both" else {mode},
                    )
                    for category, record in records.items():
                        category_scores = values[categories == category]
                        self.assertEqual(len(category_scores), record["sample_count"])
                        self.assertTrue(
                            np.isclose(
                                np.quantile(category_scores, 0.95),
                                record["threshold"],
                            ),
                            category,
                        )

    def test_both_reuses_mvtec_calibration_for_shared_checkpoint(self) -> None:
        mvtec = json.loads(
            (THRESHOLD_ROOT / "mvtec" / "category_thresholds.json").read_text(
                encoding="utf-8"
            )
        )["categories"]
        both = json.loads(
            (THRESHOLD_ROOT / "both" / "category_thresholds.json").read_text(
                encoding="utf-8"
            )
        )["categories"]
        for category, record in mvtec.items():
            combined = both[category]
            self.assertEqual(
                combined["calibration_sample_ids"], record["calibration_sample_ids"]
            )
            self.assertEqual(combined["sample_count"], record["sample_count"])
            for field in (
                "threshold",
                "score_min",
                "score_max",
                "score_mean",
                "score_std",
            ):
                self.assertTrue(
                    np.isclose(combined[field], record[field], rtol=1e-6, atol=1e-7),
                    f"{category}/{field}",
                )


if __name__ == "__main__":
    unittest.main()
