from __future__ import annotations

import unittest

import numpy as np

from blackbox_evaluation_pipeline.universal_eval.metrics import (
    binary_classification_metrics,
    image_metrics,
    optimal_f1_operating_point,
    pixel_metrics,
    targeted_attack_metrics,
    targeted_region_pixel_metrics,
)


class ContinuousMetricTests(unittest.TestCase):
    def test_f1_optimal_threshold(self) -> None:
        point = optimal_f1_operating_point(
            [0, 0, 1, 1], [0.1, 0.2, 0.6, 0.9]
        )
        self.assertAlmostEqual(point["threshold"], 0.6)
        self.assertAlmostEqual(point["f1"], 1.0)

    def test_perfect_image_ranking_is_one_hundred(self) -> None:
        result = image_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(result["i_auroc"], 100.0)
        self.assertAlmostEqual(result["i_ap"], 100.0)
        self.assertAlmostEqual(result["i_f1_max"], 100.0)

    def test_perfect_pixel_ranking_is_one_hundred(self) -> None:
        masks = np.asarray([[[0, 0], [1, 1]]], dtype=np.uint8)
        maps = np.asarray([[[0.1, 0.2], [0.8, 0.9]]], dtype=np.float32)
        result = pixel_metrics(masks, maps, fpr_limit=0.3, max_thresholds=20)
        self.assertAlmostEqual(result["p_auroc"], 100.0)
        self.assertAlmostEqual(result["p_f1_max"], 100.0)
        self.assertAlmostEqual(result["p_f1_threshold"], 0.8)
        self.assertGreaterEqual(result["aupro"], 99.0)

    def test_binary_classification_metrics(self) -> None:
        result = binary_classification_metrics([0, 0, 1, 1], [0, 1, 1, 0])
        self.assertEqual(result, {"accuracy": 50.0, "fpr": 50.0, "fnr": 50.0})

    def test_targeted_success_excludes_preexisting_target_predictions(self) -> None:
        result = targeted_attack_metrics(
            [0, 1, 0],
            [1, 1, 0],
            [True, True, False],
            source_label=0,
            target_label=1,
        )
        self.assertEqual(result["attack_flip_rate"], 50.0)
        self.assertEqual(result["targeted_attack_success_rate"], 100.0)
        self.assertEqual(result["targeted_success_eligible_count"], 1)

    def test_region_pixel_success_ignores_changes_outside_target(self) -> None:
        clean = np.zeros((3, 3), dtype=np.float32)
        adversarial = np.ones((3, 3), dtype=np.float32)
        adversarial[1, 1] = 0.0
        region = np.zeros((3, 3), dtype=bool)
        region[1, 1] = True
        result = targeted_region_pixel_metrics(
            clean,
            adversarial,
            region,
            threshold=0.5,
            source_label=0,
            target_label=1,
        )
        self.assertEqual(result["target_region_pixel_count"], 1)
        self.assertEqual(result["target_region_pixel_eligible_count"], 1)
        self.assertEqual(result["target_region_pixel_flip_rate"], 0.0)
        self.assertEqual(result["target_region_pixel_attack_success"], 0)

    def test_region_pixel_success_requires_half_of_eligible_region(self) -> None:
        clean = np.ones((2, 2), dtype=np.float32)
        adversarial = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        result = targeted_region_pixel_metrics(
            clean,
            adversarial,
            np.ones((2, 2), dtype=bool),
            threshold=0.5,
            source_label=1,
            target_label=0,
            minimum_flip_fraction=0.5,
        )
        self.assertEqual(result["target_region_pixel_flip_rate"], 50.0)
        self.assertEqual(result["target_region_pixel_attack_success"], 1)


if __name__ == "__main__":
    unittest.main()
