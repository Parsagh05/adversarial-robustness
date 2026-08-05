from __future__ import annotations

import unittest

import numpy as np

from blackbox_evaluation_pipeline.universal_eval.metrics import (
    binary_classification_metrics,
    image_metrics,
    pixel_metrics,
    targeted_attack_metrics,
)


class ContinuousMetricTests(unittest.TestCase):
    def test_perfect_image_ranking_is_one_hundred(self) -> None:
        result = image_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(result["i_auroc"], 100.0)
        self.assertAlmostEqual(result["i_ap"], 100.0)

    def test_perfect_pixel_ranking_is_one_hundred(self) -> None:
        masks = np.asarray([[[0, 0], [1, 1]]], dtype=np.uint8)
        maps = np.asarray([[[0.1, 0.2], [0.8, 0.9]]], dtype=np.float32)
        result = pixel_metrics(masks, maps, fpr_limit=0.3, max_thresholds=20)
        self.assertAlmostEqual(result["p_auroc"], 100.0)
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


if __name__ == "__main__":
    unittest.main()
