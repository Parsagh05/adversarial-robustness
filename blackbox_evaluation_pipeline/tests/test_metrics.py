from __future__ import annotations

import unittest

import numpy as np

from blackbox_evaluation_pipeline.universal_eval.metrics import (
    image_metrics,
    pixel_metrics,
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


if __name__ == "__main__":
    unittest.main()
