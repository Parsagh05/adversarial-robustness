from __future__ import annotations

import unittest

import numpy as np

from adversarial_harness.metrics import (
    per_image_classification_metrics,
    per_image_map_metrics,
)


class PerImageMapMetricsTests(unittest.TestCase):
    def test_normal_false_alarm_map_success_has_no_localization_metric(self) -> None:
        clean = np.zeros((3, 3), dtype=np.float32)
        adversarial = np.full((3, 3), 0.25, dtype=np.float32)
        result = per_image_map_metrics(
            np.zeros((3, 3), dtype=np.uint8),
            clean,
            adversarial,
            target_label=1,
            map_success_min_mean_shift=0.0,
            map_false_positive_threshold=0.1,
        )

        self.assertEqual(result["map_directional_success"], 1.0)
        self.assertAlmostEqual(result["map_directional_mean_shift"], 0.25)
        self.assertAlmostEqual(result["map_directional_pixel_fraction"], 1.0)
        self.assertAlmostEqual(result["map_absolute_shift"], 0.25)
        self.assertAlmostEqual(result["clean_false_positive_map_area"], 0.0)
        self.assertAlmostEqual(result["adversarial_false_positive_map_area"], 1.0)
        self.assertAlmostEqual(result["false_positive_map_area_increase"], 1.0)
        self.assertTrue(np.isnan(result["image_p_ap_drop"]))
        self.assertTrue(np.isnan(result["localization_degradation_success"]))

    def test_uniform_suppression_changes_map_but_preserves_localization(self) -> None:
        mask = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
        clean = np.asarray([[0.9, 0.8], [0.2, 0.1]], dtype=np.float32)
        adversarial = clean - 0.2
        result = per_image_map_metrics(
            mask,
            clean,
            adversarial,
            target_label=0,
            map_success_min_mean_shift=0.0,
        )

        self.assertEqual(result["map_directional_success"], 1.0)
        self.assertAlmostEqual(result["map_directional_mean_shift"], 0.2, places=6)
        self.assertAlmostEqual(result["image_p_auroc_drop"], 0.0)
        self.assertAlmostEqual(result["image_p_ap_drop"], 0.0)
        self.assertAlmostEqual(result["image_aupro_drop"], 0.0)
        self.assertAlmostEqual(result["localization_contrast_drop"], 0.0, places=6)
        self.assertEqual(result["localization_degradation_success"], 0.0)

    def test_inverted_map_is_localization_degradation(self) -> None:
        mask = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
        clean = np.asarray([[0.9, 0.8], [0.2, 0.1]], dtype=np.float32)
        adversarial = np.asarray([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32)
        result = per_image_map_metrics(
            mask,
            clean,
            adversarial,
            target_label=0,
            map_success_min_mean_shift=0.0,
        )

        self.assertGreater(result["image_p_auroc_drop"], 0.0)
        self.assertGreater(result["image_p_ap_drop"], 0.0)
        self.assertGreater(result["image_aupro_drop"], 0.0)
        self.assertGreater(result["localization_contrast_drop"], 0.0)
        self.assertEqual(result["localization_degradation_success"], 1.0)

    def test_exactly_half_directional_pixels_is_not_success(self) -> None:
        clean = np.zeros((2, 2), dtype=np.float32)
        adversarial = np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)

        result = per_image_map_metrics(
            np.zeros((2, 2), dtype=np.uint8),
            clean,
            adversarial,
            target_label=1,
            map_success_min_mean_shift=0.0,
        )

        self.assertAlmostEqual(result["map_directional_pixel_fraction"], 0.5)
        self.assertEqual(result["map_directional_success"], 0.0)

    def test_map_change_without_classification_flip(self) -> None:
        classification = per_image_classification_metrics(
            clean_score=0.1,
            adversarial_score=0.2,
            decision_threshold=0.5,
            source_label=0,
            target_label=1,
        )
        maps = per_image_map_metrics(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            target_label=1,
            map_success_min_mean_shift=0.0,
        )

        self.assertEqual(classification["classification_flip"], 0.0)
        self.assertEqual(classification["image_targeted_success"], 0.0)
        self.assertEqual(maps["map_directional_success"], 1.0)

    def test_classification_flip_without_localization_degradation(self) -> None:
        classification = per_image_classification_metrics(
            clean_score=0.8,
            adversarial_score=0.2,
            decision_threshold=0.5,
            source_label=1,
            target_label=0,
        )
        mask = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
        clean = np.asarray([[0.9, 0.8], [0.2, 0.1]], dtype=np.float32)
        maps = per_image_map_metrics(
            mask,
            clean,
            clean - 0.2,
            target_label=0,
            map_success_min_mean_shift=0.0,
        )

        self.assertEqual(classification["classification_flip"], 1.0)
        self.assertEqual(classification["image_targeted_success"], 1.0)
        self.assertEqual(maps["map_directional_success"], 1.0)
        self.assertEqual(maps["localization_degradation_success"], 0.0)


if __name__ == "__main__":
    unittest.main()
