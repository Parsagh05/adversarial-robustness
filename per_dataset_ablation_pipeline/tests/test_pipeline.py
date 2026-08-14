from __future__ import annotations

import unittest

import numpy as np

from evaluation.universal_eval.metrics import targeted_region_pixel_metrics
from evaluation.universal_eval.qualitative import select_representative_rows
from evaluation.universal_eval.runner import EvaluationConfig


class AblationPipelineTests(unittest.TestCase):
    def test_all_pixel_threshold_modes_are_accepted(self) -> None:
        for mode in ("fixed_0_5", "image_f1", "clean_pixel_f1"):
            config = EvaluationConfig(
                artifacts_root="artifacts",
                output_root="output",
                model_name="model",
                model_kwargs_by_target={},
                pixel_threshold_mode=mode,
            )
            self.assertEqual(config.pixel_threshold_mode, mode)

    def test_region_success_does_not_count_outside_pixels(self) -> None:
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
        self.assertEqual(result["target_region_pixel_flip_rate"], 0.0)
        self.assertEqual(result["target_region_pixel_attack_success"], 0)

    def test_visual_selection_uses_region_pixel_rate(self) -> None:
        rows = [
            {
                "sample_id": "low",
                "target_region_pixel_success_eligible": 1,
                "target_region_pixel_attack_success": 1,
                "target_region_pixel_flip_rate": 60.0,
            },
            {
                "sample_id": "high",
                "target_region_pixel_success_eligible": 1,
                "target_region_pixel_attack_success": 1,
                "target_region_pixel_flip_rate": 90.0,
            },
        ]
        selected = select_representative_rows(
            rows, selection_basis="target_region_pixel"
        )
        self.assertEqual(selected[0][1]["sample_id"], "high")


if __name__ == "__main__":
    unittest.main()

