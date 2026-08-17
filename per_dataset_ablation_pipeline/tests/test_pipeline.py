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

    def test_threshold_sweep_requires_one_output_per_mode(self) -> None:
        modes = ("fixed_0_5", "image_f1", "clean_pixel_f1")
        config = EvaluationConfig(
            artifacts_root="artifacts",
            output_root="unused",
            model_name="model",
            model_kwargs_by_target={},
            pixel_threshold_modes=modes,
            output_roots_by_pixel_threshold_mode={mode: f"out/{mode}" for mode in modes},
        )
        self.assertEqual(config.pixel_threshold_modes, modes)

        with self.assertRaisesRegex(ValueError, "must define every selected mode"):
            EvaluationConfig(
                artifacts_root="artifacts",
                output_root="unused",
                model_name="model",
                model_kwargs_by_target={},
                pixel_threshold_modes=modes,
                output_roots_by_pixel_threshold_mode={"fixed_0_5": "out/fixed"},
            )

    def test_loss_formulation_filter_is_accepted_and_defaults_to_every_family(
        self,
    ) -> None:
        both = EvaluationConfig(
            artifacts_root="artifacts",
            output_root="output",
            model_name="model",
            model_kwargs_by_target={},
        )
        self.assertIsNone(both.attack_loss_formulations)

        margin_only = EvaluationConfig(
            artifacts_root="artifacts",
            output_root="output",
            model_name="model",
            model_kwargs_by_target={},
            attack_loss_formulations=("margin_topk",),
        )
        self.assertEqual(margin_only.attack_loss_formulations, ("margin_topk",))

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
