from __future__ import annotations

import unittest

from blackbox_evaluation_pipeline.universal_eval.qualitative import (
    select_representative_rows,
)


class QualitativeSelectionTests(unittest.TestCase):
    def test_selects_strongest_median_success_and_worst_failure(self) -> None:
        rows = [
            {
                "sample_id": "success-low",
                "targeted_success_eligible": 1,
                "targeted_attack_success": 1,
                "adversarial_target_margin": 0.1,
            },
            {
                "sample_id": "success-mid",
                "targeted_success_eligible": 1,
                "targeted_attack_success": 1,
                "adversarial_target_margin": 0.4,
            },
            {
                "sample_id": "success-high",
                "targeted_success_eligible": 1,
                "targeted_attack_success": 1,
                "adversarial_target_margin": 0.9,
            },
            {
                "sample_id": "failure-near",
                "targeted_success_eligible": 1,
                "targeted_attack_success": 0,
                "adversarial_target_margin": -0.1,
            },
            {
                "sample_id": "failure-worst",
                "targeted_success_eligible": 1,
                "targeted_attack_success": 0,
                "adversarial_target_margin": -0.8,
            },
            {
                "sample_id": "clean-misclassification",
                "targeted_success_eligible": 0,
                "targeted_attack_success": 0,
                "adversarial_target_margin": -1.0,
            },
        ]
        selected = select_representative_rows(rows)
        self.assertEqual(
            [(name, row["sample_id"]) for name, row in selected],
            [
                ("strongest_success", "success-high"),
                ("median_success", "success-mid"),
                ("worst_failure", "failure-worst"),
            ],
        )

    def test_selects_by_target_region_pixel_success(self) -> None:
        rows = [
            {
                "sample_id": "pixel-low",
                "target_region_pixel_success_eligible": 1,
                "target_region_pixel_attack_success": 1,
                "target_region_pixel_flip_rate": 55.0,
            },
            {
                "sample_id": "pixel-high",
                "target_region_pixel_success_eligible": 1,
                "target_region_pixel_attack_success": 1,
                "target_region_pixel_flip_rate": 90.0,
            },
            {
                "sample_id": "pixel-failure",
                "target_region_pixel_success_eligible": 1,
                "target_region_pixel_attack_success": 0,
                "target_region_pixel_flip_rate": 10.0,
            },
        ]
        selected = select_representative_rows(
            rows, selection_basis="target_region_pixel"
        )
        self.assertEqual(
            [(name, row["sample_id"]) for name, row in selected],
            [
                ("strongest_success", "pixel-high"),
                ("median_success", "pixel-low"),
                ("worst_failure", "pixel-failure"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
