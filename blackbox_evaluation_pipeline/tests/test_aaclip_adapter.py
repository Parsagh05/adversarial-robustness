from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from blackbox_evaluation_pipeline.universal_eval.adapters.aaclip import AACLIPAdapter
from blackbox_evaluation_pipeline.universal_eval.datasets import EvaluationSample
from blackbox_evaluation_pipeline.universal_eval.runner import (
    _postprocess_prediction_scores,
)


class AACLIPScoreTests(unittest.TestCase):
    def test_official_image_score_aggregation_is_per_category(self) -> None:
        adapter = AACLIPAdapter.__new__(AACLIPAdapter)
        result = adapter.postprocess_image_scores(
            scores=np.asarray([2.0, 4.0, 10.0, 20.0]),
            map_mins=np.asarray([1.0, 2.0, 100.0, 110.0]),
            map_maxs=np.asarray([3.0, 5.0, 120.0, 140.0]),
            categories=["a", "a", "b", "b"],
        )
        expected = np.asarray(
            [
                0.5 * ((3.0 - 1.0) / (5.0 - 1.0)) + 0.5 * 0.0,
                0.5 * 1.0 + 0.5 * 1.0,
                0.5 * ((120.0 - 100.0) / (140.0 - 100.0)) + 0.5 * 0.0,
                0.5 * 1.0 + 0.5 * 1.0,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result, expected)

    def test_adversarial_scores_use_frozen_clean_normalization(self) -> None:
        adapter = AACLIPAdapter.__new__(AACLIPAdapter)
        clean_scores = np.asarray([2.0, 4.0])
        clean_map_mins = np.asarray([1.0, 2.0])
        clean_map_maxs = np.asarray([3.0, 5.0])
        categories = ["a", "a"]
        clean = adapter.postprocess_image_scores(
            clean_scores, clean_map_mins, clean_map_maxs, categories
        )

        # The first image is untouched, while the second adversarial image
        # changes the cohort extrema dramatically.
        adversarial = adapter.postprocess_image_scores_with_reference(
            scores=np.asarray([2.0, 100.0]),
            map_mins=np.asarray([1.0, 2.0]),
            map_maxs=np.asarray([3.0, 200.0]),
            categories=categories,
            reference_scores=clean_scores,
            reference_map_mins=clean_map_mins,
            reference_map_maxs=clean_map_maxs,
            reference_categories=categories,
        )
        independently_normalized = adapter.postprocess_image_scores(
            np.asarray([2.0, 100.0]),
            np.asarray([1.0, 2.0]),
            np.asarray([3.0, 200.0]),
            categories,
        )

        self.assertEqual(adversarial[0], clean[0])
        self.assertNotEqual(independently_normalized[0], clean[0])

    def test_runner_passes_raw_clean_reference_to_adversarial_scoring(self) -> None:
        adapter = AACLIPAdapter.__new__(AACLIPAdapter)
        samples = [
            EvaluationSample(
                dataset="mvtec",
                category="a",
                defect_type="good",
                image_path=Path(f"{index:03d}.png"),
                mask_path=None,
                label=0,
            )
            for index in range(2)
        ]
        first, second = (sample.protocol_id for sample in samples)
        raw_clean = {
            first: (2.0, np.asarray([[1.0, 3.0]], dtype=np.float32)),
            second: (4.0, np.asarray([[2.0, 5.0]], dtype=np.float32)),
        }
        raw_adversarial = {
            first: raw_clean[first],
            second: (100.0, np.asarray([[2.0, 200.0]], dtype=np.float32)),
        }

        clean = _postprocess_prediction_scores(adapter, samples, raw_clean)
        adversarial = _postprocess_prediction_scores(
            adapter,
            samples,
            raw_adversarial,
            reference_samples=samples,
            reference_predictions=raw_clean,
        )

        self.assertEqual(adversarial[first][0], clean[first][0])


if __name__ == "__main__":
    unittest.main()
