from __future__ import annotations

import unittest

import numpy as np

from blackbox_evaluation_pipeline.universal_eval.adapters.aaclip import AACLIPAdapter


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


if __name__ == "__main__":
    unittest.main()
