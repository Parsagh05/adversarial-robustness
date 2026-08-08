from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from blackbox_evaluation_pipeline.universal_eval.adapters import (
    ModelAdapter,
    register_adapter,
)
from blackbox_evaluation_pipeline.universal_eval.thresholds import (
    ThresholdCalibrationConfig,
    calibrate_thresholds,
)


@register_adapter("threshold_unit_test_adapter")
class ThresholdUnitTestAdapter(ModelAdapter):
    model_name = "threshold-unit-test"

    def __init__(self, **_: object) -> None:
        pass

    def predict(self, images_01: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        scores = images_01.mean(dim=(1, 2, 3)).numpy().astype(np.float32)
        maps = images_01.mean(dim=1).numpy().astype(np.float32)
        return scores, maps

    def predict_with_categories(
        self, images_01: torch.Tensor, categories: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if categories != ["toy"] * len(images_01):
            raise AssertionError(f"Unexpected categories: {categories}")
        return self.predict(images_01)

    def release(self) -> None:
        pass


class ThresholdCalibrationTests(unittest.TestCase):
    def test_mvtc_clean_f1_uses_fixed_evaluation_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "mvtec"
            test_good = dataset / "toy" / "test" / "good"
            test_bad = dataset / "toy" / "test" / "crack"
            mask_bad = dataset / "toy" / "ground_truth" / "crack"
            test_good.mkdir(parents=True)
            test_bad.mkdir(parents=True)
            mask_bad.mkdir(parents=True)
            for index, value in enumerate((0, 51)):
                Image.fromarray(
                    np.full((4, 4, 3), value, dtype=np.uint8)
                ).save(test_good / f"{index:03d}.png")
            for index, value in enumerate((153, 255)):
                Image.fromarray(
                    np.full((4, 4, 3), value, dtype=np.uint8)
                ).save(test_bad / f"{index:03d}.png")
                Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(
                    mask_bad / f"{index:03d}_mask.png"
                )
            evaluation_index = root / "evaluation_test_indices.csv"
            evaluation_index.write_text(
                "protocol_id,dataset,category,label,partition\n"
                "test/toy/good/000,mvtec,toy,0,evaluation\n"
                "test/toy/good/001,mvtec,toy,0,evaluation\n"
                "test/toy/crack/000,mvtec,toy,1,evaluation\n"
                "test/toy/crack/001,mvtec,toy,1,fit\n",
                encoding="utf-8",
            )

            generated = calibrate_thresholds(
                ThresholdCalibrationConfig(
                    output_root=str(root / "output"),
                    model_name="threshold_unit_test_adapter",
                    model_kwargs_by_target={"mvtec": {"unused": True}},
                    datasets=("mvtec",),
                    mvtec_root=str(dataset),
                    device="cpu",
                    batch_size=2,
                    image_size=4,
                    evaluation_index_path=str(evaluation_index),
                )
            )
            payload = json.loads(generated["mvtec"].read_text(encoding="utf-8"))
            record = payload["categories"]["toy"]
            self.assertEqual(record["sample_count"], 3)
            self.assertEqual(record["normal_count"], 2)
            self.assertEqual(record["anomaly_count"], 1)
            self.assertEqual(
                record["calibration_sample_ids"],
                [
                    "test/toy/good/000",
                    "test/toy/good/001",
                    "test/toy/crack/000",
                ],
            )
            self.assertAlmostEqual(record["threshold"], 153 / 255, places=6)
            self.assertAlmostEqual(record["f1_max"], 100.0, places=6)
            self.assertEqual(payload["threshold_mode"], "clean_f1_optimal")
            self.assertTrue(payload["uses_test_labels"])
            self.assertFalse(payload["official_model_threshold"])
            with np.load(
                generated["mvtec"].parent / "clean_evaluation_scores.npz"
            ) as scores:
                np.testing.assert_array_equal(scores["labels"], [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
