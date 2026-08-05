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

    def release(self) -> None:
        pass


class ThresholdCalibrationTests(unittest.TestCase):
    def test_mvtc_q95_uses_only_train_good_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "mvtec"
            train_good = dataset / "toy" / "train" / "good"
            test_good = dataset / "toy" / "test" / "good"
            train_good.mkdir(parents=True)
            test_good.mkdir(parents=True)
            for index, value in enumerate((0, 128, 255)):
                Image.fromarray(
                    np.full((4, 4, 3), value, dtype=np.uint8)
                ).save(train_good / f"{index:03d}.png")
            # This score must never enter calibration.
            Image.fromarray(np.full((4, 4, 3), 64, dtype=np.uint8)).save(
                test_good / "999.png"
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
                    quantile=0.95,
                )
            )
            payload = json.loads(generated["mvtec"].read_text(encoding="utf-8"))
            record = payload["categories"]["toy"]
            self.assertEqual(record["sample_count"], 3)
            self.assertEqual(
                record["calibration_sample_ids"],
                [
                    "train/toy/good/000",
                    "train/toy/good/001",
                    "train/toy/good/002",
                ],
            )
            expected = float(np.quantile([0.0, 128 / 255, 1.0], 0.95))
            self.assertAlmostEqual(record["threshold"], expected, places=6)
            self.assertFalse(payload["official_model_threshold"])


if __name__ == "__main__":
    unittest.main()
