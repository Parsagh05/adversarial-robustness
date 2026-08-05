from __future__ import annotations

import csv
import hashlib
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
from blackbox_evaluation_pipeline.universal_eval.runner import (
    EvaluationConfig,
    run_evaluation,
)


@register_adapter("unit_test_adapter")
class UnitTestAdapter(ModelAdapter):
    model_name = "unit-test"

    def __init__(self, **_: object) -> None:
        pass

    def predict(self, images_01: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        scores = images_01.mean(dim=(1, 2, 3)).numpy().astype(np.float32)
        maps = images_01.mean(dim=1).numpy().astype(np.float32)
        return scores, maps

    def release(self) -> None:
        pass


class EndToEndRunnerTests(unittest.TestCase):
    def test_manifest_ids_drive_attack_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "mvtec"
            good = dataset / "toy" / "test" / "good" / "000.png"
            bad = dataset / "toy" / "test" / "crack" / "001.png"
            mask = dataset / "toy" / "ground_truth" / "crack" / "001_mask.png"
            for path in (good, bad, mask):
                path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(good)
            Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8)).save(bad)
            mask_array = np.zeros((4, 4), dtype=np.uint8)
            mask_array[1:3, 1:3] = 255
            Image.fromarray(mask_array).save(mask)

            artifacts = root / "canonical"
            delta_path = artifacts / "visa_to_mvtec" / "all_categories.pt"
            delta_path.parent.mkdir(parents=True)
            torch.save({"delta": torch.full((1, 3, 4, 4), 0.02)}, delta_path)
            checksum = hashlib.sha256(delta_path.read_bytes()).hexdigest()
            record = {
                "source_dataset": "visa",
                "target_dataset": "mvtec",
                "direction": "normal_to_abnormal",
                "loss_mode": "global",
                "scope": "dataset",
                "source_label": 0,
                "target_label": 1,
                "target_evaluation_all_count": 2,
                "target_attacked_label_count": 1,
                "target_evaluation_all_sample_ids": [
                    "test/toy/good/000",
                    "test/toy/crack/001",
                ],
                "target_attacked_sample_ids": ["test/toy/good/000"],
                "artifact_path": "/kaggle/working/canonical/visa_to_mvtec/all_categories.pt",
                "artifact_file_sha256": checksum,
                "epsilon": 8 / 255,
                "image_size": 4,
            }
            (artifacts / "all_canonical_attack_artifacts.json").write_text(
                json.dumps([record]), encoding="utf-8"
            )
            output = root / "output"
            summary = run_evaluation(
                EvaluationConfig(
                    artifacts_root=str(artifacts),
                    output_root=str(output),
                    model_name="unit_test_adapter",
                    model_kwargs_by_target={"mvtec": {"unused": True}},
                    mvtec_root=str(dataset),
                    device="cpu",
                    batch_size=1,
                    metric_size=4,
                    anomaly_map_sigma=0,
                    aupro_max_thresholds=10,
                )
            )
            self.assertTrue(summary.is_file())
            with summary.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_count"], "2")
            self.assertEqual(rows[0]["attacked_count"], "1")
            with (output / "per_image.csv").open(newline="", encoding="utf-8") as handle:
                per_image = {row["sample_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(per_image["test/toy/good/000"]["attacked"], "1")
            self.assertEqual(per_image["test/toy/crack/001"]["attacked"], "0")
            self.assertGreater(
                float(per_image["test/toy/good/000"]["directional_score_shift"]),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
