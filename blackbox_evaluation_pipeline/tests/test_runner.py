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
    _predict_adversarial,
    run_evaluation,
)
from blackbox_evaluation_pipeline.universal_eval.datasets import EvaluationSample


@register_adapter("unit_test_adapter")
class UnitTestAdapter(ModelAdapter):
    model_name = "unit-test"

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


class EndToEndRunnerTests(unittest.TestCase):
    def test_per_image_deltas_are_matched_to_protocol_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = []
            for index in range(3):
                image_path = root / f"{index:03d}.png"
                Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image_path)
                samples.append(
                    EvaluationSample(
                        dataset="mvtec",
                        category="toy",
                        defect_type="good",
                        image_path=image_path,
                        mask_path=None,
                        label=0,
                    )
                )
            attacked = {samples[0].protocol_id, samples[2].protocol_id}
            delta = torch.stack(
                [torch.full((3, 4, 4), 0.01), torch.full((3, 4, 4), 0.03)]
            )
            predictions, linf = _predict_adversarial(
                UnitTestAdapter(),
                samples,
                delta,
                attacked,
                {samples[0].protocol_id: 0, samples[2].protocol_id: 1},
                image_size=4,
                batch_size=3,
                description="per-image-test",
            )

            self.assertAlmostEqual(predictions[samples[0].protocol_id][0], 0.01)
            self.assertAlmostEqual(predictions[samples[1].protocol_id][0], 0.0)
            self.assertAlmostEqual(predictions[samples[2].protocol_id][0], 0.03)
            self.assertAlmostEqual(linf[samples[0].protocol_id], 0.01)
            self.assertAlmostEqual(linf[samples[2].protocol_id], 0.03)

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

            artifacts = root / "adversarial-attacks-vlm-survey"
            bundle = artifacts / "canonical_clip_per_dataset"
            delta_path = (
                bundle / "noises" / "mvtec" / "f1" / "perturbations"
                / "dataset__normal_to_abnormal__global.pt"
            )
            delta_path.parent.mkdir(parents=True)
            torch.save({"delta": torch.full((1, 3, 4, 4), 0.02)}, delta_path)
            checksum = hashlib.sha256(delta_path.read_bytes()).hexdigest()
            (bundle / "attack_manifest.csv").write_text(
                "scope,source_dataset,target_dataset,category,direction,source_label,"
                "target_label,loss_mode,evaluation_attacked_image_count,noise_file,"
                "noise_tensor_key,artifact_sha256,image_size,epsilon\n"
                + "dataset,mvtec,mvtec,,normal_to_abnormal,0,1,global,1,"
                + "noises/mvtec/f1/perturbations/dataset__normal_to_abnormal__global.pt,"
                + f"delta,{checksum},4,{8 / 255}\n",
                encoding="utf-8",
            )
            (bundle / "evaluation_test_indices.csv").write_text(
                "protocol_id,dataset,category,label,partition\n"
                "test/toy/good/000,mvtec,toy,0,evaluation\n"
                "test/toy/crack/001,mvtec,toy,1,evaluation\n",
                encoding="utf-8",
            )
            threshold_path = root / "thresholds" / "category_thresholds.json"
            threshold_path.parent.mkdir(parents=True)
            threshold_path.write_text(
                json.dumps(
                    {
                        "target_model": "unit-test-adapter",
                        "dataset": "mvtec",
                        "threshold_mode": "normal_train_quantile",
                        "categories": {"toy": {"threshold": 0.01}},
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            qualitative_output = root / "qualitative"
            summary = run_evaluation(
                EvaluationConfig(
                    artifacts_root=str(artifacts),
                    output_root=str(output),
                    model_name="unit_test_adapter",
                    model_kwargs_by_target={"mvtec": {"unused": True}},
                    thresholds_by_target={"mvtec": str(threshold_path)},
                    mvtec_root=str(dataset),
                    device="cpu",
                    batch_size=1,
                    metric_size=4,
                    anomaly_map_sigma=0,
                    aupro_max_thresholds=10,
                    save_qualitative_samples=True,
                    qualitative_output_root=str(qualitative_output),
                )
            )
            self.assertTrue(summary.is_file())
            with summary.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_count"], "2")
            self.assertEqual(rows[0]["attacked_count"], "1")
            self.assertEqual(float(rows[0]["clean_accuracy"]), 100.0)
            self.assertEqual(float(rows[0]["adversarial_accuracy"]), 50.0)
            self.assertEqual(float(rows[0]["attack_flip_rate"]), 100.0)
            self.assertEqual(float(rows[0]["targeted_attack_success_rate"]), 100.0)
            with (output / "per_image.csv").open(newline="", encoding="utf-8") as handle:
                per_image = {row["sample_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(per_image["test/toy/good/000"]["attacked"], "1")
            self.assertEqual(per_image["test/toy/crack/001"]["attacked"], "0")
            self.assertEqual(
                per_image["test/toy/good/000"]["clean_binary_prediction"], "0"
            )
            self.assertEqual(
                per_image["test/toy/good/000"]["adversarial_binary_prediction"],
                "1",
            )
            self.assertEqual(per_image["test/toy/good/000"]["attack_flipped"], "1")
            self.assertGreater(
                float(per_image["test/toy/good/000"]["directional_score_shift"]),
                0.0,
            )
            prediction_path = output / "predictions" / (
                "mvtec__mvtec__normal_to_abnormal__global__per_dataset.npz"
            )
            with np.load(prediction_path, allow_pickle=False) as predictions:
                self.assertEqual(
                    predictions["clean_binary_predictions"].tolist(), [0, 1]
                )
                self.assertEqual(
                    predictions["adversarial_binary_predictions"].tolist(), [1, 1]
                )
            sample_folder = qualitative_output / (
                "mvtec__mvtec__normal_to_abnormal__global__per_dataset"
            ) / "strongest_success__test__toy__good__000"
            self.assertEqual(
                {path.name for path in sample_folder.iterdir()},
                {
                    "clean.png",
                    "adversarial.png",
                    "difference_x10.png",
                    "clean_heatmap.png",
                    "adversarial_heatmap.png",
                    "clean_overlay.png",
                    "adversarial_overlay.png",
                    "ground_truth_mask.png",
                    "heatmap_difference.png",
                    "metrics.json",
                },
            )


if __name__ == "__main__":
    unittest.main()
