from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from adversarial_harness.config import AttackConfig, ExperimentConfig
from adversarial_harness.dataset import (
    MVTecSample,
    discover_anomaly_datasets,
    discover_visa,
    discover_visa_train_normal,
)
from adversarial_harness.split_manifest import (
    create_matched_split_manifest,
    load_matched_split_manifest,
)
from adversarial_harness.runner import AdversarialExperiment, _samples_by_id


class DatasetSelectionTests(unittest.TestCase):
    def _make_visa(self, root: Path) -> None:
        rows = []
        for category in ("candle", "pcb1"):
            for split, label, count in (
                ("train", "normal", 2),
                ("test", "normal", 2),
                ("test", "anomaly", 2),
            ):
                for index in range(count):
                    image = Path(category) / "Data" / "Images" / label.title() / f"{split}_{index}.JPG"
                    (root / image).parent.mkdir(parents=True, exist_ok=True)
                    (root / image).write_bytes(b"image")
                    mask = ""
                    if label == "anomaly":
                        mask_path = Path(category) / "Data" / "Masks" / "Anomaly" / f"{split}_{index}.png"
                        (root / mask_path).parent.mkdir(parents=True, exist_ok=True)
                        (root / mask_path).write_bytes(b"mask")
                        mask = mask_path.as_posix()
                    rows.append(
                        {
                            "object": category,
                            "split": split,
                            "label": label,
                            "image": image.as_posix(),
                            "mask": mask,
                        }
                    )
        manifest = root / "split_csv" / "1cls.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("object", "split", "label", "image", "mask")
            )
            writer.writeheader()
            writer.writerows(rows)

    def _make_mvtec(self, root: Path) -> None:
        for split in ("train", "test"):
            for index in range(2):
                image = root / "bottle" / split / "good" / f"{index:03d}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"image")
        for index in range(2):
            anomaly = root / "bottle" / "test" / "broken" / f"{index:03d}.png"
            mask = (
                root
                / "bottle"
                / "ground_truth"
                / "broken"
                / f"{index:03d}_mask.png"
            )
            anomaly.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            anomaly.write_bytes(b"image")
            mask.write_bytes(b"mask")

    def test_visa_manifest_discovers_test_and_train_normal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "VisA_20220922"
            self._make_visa(root)
            test = discover_visa(str(root), categories=("candle",))
            train = discover_visa_train_normal(str(root), categories=("candle",))

            self.assertEqual([sample.label for sample in test].count(0), 2)
            self.assertEqual([sample.label for sample in test].count(1), 2)
            self.assertTrue(all(sample.dataset == "visa" for sample in test + train))
            self.assertTrue(
                all(sample.mask_path is not None for sample in test if sample.label == 1)
            )
            self.assertTrue(all(sample.protocol_id.startswith("test/visa/") for sample in test))
            self.assertEqual(len(train), 2)
            lookup = _samples_by_id(test)
            for sample in test:
                self.assertIs(lookup[sample.sample_id], sample)
                self.assertIs(lookup[sample.protocol_id], sample)

    def test_visa_representative_examples_accept_detail_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sample = MVTecSample(
                index=0,
                category="candle",
                defect_type="normal",
                image_path=Path(temporary) / "0214.JPG",
                mask_path=None,
                label=0,
                dataset="visa",
            )
            config = ExperimentConfig(
                mvtec_root=None,
                visa_root="visa",
                dataset="visa",
                output_root=str(Path(temporary) / "output"),
                anomalyclip_root="AnomalyCLIP",
                anomalyclip_checkpoint="checkpoint.pth",
                metric_size=4,
                anomaly_map_sigma=0.0,
                save_adversarial_examples=1,
                resume=False,
                attack=AttackConfig(image_size=4),
            )
            experiment = AdversarialExperiment(config)
            experiment.category_thresholds = {"candle": 0.5}
            experiment.image_loader = lambda _: torch.zeros((3, 4, 4))
            details = [
                {
                    "sample_id": "visa/candle/normal/0214",
                    "category": "candle",
                    "targeted_success": 1,
                    "directional_score_shift": 0.25,
                    "lpips": 0.01,
                    "ssim": 0.99,
                    "clean_score": 0.1,
                    "adversarial_score": 0.35,
                }
            ]

            experiment._save_representative_examples(
                [sample],
                details,
                np.zeros((1, 2, 2), dtype=np.float32),
                np.ones((1, 2, 2), dtype=np.float32),
                {"all_categories": torch.zeros((1, 3, 4, 4))},
                "dataset",
                "dataset__normal_to_abnormal__global",
            )

            manifest = (
                Path(config.output_root)
                / "adversarial_examples"
                / "dataset__normal_to_abnormal__global"
                / "manifest.json"
            )
            self.assertTrue(manifest.is_file())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["categories"]["candle"]["successful_attack"]["sample_id"],
                sample.protocol_id,
            )

    def test_per_image_resume_skips_already_processed_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            samples = [
                MVTecSample(
                    index=index,
                    category="bottle",
                    defect_type="good",
                    image_path=Path(temporary) / f"{index:03d}.png",
                    mask_path=None,
                    label=0,
                )
                for index in range(2)
            ]
            config = ExperimentConfig(
                mvtec_root="mvtec",
                output_root=str(Path(temporary) / "output"),
                anomalyclip_root="AnomalyCLIP",
                anomalyclip_checkpoint="checkpoint.pth",
                device="cpu",
                target_batch_size=1,
                compute_lpips=False,
                resume=True,
                attack=AttackConfig(
                    image_size=4,
                    scopes=("per_image",),
                    directions=("normal_to_abnormal",),
                    loss_modes=("global",),
                    per_image_batch_size=1,
                ),
            )
            experiment = AdversarialExperiment(config)
            experiment.surrogate = object()
            experiment.category_thresholds = {"bottle": 0.5}
            loaded_indices = []
            experiment.image_loader = lambda sample: (
                loaded_indices.append(sample.index) or torch.zeros((3, 4, 4))
            )

            class FakeTarget:
                @staticmethod
                def predict(images):
                    count = len(images)
                    return (
                        np.full(count, 0.75, dtype=np.float32),
                        np.ones((count, 2, 2), dtype=np.float32),
                    )

            class FakeAttacker:
                def __init__(self, surrogate, attack):
                    pass

                @staticmethod
                def perturb_batch(clean, categories, target_label, mode):
                    return clean, None

            experiment.target = FakeTarget()
            partial_dir = (
                Path(config.output_root)
                / "partial"
                / "per_image__normal_to_abnormal__global"
            )
            partial_dir.mkdir(parents=True)
            np.savez_compressed(
                partial_dir / "target_outputs_partial.npz",
                indices=np.asarray([0], dtype=np.int64),
                adversarial_scores=np.asarray([0.6], dtype=np.float32),
                adversarial_lowres_maps=np.ones((1, 2, 2), dtype=np.float32),
            )
            with (partial_dir / "per_image_partial.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=("sample_id", "category"))
                writer.writeheader()
                writer.writerow(
                    {"sample_id": samples[0].sample_id, "category": "bottle"}
                )

            with patch("adversarial_harness.runner.TargetedPGD", FakeAttacker), patch(
                "adversarial_harness.runner.perceptual_metrics",
                return_value=(
                    np.zeros(1, dtype=np.float32),
                    np.ones(1, dtype=np.float32),
                    np.zeros(1, dtype=np.float32),
                ),
            ):
                _, _, details = experiment._attack_and_predict(
                    samples,
                    samples,
                    samples,
                    np.asarray([0.1, 0.2], dtype=np.float32),
                    np.zeros((2, 2, 2), dtype=np.float32),
                    "per_image",
                    "normal_to_abnormal",
                    "global",
                    object(),
                )

            self.assertEqual(loaded_indices, [1])
            self.assertEqual(len(details), 2)
            self.assertEqual({row["sample_id"] for row in details}, {
                samples[0].sample_id,
                samples[1].sample_id,
            })

    def test_universal_resume_reuses_delta_and_complete_partial_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            samples = [
                MVTecSample(
                    index=index,
                    category="bottle",
                    defect_type="good",
                    image_path=Path(temporary) / f"{index:03d}.png",
                    mask_path=None,
                    label=0,
                )
                for index in range(2)
            ]
            config = ExperimentConfig(
                mvtec_root="mvtec",
                output_root=str(Path(temporary) / "output"),
                anomalyclip_root="AnomalyCLIP",
                anomalyclip_checkpoint="checkpoint.pth",
                device="cpu",
                resume=True,
                attack=AttackConfig(
                    image_size=4,
                    scopes=("dataset",),
                    directions=("normal_to_abnormal",),
                    loss_modes=("global",),
                ),
            )
            experiment = AdversarialExperiment(config)
            experiment.surrogate = object()
            experiment.category_thresholds = {"bottle": 0.5}
            experiment.image_loader = lambda _: self.fail(
                "complete partial predictions should skip image loading"
            )

            class FakeTarget:
                @staticmethod
                def predict(images):
                    raise AssertionError(
                        "complete partial predictions should skip target inference"
                    )

            class FakeAttacker:
                def __init__(self, surrogate, attack):
                    pass

                @staticmethod
                def optimize_universal(*args, **kwargs):
                    raise AssertionError("saved universal delta should be reused")

            experiment.target = FakeTarget()
            condition = "dataset__normal_to_abnormal__global"
            partial_dir = Path(config.output_root) / "partial" / condition
            partial_dir.mkdir(parents=True)
            np.savez_compressed(
                partial_dir / "target_outputs_partial.npz",
                indices=np.asarray([0, 1], dtype=np.int64),
                adversarial_scores=np.asarray([0.6, 0.7], dtype=np.float32),
                adversarial_lowres_maps=np.ones((2, 2, 2), dtype=np.float32),
            )
            with (partial_dir / "per_image_partial.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=("sample_id", "category"))
                writer.writeheader()
                writer.writerows(
                    {
                        "sample_id": sample.sample_id,
                        "category": sample.category,
                    }
                    for sample in samples
                )

            perturbation_path = (
                Path(config.output_root)
                / "perturbations"
                / condition
                / "all_categories.pt"
            )
            perturbation_path.parent.mkdir(parents=True)
            torch.save(
                {
                    "delta": torch.zeros((1, 3, 4, 4)),
                    "epsilon": config.attack.epsilon,
                    "scope": "dataset",
                    "direction": "normal_to_abnormal",
                    "loss_mode": "global",
                    "group": "all_categories",
                    "universal_protocol": config.universal_protocol,
                    "fit_sample_ids": [sample.protocol_id for sample in samples],
                    "evaluation_sample_ids": [
                        sample.protocol_id for sample in samples
                    ],
                },
                perturbation_path,
            )

            with patch("adversarial_harness.runner.TargetedPGD", FakeAttacker):
                scores, maps, details = experiment._attack_and_predict(
                    samples,
                    samples,
                    samples,
                    np.asarray([0.1, 0.2], dtype=np.float32),
                    np.zeros((2, 2, 2), dtype=np.float32),
                    "dataset",
                    "normal_to_abnormal",
                    "global",
                    object(),
                )

            np.testing.assert_allclose(scores, [0.6, 0.7])
            np.testing.assert_allclose(maps, 1.0)
            self.assertEqual(len(details), 2)

    def test_both_mode_combines_and_reindexes_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mvtec = Path(temporary) / "mvtec"
            visa = Path(temporary) / "visa"
            self._make_mvtec(mvtec)
            self._make_visa(visa)

            samples = discover_anomaly_datasets(
                "both",
                mvtec_root=str(mvtec),
                visa_root=str(visa),
                categories=("bottle", "candle"),
            )
            self.assertEqual({sample.dataset for sample in samples}, {"mvtec", "visa"})
            self.assertEqual({sample.category for sample in samples}, {"bottle", "candle"})
            self.assertEqual([sample.index for sample in samples], list(range(len(samples))))
            self.assertEqual(len({sample.protocol_id for sample in samples}), len(samples))

    def test_generator_and_loader_support_visa_and_both(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mvtec = Path(temporary) / "mvtec"
            visa = Path(temporary) / "visa"
            self._make_mvtec(mvtec)
            self._make_visa(visa)

            for mode in ("visa", "both"):
                csv_path = Path(temporary) / f"{mode}.csv"
                json_path = Path(temporary) / f"{mode}.json"
                create_matched_split_manifest(
                    mvtec_root=str(mvtec) if mode == "both" else None,
                    visa_root=str(visa),
                    dataset=mode,
                    csv_path=str(csv_path),
                    json_path=str(json_path),
                )
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
                self.assertEqual(metadata["dataset"], mode)
                samples = discover_anomaly_datasets(
                    mode,
                    mvtec_root=str(mvtec) if mode == "both" else None,
                    visa_root=str(visa),
                )
                loaded = load_matched_split_manifest(
                    samples,
                    str(csv_path),
                    str(json_path),
                    split_seed=111,
                    fit_fraction=0.5,
                    expected_dataset=mode,
                )
                self.assertEqual(len(loaded.samples), metadata["total_selected_rows"])
                if mode == "visa":
                    self.assertTrue(
                        all(
                            sample.protocol_id.startswith("test/visa/")
                            for sample in loaded.samples
                        )
                    )

            with self.assertRaisesRegex(ValueError, "generated for dataset"):
                load_matched_split_manifest(
                    discover_anomaly_datasets("visa", visa_root=str(visa)),
                    str(Path(temporary) / "visa.csv"),
                    str(Path(temporary) / "visa.json"),
                    split_seed=111,
                    fit_fraction=0.5,
                    expected_dataset="both",
                )

    def test_config_requires_only_roots_selected_by_mode(self) -> None:
        common = {
            "output_root": "output",
            "anomalyclip_root": "AnomalyCLIP",
            "anomalyclip_checkpoint": "checkpoint.pth",
        }
        visa = ExperimentConfig(
            mvtec_root=None, visa_root="visa", dataset="visa", **common
        )
        self.assertEqual(visa.dataset, "visa")
        with self.assertRaisesRegex(ValueError, "visa_root"):
            ExperimentConfig(mvtec_root="mvtec", dataset="both", **common)
        manifest_config = ExperimentConfig(
            mvtec_root=None,
            visa_root="visa",
            dataset="visa",
            use_split_manifest=True,
            split_manifest_csv="split.csv",
            split_manifest_json="split.json",
            **common,
        )
        self.assertTrue(manifest_config.use_split_manifest)
        automatic_manifest = ExperimentConfig(
            mvtec_root="mvtec",
            dataset="mvtec",
            use_split_manifest=True,
            **common,
        )
        self.assertEqual(
            Path(automatic_manifest.split_manifest_csv).name,
            "mvtec_matched_test_per_category_v1_seed111.csv",
        )

    def test_cross_dataset_protocol_fits_source_and_evaluates_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mvtec = Path(temporary) / "mvtec"
            visa = Path(temporary) / "visa"
            self._make_mvtec(mvtec)
            self._make_visa(visa)
            config = ExperimentConfig(
                mvtec_root=str(mvtec),
                visa_root=str(visa),
                dataset="mvtec_to_visa",
                output_root=str(Path(temporary) / "output"),
                anomalyclip_root="AnomalyCLIP",
                anomalyclip_checkpoint="checkpoint.pth",
                universal_protocol="held_out",
                resume=False,
                attack=AttackConfig(scopes=("dataset",)),
            )
            experiment = AdversarialExperiment(config)
            target_test = discover_anomaly_datasets("visa", visa_root=str(visa))
            target_train = discover_anomaly_datasets(
                "visa", visa_root=str(visa), train_normal=True
            )
            source_test = discover_anomaly_datasets(
                "mvtec", mvtec_root=str(mvtec)
            )
            source_train = discover_anomaly_datasets(
                "mvtec", mvtec_root=str(mvtec), train_normal=True
            )

            normal_fit, normal_eval, normal_all = experiment._protocol_samples(
                target_test,
                target_train,
                "dataset",
                "normal_to_abnormal",
                source_test_samples=source_test,
                source_train_normal_samples=source_train,
            )
            anomaly_fit, anomaly_eval, anomaly_all = experiment._protocol_samples(
                target_test,
                target_train,
                "dataset",
                "abnormal_to_normal",
                source_test_samples=source_test,
                source_train_normal_samples=source_train,
            )
            self.assertTrue(all(sample.dataset == "mvtec" for sample in normal_fit))
            self.assertTrue(all(sample.dataset == "mvtec" for sample in anomaly_fit))
            self.assertTrue(all(sample.dataset == "visa" for sample in normal_eval))
            self.assertTrue(all(sample.dataset == "visa" for sample in anomaly_eval))
            self.assertTrue(all(sample.dataset == "visa" for sample in normal_all))
            self.assertTrue(all(sample.dataset == "visa" for sample in anomaly_all))

            csv_path = Path(temporary) / "both_cross.csv"
            json_path = Path(temporary) / "both_cross.json"
            create_matched_split_manifest(
                mvtec_root=str(mvtec),
                visa_root=str(visa),
                dataset="both",
                csv_path=str(csv_path),
                json_path=str(json_path),
            )
            combined = discover_anomaly_datasets(
                "both", mvtec_root=str(mvtec), visa_root=str(visa)
            )
            loaded = load_matched_split_manifest(
                combined,
                str(csv_path),
                str(json_path),
                split_seed=111,
                fit_fraction=0.5,
                expected_dataset="both",
            )
            experiment.config.use_split_manifest = True
            experiment.split_assignments = loaded.assignments
            manifest_source = [
                sample for sample in loaded.samples if sample.dataset == "mvtec"
            ]
            manifest_target = [
                sample for sample in loaded.samples if sample.dataset == "visa"
            ]
            fit, evaluation, evaluation_all = experiment._protocol_samples(
                manifest_target,
                target_train,
                "dataset",
                "abnormal_to_normal",
                source_test_samples=manifest_source,
                source_train_normal_samples=source_train,
            )
            self.assertTrue(
                all(experiment.split_assignments[sample.protocol_id] == "fit" for sample in fit)
            )
            self.assertTrue(
                all(
                    experiment.split_assignments[sample.protocol_id] == "evaluation"
                    for sample in evaluation_all
                )
            )
            self.assertTrue(all(sample.dataset == "visa" for sample in evaluation))

            reverse_config = ExperimentConfig(
                mvtec_root=str(mvtec),
                visa_root=str(visa),
                dataset="visa_to_mvtec",
                output_root=str(Path(temporary) / "reverse_output"),
                anomalyclip_root="AnomalyCLIP",
                anomalyclip_checkpoint="checkpoint.pth",
                universal_protocol="held_out",
                resume=False,
                attack=AttackConfig(scopes=("dataset",)),
            )
            reverse = AdversarialExperiment(reverse_config)
            reverse_fit, reverse_evaluation, reverse_all = reverse._protocol_samples(
                discover_anomaly_datasets("mvtec", mvtec_root=str(mvtec)),
                discover_anomaly_datasets(
                    "mvtec", mvtec_root=str(mvtec), train_normal=True
                ),
                "dataset",
                "abnormal_to_normal",
                source_test_samples=discover_anomaly_datasets(
                    "visa", visa_root=str(visa)
                ),
                source_train_normal_samples=discover_anomaly_datasets(
                    "visa", visa_root=str(visa), train_normal=True
                ),
            )
            self.assertTrue(all(sample.dataset == "visa" for sample in reverse_fit))
            self.assertTrue(
                all(sample.dataset == "mvtec" for sample in reverse_evaluation)
            )
            self.assertTrue(all(sample.dataset == "mvtec" for sample in reverse_all))

        with self.assertRaisesRegex(ValueError, "only the universal 'dataset'"):
            ExperimentConfig(
                mvtec_root="mvtec",
                visa_root="visa",
                dataset="visa_to_mvtec",
                output_root="output",
                anomalyclip_root="AnomalyCLIP",
                anomalyclip_checkpoint="checkpoint.pth",
                universal_protocol="held_out",
            )


if __name__ == "__main__":
    unittest.main()
