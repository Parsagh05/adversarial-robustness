from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from adversarial_harness.config import AttackConfig, ExperimentConfig
from adversarial_harness.dataset import (
    discover_anomaly_datasets,
    discover_visa,
    discover_visa_train_normal,
)
from adversarial_harness.split_manifest import (
    create_matched_split_manifest,
    load_matched_split_manifest,
)
from adversarial_harness.runner import AdversarialExperiment


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
