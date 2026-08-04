from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from adversarial_harness.dataset import discover_mvtec
from adversarial_harness.config import AttackConfig, ExperimentConfig
from adversarial_harness.runner import AdversarialExperiment
from adversarial_harness.split_manifest import (
    MATCHED_SPLIT_PROTOCOL,
    create_matched_split_manifest,
    load_matched_split_manifest,
)


class SplitManifestTests(unittest.TestCase):
    def _make_dataset(self, root: Path) -> None:
        specifications = {
            "bottle": {"good": 6, "broken": 8},
            "carpet": {"good": 5, "hole": 4},
        }
        for category, defect_counts in specifications.items():
            for defect_type, count in defect_counts.items():
                image_dir = root / category / "test" / defect_type
                image_dir.mkdir(parents=True, exist_ok=True)
                if defect_type != "good":
                    mask_dir = root / category / "ground_truth" / defect_type
                    mask_dir.mkdir(parents=True, exist_ok=True)
                for index in range(count):
                    (image_dir / f"{index:03d}.png").write_bytes(b"test")
                    if defect_type != "good":
                        (mask_dir / f"{index:03d}_mask.png").write_bytes(b"mask")

    def test_generation_is_balanced_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mvtec"
            self._make_dataset(root)
            csv_path = Path(temporary) / "splits" / "split.csv"
            json_path = Path(temporary) / "splits" / "split.json"
            create_matched_split_manifest(
                str(root), str(csv_path), str(json_path), split_seed=111
            )

            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["protocol_version"], MATCHED_SPLIT_PROTOCOL)
            self.assertEqual(metadata["category_counts"]["bottle"]["fit_normal"], 3)
            self.assertEqual(metadata["category_counts"]["bottle"]["fit_anomaly"], 3)
            self.assertEqual(metadata["category_counts"]["carpet"]["fit_normal"], 2)
            self.assertEqual(metadata["category_counts"]["carpet"]["fit_anomaly"], 2)
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 23)
            self.assertTrue(any(row["partition"] == "excluded_balance" for row in rows))
            second_csv = Path(temporary) / "second.csv"
            second_json = Path(temporary) / "second.json"
            create_matched_split_manifest(
                str(root), str(second_csv), str(second_json), split_seed=111
            )
            self.assertEqual(csv_path.read_bytes(), second_csv.read_bytes())
            second_metadata = json.loads(second_json.read_text(encoding="utf-8"))
            self.assertEqual(metadata["csv_sha256"], second_metadata["csv_sha256"])
            self.assertEqual(
                metadata["category_counts"], second_metadata["category_counts"]
            )

    def test_loading_and_smaller_cap_keep_all_four_strata_equal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mvtec"
            self._make_dataset(root)
            csv_path = Path(temporary) / "split.csv"
            json_path = Path(temporary) / "split.json"
            create_matched_split_manifest(
                str(root), str(csv_path), str(json_path), split_seed=111
            )
            samples = discover_mvtec(str(root))
            loaded = load_matched_split_manifest(
                samples,
                str(csv_path),
                str(json_path),
                split_seed=111,
                fit_fraction=0.5,
                max_samples_per_category=6,
            )
            # Six cannot preserve four exactly equal strata, so it safely rounds
            # down to four samples per category.
            self.assertEqual(len(loaded.samples), 8)
            self.assertEqual(
                loaded.metadata["runtime_counts"]["bottle"],
                {
                    "fit_normal": 1,
                    "fit_anomaly": 1,
                    "evaluation_normal": 1,
                    "evaluation_anomaly": 1,
                },
            )
            self.assertEqual(
                [sample.index for sample in loaded.samples], list(range(8))
            )

    def test_loader_rejects_modified_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mvtec"
            self._make_dataset(root)
            csv_path = Path(temporary) / "split.csv"
            json_path = Path(temporary) / "split.json"
            create_matched_split_manifest(
                str(root), str(csv_path), str(json_path), split_seed=111
            )
            csv_path.write_text(
                csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "hash"):
                load_matched_split_manifest(
                    discover_mvtec(str(root)),
                    str(csv_path),
                    str(json_path),
                    split_seed=111,
                    fit_fraction=0.5,
                )

    def test_loader_accepts_git_line_ending_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mvtec"
            self._make_dataset(root)
            csv_path = Path(temporary) / "split.csv"
            json_path = Path(temporary) / "split.json"
            create_matched_split_manifest(
                str(root), str(csv_path), str(json_path), split_seed=111
            )
            normalized = csv_path.read_bytes().replace(b"\r\n", b"\n")
            csv_path.write_bytes(normalized)
            loaded = load_matched_split_manifest(
                discover_mvtec(str(root)),
                str(csv_path),
                str(json_path),
                split_seed=111,
                fit_fraction=0.5,
            )
            self.assertGreater(len(loaded.samples), 0)

    def test_loader_requires_minimum_cap_of_four(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mvtec"
            self._make_dataset(root)
            csv_path = Path(temporary) / "split.csv"
            json_path = Path(temporary) / "split.json"
            create_matched_split_manifest(str(root), str(csv_path), str(json_path))
            with self.assertRaisesRegex(ValueError, "at least 4"):
                load_matched_split_manifest(
                    discover_mvtec(str(root)),
                    str(csv_path),
                    str(json_path),
                    split_seed=111,
                    fit_fraction=0.5,
                    max_samples_per_category=2,
                )

    def test_runner_uses_same_manifest_evaluation_cohort_for_both_directions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mvtec"
            self._make_dataset(root)
            csv_path = Path(temporary) / "split.csv"
            json_path = Path(temporary) / "split.json"
            create_matched_split_manifest(str(root), str(csv_path), str(json_path))
            loaded = load_matched_split_manifest(
                discover_mvtec(str(root)),
                str(csv_path),
                str(json_path),
                split_seed=111,
                fit_fraction=0.5,
            )
            config = ExperimentConfig(
                mvtec_root=str(root),
                output_root=str(Path(temporary) / "output"),
                anomalyclip_root=str(Path(temporary) / "AnomalyCLIP"),
                anomalyclip_checkpoint=str(Path(temporary) / "checkpoint.pth"),
                universal_protocol="held_out",
                use_split_manifest=True,
                split_manifest_csv=str(csv_path),
                split_manifest_json=str(json_path),
                attack=AttackConfig(scopes=("dataset",)),
            )
            experiment = AdversarialExperiment(config)
            experiment.split_assignments = loaded.assignments
            normal_fit, normal_eval, normal_all = experiment._protocol_samples(
                loaded.samples, [], "dataset", "normal_to_abnormal"
            )
            anomaly_fit, anomaly_eval, anomaly_all = experiment._protocol_samples(
                loaded.samples, [], "dataset", "abnormal_to_normal"
            )
            self.assertTrue(all(sample.label == 0 for sample in normal_fit + normal_eval))
            self.assertTrue(all(sample.label == 1 for sample in anomaly_fit + anomaly_eval))
            self.assertEqual(
                {sample.protocol_id for sample in normal_all},
                {sample.protocol_id for sample in anomaly_all},
            )
            self.assertEqual(len(normal_fit), len(anomaly_fit))
            self.assertEqual(len(normal_eval), len(anomaly_eval))
            self.assertFalse(
                {sample.protocol_id for sample in normal_fit}
                & {sample.protocol_id for sample in normal_eval}
            )


if __name__ == "__main__":
    unittest.main()
