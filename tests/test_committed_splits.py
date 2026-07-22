from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = ROOT / "splits"
MODES = ("mvtec", "visa", "both")
PROTOCOL = "matched_test_per_category_v1"


class CommittedSplitArtifactTests(unittest.TestCase):
    def _load(self, mode: str) -> tuple[dict, list[dict[str, str]]]:
        stem = f"{mode}_matched_test_per_category_v1_seed111"
        csv_path = SPLIT_ROOT / f"{stem}.csv"
        json_path = SPLIT_ROOT / f"{stem}.json"
        self.assertTrue(csv_path.is_file(), csv_path)
        self.assertTrue(json_path.is_file(), json_path)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        self.assertEqual(digest, metadata["csv_sha256"])
        return metadata, rows

    def test_hashes_counts_ids_and_strata(self) -> None:
        for mode in MODES:
            with self.subTest(mode=mode):
                metadata, rows = self._load(mode)
                self.assertEqual(metadata["dataset"], mode)
                self.assertEqual(metadata["protocol_version"], PROTOCOL)
                self.assertEqual(len(rows), metadata["total_manifest_rows"])
                self.assertEqual(
                    len(
                        [
                            row
                            for row in rows
                            if row["partition"] in {"fit", "evaluation"}
                        ]
                    ),
                    metadata["total_selected_rows"],
                )
                ids = [row["protocol_id"] for row in rows]
                self.assertEqual(len(ids), len(set(ids)))
                for category, expected in metadata["category_counts"].items():
                    category_rows = [
                        row for row in rows if row["category"] == category
                    ]
                    actual = {
                        "fit_normal": sum(
                            row["partition"] == "fit" and row["label"] == "0"
                            for row in category_rows
                        ),
                        "fit_anomaly": sum(
                            row["partition"] == "fit" and row["label"] == "1"
                            for row in category_rows
                        ),
                        "evaluation_normal": sum(
                            row["partition"] == "evaluation"
                            and row["label"] == "0"
                            for row in category_rows
                        ),
                        "evaluation_anomaly": sum(
                            row["partition"] == "evaluation"
                            and row["label"] == "1"
                            for row in category_rows
                        ),
                    }
                    for key, count in actual.items():
                        self.assertEqual(count, expected[key], f"{mode}/{category}/{key}")
                    self.assertEqual(len(set(actual.values())), 1)

    def test_both_is_exact_union_of_standalone_manifests(self) -> None:
        _, mvtec = self._load("mvtec")
        _, visa = self._load("visa")
        _, both = self._load("both")
        standalone = {row["protocol_id"]: row for row in mvtec + visa}
        combined = {row["protocol_id"]: row for row in both}
        self.assertEqual(combined, standalone)


if __name__ == "__main__":
    unittest.main()
