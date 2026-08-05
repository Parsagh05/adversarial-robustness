from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from blackbox_evaluation_pipeline.universal_eval.artifacts import load_manifest


class ArtifactManifestTests(unittest.TestCase):
    def test_stale_absolute_paths_are_resolved_inside_extracted_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "canonical"
            delta_path = root / "mvtec_to_visa" / "all_categories.pt"
            delta_path.parent.mkdir(parents=True)
            torch.save(
                {"delta": torch.zeros(1, 3, 4, 4), "metadata": {}},
                delta_path,
            )
            checksum = hashlib.sha256(delta_path.read_bytes()).hexdigest()
            record = {
                "source_dataset": "mvtec",
                "target_dataset": "visa",
                "direction": "normal_to_abnormal",
                "loss_mode": "global",
                "scope": "dataset",
                "source_label": 0,
                "target_label": 1,
                "target_evaluation_all_sample_ids": ["test/visa/x/normal/001"],
                "target_attacked_sample_ids": ["test/visa/x/normal/001"],
                "artifact_path": "/kaggle/working/canonical/mvtec_to_visa/all_categories.pt",
                "artifact_file_sha256": checksum,
                "epsilon": 8 / 255,
                "image_size": 4,
            }
            (root / "all_canonical_attack_artifacts.json").write_text(
                json.dumps([record]), encoding="utf-8"
            )
            artifacts = load_manifest(root)
            self.assertEqual(artifacts[0].delta_path, delta_path)
            self.assertEqual(tuple(artifacts[0].load_delta().shape), (1, 3, 4, 4))


if __name__ == "__main__":
    unittest.main()
