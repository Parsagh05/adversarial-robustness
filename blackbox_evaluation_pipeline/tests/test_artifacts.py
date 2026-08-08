from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from blackbox_evaluation_pipeline.universal_eval.artifacts import load_manifest


MANIFEST_HEADER = (
    "scope,source_dataset,target_dataset,category,direction,source_label,"
    "target_label,loss_mode,evaluation_attacked_image_count,noise_file,"
    "noise_tensor_key,artifact_sha256,image_size,epsilon\n"
)
INDEX_HEADER = "protocol_id,dataset,category,label,partition\n"


class ArtifactManifestTests(unittest.TestCase):
    def test_dataset_root_selects_scope_and_resolves_noise_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root = Path(temporary) / "adversarial-attacks-vlm-survey"
            bundle = dataset_root / "canonical_clip_per_dataset"
            delta_path = (
                bundle / "noises" / "mvtec" / "f1" / "perturbations"
                / "dataset__normal_to_abnormal__global.pt"
            )
            delta_path.parent.mkdir(parents=True)
            torch.save({"delta": torch.zeros(1, 3, 4, 4)}, delta_path)
            checksum = hashlib.sha256(delta_path.read_bytes()).hexdigest()
            (bundle / "attack_manifest.csv").write_text(
                MANIFEST_HEADER
                + "dataset,mvtec,mvtec,,normal_to_abnormal,0,1,global,1,"
                + "noises/mvtec/f1/perturbations/dataset__normal_to_abnormal__global.pt,"
                + f"delta,{checksum},4,{8 / 255}\n",
                encoding="utf-8",
            )
            (bundle / "evaluation_test_indices.csv").write_text(
                INDEX_HEADER
                + "test/toy/good/000,mvtec,toy,0,evaluation\n"
                + "test/toy/crack/001,mvtec,toy,1,evaluation\n",
                encoding="utf-8",
            )

            artifacts = load_manifest(dataset_root, scopes=("per_dataset",))

            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].delta_path, delta_path)
            self.assertEqual(artifacts[0].record["scope"], "per_dataset")
            self.assertEqual(artifacts[0].attacked_ids, ("test/toy/good/000",))
            self.assertEqual(tuple(artifacts[0].load_delta().shape), (1, 3, 4, 4))

    def test_per_image_tensor_rows_follow_evaluation_csv_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "canonical_clip_per_image"
            delta_path = bundle / "noises" / "visa" / "perturbations" / "toy.pt"
            delta_path.parent.mkdir(parents=True)
            torch.save({"delta": torch.zeros(2, 3, 4, 4)}, delta_path)
            checksum = hashlib.sha256(delta_path.read_bytes()).hexdigest()
            (bundle / "attack_manifest.csv").write_text(
                MANIFEST_HEADER
                + "per_image,visa,visa,toy,abnormal_to_normal,1,0,local,2,"
                + f"noises/visa/perturbations/toy.pt,delta,{checksum},4,{8 / 255}\n",
                encoding="utf-8",
            )
            (bundle / "evaluation_test_indices.csv").write_text(
                INDEX_HEADER
                + "test/visa/toy/normal/001,visa,toy,0,evaluation\n"
                + "test/visa/toy/bad/002,visa,toy,1,evaluation\n"
                + "test/visa/toy/bad/003,visa,toy,1,evaluation\n",
                encoding="utf-8",
            )

            artifact = load_manifest(bundle, scopes=("per_image",))[0]

            self.assertEqual(
                artifact.delta_indices(),
                {"test/visa/toy/bad/002": 0, "test/visa/toy/bad/003": 1},
            )
            self.assertEqual(tuple(artifact.load_delta().shape), (2, 3, 4, 4))


if __name__ == "__main__":
    unittest.main()
