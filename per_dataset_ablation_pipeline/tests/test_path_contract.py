from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from evaluation.universal_eval.artifacts import _resolve_noise_path, load_manifest
from path_contract import (
    BUNDLE_DIRECTORY,
    bundle_path,
    ensure_bundle_protocol_files,
)


class PathContractTests(unittest.TestCase):
    def test_bundle_path_is_namespaced_by_setup(self) -> None:
        root = Path("outputs/datasets_mvtec")
        self.assertEqual(
            bundle_path(root, "steps500_eps2"),
            root
            / "setups"
            / "steps500_eps2"
            / "attack_generation"
            / BUNDLE_DIRECTORY,
        )

    def test_legacy_directory_bundle_is_repaired_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "datasets_mvtec"
            bundle = bundle_path(results, "steps500_eps2")
            protocol = bundle.parent / "protocol"
            protocol.mkdir(parents=True)
            (protocol / "attack_train_indices.csv").write_text(
                "protocol_id,dataset,category,label,partition\n"
                "train/toy/good/000,mvtec,toy,0,attack_train\n",
                encoding="utf-8",
            )
            (protocol / "evaluation_test_indices.csv").write_text(
                "protocol_id,dataset,category,label,partition\n"
                "test/toy/good/000,mvtec,toy,0,evaluation\n"
                "test/toy/crack/001,mvtec,toy,1,evaluation\n",
                encoding="utf-8",
            )

            # Legacy uncompressed layout omitted the manifest's noises/ prefix.
            delta = (
                bundle
                / "mvtec"
                / "f1"
                / "perturbations"
                / "dataset__normal_to_abnormal__global.pt"
            )
            delta.parent.mkdir(parents=True)
            torch.save({"delta": torch.zeros((1, 3, 4, 4))}, delta)
            checksum = hashlib.sha256(delta.read_bytes()).hexdigest()
            (bundle / "attack_manifest.csv").write_text(
                "scope,source_dataset,target_dataset,direction,source_label,"
                "target_label,loss_mode,evaluation_attacked_image_count,noise_file,"
                "noise_tensor_key,artifact_sha256,image_size,epsilon\n"
                "dataset,mvtec,mvtec,normal_to_abnormal,0,1,global,1,"
                "noises/mvtec/f1/perturbations/dataset__normal_to_abnormal__global.pt,"
                f"delta,{checksum},4,{2 / 255}\n",
                encoding="utf-8",
            )

            repaired = ensure_bundle_protocol_files(bundle)
            self.assertEqual(len(repaired), 2)
            artifacts = load_manifest(
                bundle,
                scopes=("per_dataset",),
                sources=("mvtec",),
                targets=("mvtec",),
                verify_files=True,
            )
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].delta_path, delta.resolve())

    def test_missing_protocol_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = bundle_path(Path(temporary), "steps500_eps2")
            with self.assertRaisesRegex(FileNotFoundError, "both"):
                ensure_bundle_protocol_files(bundle)

    def test_canonical_noise_path_resolves_inside_noises_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / BUNDLE_DIRECTORY
            delta = bundle / "noises" / "mvtec" / "f1" / "delta.pt"
            delta.parent.mkdir(parents=True)
            delta.touch()
            self.assertEqual(
                _resolve_noise_path(bundle, "noises/mvtec/f1/delta.pt"),
                delta,
            )


if __name__ == "__main__":
    unittest.main()
