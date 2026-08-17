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
    evaluation_mode_root,
)


class PathContractTests(unittest.TestCase):
    def test_bundle_path_is_namespaced_by_setup_and_formulation(self) -> None:
        root = Path("outputs/datasets_mvtec")
        self.assertEqual(
            bundle_path(root, "steps500_eps2", "ce_focal_dice"),
            root
            / "setups"
            / "steps500_eps2"
            / "ce_focal_dice"
            / "attack_generation"
            / BUNDLE_DIRECTORY,
        )
        self.assertEqual(
            evaluation_mode_root(root, "steps500_eps2", "margin_topk", "fixed_0_5"),
            root
            / "setups"
            / "steps500_eps2"
            / "margin_topk"
            / "evaluation"
            / "fixed_0_5",
        )

    def test_two_formulations_of_one_setup_never_share_a_folder(self) -> None:
        # Generation rewrites attack_manifest.csv for the whole bundle and
        # evaluation reads completeness off one summary.csv, so a shared folder
        # would make two single-formulation runs discard each other's work.
        root = Path("outputs/datasets_mvtec")
        self.assertNotEqual(
            bundle_path(root, "steps500_eps2", "ce_focal_dice"),
            bundle_path(root, "steps500_eps2", "margin_topk"),
        )
        self.assertNotEqual(
            evaluation_mode_root(root, "steps500_eps2", "ce_focal_dice", "fixed_0_5"),
            evaluation_mode_root(root, "steps500_eps2", "margin_topk", "fixed_0_5"),
        )

    def test_path_components_must_be_plain_directory_names(self) -> None:
        for setup_id, formulation in (
            ("../escape", "ce_focal_dice"),
            ("steps500_eps2", "nested/name"),
            ("steps500_eps2", ".."),
            ("steps500_eps2", ""),
        ):
            with self.subTest(setup_id=setup_id, loss_formulation=formulation):
                with self.assertRaises(ValueError):
                    bundle_path(Path("outputs"), setup_id, formulation)

    def test_legacy_directory_bundle_is_repaired_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "datasets_mvtec"
            bundle = bundle_path(results, "steps500_eps2", "ce_focal_dice")
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

    def _mixed_formulation_bundle(self, root: Path) -> Path:
        """Write a bundle whose manifest mixes a legacy and a margin_topk row."""

        bundle = root / BUNDLE_DIRECTORY
        perturbations = bundle / "mvtec" / "f1" / "perturbations"
        perturbations.mkdir(parents=True)
        (bundle / "evaluation_test_indices.csv").write_text(
            "protocol_id,dataset,category,label,partition\n"
            "test/toy/good/000,mvtec,toy,0,evaluation\n"
            "test/toy/crack/001,mvtec,toy,1,evaluation\n",
            encoding="utf-8",
        )
        rows = []
        # The legacy row leaves loss_formulation empty, exactly as bundles
        # generated before this ablation existed do.
        for stem, formulation in (
            ("dataset__normal_to_abnormal__global", ""),
            ("dataset__normal_to_abnormal__global__margin_topk", "margin_topk"),
        ):
            delta = perturbations / f"{stem}.pt"
            torch.save({"delta": torch.zeros((1, 3, 4, 4))}, delta)
            checksum = hashlib.sha256(delta.read_bytes()).hexdigest()
            rows.append(
                "dataset,mvtec,mvtec,normal_to_abnormal,0,1,global,1,"
                f"noises/mvtec/f1/perturbations/{stem}.pt,"
                f"delta,{checksum},4,{2 / 255},{formulation}\n"
            )
        (bundle / "attack_manifest.csv").write_text(
            "scope,source_dataset,target_dataset,direction,source_label,"
            "target_label,loss_mode,evaluation_attacked_image_count,noise_file,"
            "noise_tensor_key,artifact_sha256,image_size,epsilon,"
            "loss_formulation\n" + "".join(rows),
            encoding="utf-8",
        )
        return bundle

    def test_legacy_and_margin_topk_rows_coexist_in_one_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._mixed_formulation_bundle(Path(temporary))

            artifacts = load_manifest(
                bundle,
                scopes=("per_dataset",),
                sources=("mvtec",),
                targets=("mvtec",),
                verify_files=True,
            )

            self.assertEqual(len(artifacts), 2)
            by_formulation = {
                artifact.record["loss_formulation"]: artifact for artifact in artifacts
            }
            self.assertEqual(sorted(by_formulation), ["ce_focal_dice", "margin_topk"])
            # The legacy row keeps the condition name it has today.
            self.assertEqual(
                by_formulation["ce_focal_dice"].name,
                "mvtec__mvtec__normal_to_abnormal__global__per_dataset",
            )
            self.assertEqual(
                by_formulation["margin_topk"].name,
                "mvtec__mvtec__normal_to_abnormal__global__margin_topk__per_dataset",
            )

    def test_loss_formulation_filter_selects_one_objective(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._mixed_formulation_bundle(Path(temporary))

            artifacts = load_manifest(
                bundle,
                scopes=("per_dataset",),
                sources=("mvtec",),
                targets=("mvtec",),
                loss_formulations=("margin_topk",),
            )

            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].record["loss_formulation"], "margin_topk")

    def test_missing_protocol_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = bundle_path(Path(temporary), "steps500_eps2", "ce_focal_dice")
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
