from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from adversarial_harness.attacks import TargetedPGD
from adversarial_harness.config import AttackConfig


class _FakeSurrogate:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.prompts = {
            "object": SimpleNamespace(
                normal_embeddings=torch.tensor([[1.0, 0.0]]),
                abnormal_embeddings=torch.tensor([[0.0, 1.0]]),
            )
        }


class _DifferentiableFakeSurrogate(_FakeSurrogate):
    def encode_visual(self, images_01, include_patches=True):
        signal = images_01.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
        token = torch.stack((1.0 - signal, signal), dim=-1)
        cls = token[:, None, :]
        patches = token[:, None, :].expand(-1, 4, -1)
        return token, [torch.cat((cls, patches), dim=1)] if include_patches else []


class MaskAwareLocalLossTests(unittest.TestCase):
    def test_defect_mask_and_fixed_normal_region_focus_local_loss(self) -> None:
        attacker = TargetedPGD(
            _FakeSurrogate(),
            AttackConfig(
                temperature=1.0,
                mask_local_loss=True,
                local_background_weight=0.0,
            ),
        )
        global_features = torch.zeros((1, 2))
        # CLS, one anomalous defect token, and three normal background tokens.
        patch_features = [
            torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        ]
        defect_mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        zero_mask = torch.zeros_like(defect_mask)

        unmasked = attacker._group_losses(
            global_features,
            patch_features,
            ["object"],
            target_label=0,
            mode="local",
        )["local"]
        masked = attacker._group_losses(
            global_features,
            patch_features,
            ["object"],
            target_label=0,
            mode="local",
            spatial_masks=defect_mask,
        )["local"]
        normal_fallback = attacker._group_losses(
            global_features,
            patch_features,
            ["object"],
            target_label=1,
            mode="local",
            spatial_masks=zero_mask,
        )["local"]
        full_image_attacker = TargetedPGD(
            _FakeSurrogate(),
            AttackConfig(
                temperature=1.0,
                mask_local_loss=True,
                local_background_weight=0.0,
                normal_local_target="full_image",
            ),
        )
        normal_full_image = full_image_attacker._group_losses(
            global_features,
            patch_features,
            ["object"],
            target_label=1,
            mode="local",
        )["local"]

        self.assertGreater(float(masked), float(unmasked))
        self.assertNotAlmostEqual(
            float(normal_fallback), float(normal_full_image), places=6
        )

    def test_one_targeted_local_step_reduces_same_batch_loss(self) -> None:
        attacker = TargetedPGD(
            _DifferentiableFakeSurrogate(),
            AttackConfig(
                image_size=2,
                epsilon=0.5,
                step_size=0.05,
                steps=1,
                random_start=False,
                temperature=1.0,
                local_focal_weight=0.5,
                local_dice_weight=0.5,
            ),
        )
        clean = torch.zeros((1, 3, 2, 2))
        before = attacker.objective(clean, ["object"], 1, "local")
        adversarial, _ = attacker.perturb_batch(clean, ["object"], 1, "local")
        after = attacker.objective(adversarial, ["object"], 1, "local")

        self.assertTrue(torch.isfinite(before))
        self.assertTrue(torch.isfinite(after))
        self.assertLess(float(after), float(before))


class MarginTopKLossTests(unittest.TestCase):
    """The relaxed s(x) = z_a(x) - z_n(x) margin and TopK anomaly-map loss."""

    def _attacker(self, **overrides) -> TargetedPGD:
        settings = {
            "temperature": 1.0,
            "loss_formulation": "margin_topk",
        }
        settings.update(overrides)
        return TargetedPGD(_FakeSurrogate(), AttackConfig(**settings))

    def test_direction_sign_maximizes_for_normal_and_minimizes_for_anomalous(
        self,
    ) -> None:
        attacker = self._attacker()
        # One abnormal token: normalized margin z_a - z_n is exactly +1.
        global_features = torch.tensor([[0.0, 1.0]])
        patch_features = [torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])]

        to_abnormal = attacker._group_losses(
            global_features, patch_features, ["object"], 1, "combined"
        )
        to_normal = attacker._group_losses(
            global_features, patch_features, ["object"], 0, "combined"
        )

        # The unsigned diagnostics are identical; only the optimized sign flips.
        self.assertAlmostEqual(
            float(to_abnormal["global_margin"]), float(to_normal["global_margin"])
        )
        self.assertAlmostEqual(
            float(to_abnormal["local_topk"]), float(to_normal["local_topk"])
        )
        self.assertAlmostEqual(
            float(to_abnormal["global"]), -float(to_abnormal["global_margin"])
        )
        self.assertAlmostEqual(
            float(to_normal["global"]), float(to_normal["global_margin"])
        )
        self.assertAlmostEqual(
            float(to_abnormal["local"]), -float(to_abnormal["local_topk"])
        )
        self.assertAlmostEqual(
            float(to_normal["local"]), float(to_normal["local_topk"])
        )

    def test_topk_ignores_tokens_below_the_cut(self) -> None:
        global_features = torch.zeros((1, 2))
        # CLS plus one abnormal token (margin +1) and three normal ones (-1).
        patch_features = [
            torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        ]

        top_one = self._attacker(margin_topk_fraction=0.01)._group_losses(
            global_features, patch_features, ["object"], 1, "local"
        )["local_topk"]
        every_token = self._attacker(margin_topk_fraction=1.0)._group_losses(
            global_features, patch_features, ["object"], 1, "local"
        )["local_topk"]

        self.assertAlmostEqual(float(top_one), 1.0, places=6)
        self.assertAlmostEqual(float(every_token), -0.5, places=6)

    def test_shipped_direction_fractions_select_the_expected_token_counts(self) -> None:
        # The shipped grid is 518/14 = 37 per side, so 1369 patch tokens. The
        # first 274 (20%) read as abnormal (+1) and the other 1095 as normal.
        abnormal = torch.tensor([0.0, 1.0]).expand(274, 2)
        normal = torch.tensor([1.0, 0.0]).expand(1095, 2)
        cls_token = torch.tensor([[1.0, 0.0]])
        patch_features = [torch.cat((cls_token, abnormal, normal))[None]]
        global_features = torch.zeros((1, 2))

        # K=20% selects exactly the 274 abnormal tokens, so the mean is +1.
        plant = self._attacker(margin_topk_fraction=0.20)._group_losses(
            global_features, patch_features, ["object"], 1, "local"
        )["local_topk"]
        # K=40% selects 548 tokens: the same 274 plus 274 normal ones.
        suppress = self._attacker(margin_topk_fraction=0.40)._group_losses(
            global_features, patch_features, ["object"], 0, "local"
        )["local_topk"]

        self.assertAlmostEqual(float(plant), 1.0, places=6)
        self.assertAlmostEqual(float(suppress), 0.0, places=6)

    def test_no_ground_truth_mask_changes_the_relaxed_loss(self) -> None:
        attacker = self._attacker()
        global_features = torch.zeros((1, 2))
        patch_features = [
            torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        ]
        defect_mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

        without_mask = attacker._group_losses(
            global_features, patch_features, ["object"], 1, "local"
        )["local"]
        with_mask = attacker._group_losses(
            global_features,
            patch_features,
            ["object"],
            1,
            "local",
            spatial_masks=defect_mask,
        )["local"]

        self.assertAlmostEqual(float(without_mask), float(with_mask), places=6)

    def test_one_targeted_step_reduces_the_margin_loss_in_every_mode(self) -> None:
        for mode in ("global", "local", "combined"):
            with self.subTest(mode=mode):
                attacker = TargetedPGD(
                    _DifferentiableFakeSurrogate(),
                    AttackConfig(
                        image_size=2,
                        epsilon=0.5,
                        step_size=0.05,
                        steps=1,
                        random_start=False,
                        temperature=1.0,
                        loss_formulation="margin_topk",
                    ),
                )
                clean = torch.zeros((1, 3, 2, 2))
                before = attacker.objective(clean, ["object"], 1, mode)
                adversarial, _ = attacker.perturb_batch(clean, ["object"], 1, mode)
                after = attacker.objective(adversarial, ["object"], 1, mode)

                self.assertTrue(torch.isfinite(before))
                self.assertTrue(torch.isfinite(after))
                self.assertLess(float(after), float(before))

    def test_unknown_formulation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "loss_formulation"):
            AttackConfig(loss_formulation="topk_only")
        with self.assertRaisesRegex(ValueError, "loss_formulations"):
            AttackConfig(loss_formulations=("ce_focal_dice", "topk_only"))
        with self.assertRaisesRegex(ValueError, "margin_topk_fraction"):
            AttackConfig(margin_topk_fraction=0.0)


if __name__ == "__main__":
    unittest.main()
