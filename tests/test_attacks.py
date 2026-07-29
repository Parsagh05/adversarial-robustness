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


class MaskAwareLocalLossTests(unittest.TestCase):
    def test_defect_mask_focuses_local_loss_and_zero_mask_falls_back(self) -> None:
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
        normal_unmasked = attacker._group_losses(
            global_features,
            patch_features,
            ["object"],
            target_label=1,
            mode="local",
        )["local"]

        self.assertGreater(float(masked), float(unmasked))
        self.assertAlmostEqual(float(normal_fallback), float(normal_unmasked), places=6)


if __name__ == "__main__":
    unittest.main()
