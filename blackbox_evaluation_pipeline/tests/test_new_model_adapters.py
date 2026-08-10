from __future__ import annotations

import unittest

import numpy as np
import torch

from blackbox_evaluation_pipeline.universal_eval.adapters import available_adapters
from blackbox_evaluation_pipeline.universal_eval.adapters.afclip import AFCLIPAdapter
from blackbox_evaluation_pipeline.universal_eval.adapters.aprilgan import (
    APRILGANAdapter,
    _LinearLayer,
)
from blackbox_evaluation_pipeline.universal_eval.adapters.filo import (
    FiLoAdapter,
    _category_name,
    _gaussian_blur_3x3_sigma4,
)


class NewAdapterRegistrationTests(unittest.TestCase):
    def test_new_adapters_are_registered(self) -> None:
        self.assertIn("afclip", available_adapters())
        self.assertIn("aprilgan", available_adapters())
        self.assertIn("filo", available_adapters())

    def test_afclip_rejects_non_336_backbone_before_loading_repository(self) -> None:
        with self.assertRaisesRegex(ValueError, "ViT-L/14@336px"):
            AFCLIPAdapter(
                repository_root="missing",
                prompt_checkpoint_path="missing",
                adaptor_checkpoint_path="missing",
                clip_model_name="ViT-L/14",
                device="cpu",
            )

    def test_filo_rejects_non_336_backbone_before_loading_repository(self) -> None:
        with self.assertRaisesRegex(ValueError, "ViT-L/14@336px"):
            FiLoAdapter(
                repository_root="missing",
                checkpoint_path="missing",
                grounding_checkpoint_path="missing",
                target_dataset="mvtec",
                clip_model_name="ViT-L-14",
                device="cpu",
            )

    def test_aprilgan_rejects_non_336_backbone_before_loading_repository(self) -> None:
        with self.assertRaisesRegex(ValueError, "ViT-L/14@336px"):
            APRILGANAdapter(
                repository_root="missing",
                checkpoint_path="missing",
                clip_model_name="ViT-L-14",
                device="cpu",
            )


class FiLoPreprocessingTests(unittest.TestCase):
    def test_category_names_match_official_space_separated_prompts(self) -> None:
        self.assertEqual(_category_name("metal_nut"), "metal nut")
        self.assertEqual(_category_name("pipe_fryum"), "pipe fryum")

    def test_official_blur_preserves_a_constant_map(self) -> None:
        value = torch.ones((2, 1, 8, 8), dtype=torch.float32)
        result = _gaussian_blur_3x3_sigma4(value)
        np.testing.assert_allclose(result.numpy(), value.numpy(), atol=1e-6)


class APRILGANProjectionTests(unittest.TestCase):
    def test_projection_drops_class_token_for_each_feature_layer(self) -> None:
        projection = _LinearLayer(dim_in=3, dim_out=2, count=2)
        tokens = [torch.randn(2, 5, 3), torch.randn(2, 5, 3)]
        projected = projection(tokens)
        self.assertEqual([tuple(value.shape) for value in projected], [(2, 4, 2)] * 2)


if __name__ == "__main__":
    unittest.main()
