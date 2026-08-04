"""AnomalyCLIP target adapter using the official repository and checkpoints."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def _prepare_import(repository_root: str | Path):
    root = Path(repository_root).expanduser().resolve()
    if not (root / "AnomalyCLIP_lib").is_dir():
        raise FileNotFoundError(
            f"Expected AnomalyCLIP_lib under official repository: {root}"
        )
    root_string = str(root)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)
    importlib.invalidate_caches()
    for module_name in ("utils", "prompt_ensemble"):
        module = sys.modules.get(module_name)
        module_path = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_path.startswith(root_string):
            sys.modules.pop(module_name, None)
    return (
        importlib.import_module("AnomalyCLIP_lib"),
        importlib.import_module("prompt_ensemble"),
    )


@register_adapter("anomalyclip")
class AnomalyCLIPAdapter(ModelAdapter):
    model_name = "AnomalyCLIP"

    def __init__(
        self,
        *,
        repository_root: str,
        checkpoint_path: str,
        device: str = "cuda",
        image_size: int = 518,
        features_list: Sequence[int] = (6, 12, 18, 24),
        feature_map_indices: Sequence[int] = (0, 1, 2, 3),
        depth: int = 9,
        n_ctx: int = 12,
        t_n_ctx: int = 4,
        dpam_layer: int = 20,
        clip_model_name: str = "ViT-L/14@336px",
        clip_download_root: str = "",
    ) -> None:
        self.device = torch.device(device)
        self.image_size = image_size
        self.features_list = tuple(features_list)
        self.feature_map_indices = set(feature_map_indices)
        self.dpam_layer = dpam_layer
        library, prompt_module = _prepare_import(repository_root)
        self.library = library
        details = {
            "Prompt_length": n_ctx,
            "learnabel_text_embedding_depth": depth,
            "learnabel_text_embedding_length": t_n_ctx,
        }
        load_kwargs: dict[str, object] = {
            "device": self.device,
            "design_details": details,
        }
        if clip_download_root:
            load_kwargs["download_root"] = clip_download_root
        self.model, _ = library.load(clip_model_name, **load_kwargs)
        self.model.eval()

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"AnomalyCLIP checkpoint not found: {checkpoint}")
        prompt_learner = prompt_module.AnomalyCLIP_PromptLearner(
            self.model.to("cpu"), details
        )
        state = torch.load(checkpoint, map_location="cpu")
        if "prompt_learner" not in state:
            raise KeyError(f"Checkpoint has no prompt_learner state: {checkpoint}")
        prompt_learner.load_state_dict(state["prompt_learner"])
        prompt_learner.to(self.device).eval()
        self.model.to(self.device)
        self.model.visual.DAPM_replace(DPAM_layer=dpam_layer)
        self.model.requires_grad_(False)
        prompt_learner.requires_grad_(False)

        with torch.inference_mode():
            prompts, tokenized, compound = prompt_learner(cls_id=None)
            text = self.model.encode_text_learn(prompts, tokenized, compound).float()
            text = torch.stack(torch.chunk(text, chunks=2, dim=0), dim=1)
            self.text_features = F.normalize(text, dim=-1).detach()
        self.prompt_learner = prompt_learner

    def predict(self, images_01: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if images_01.ndim != 4 or images_01.shape[1] != 3:
            raise ValueError("Model input must have shape [B, 3, H, W]")
        if images_01.shape[-2:] != (self.image_size, self.image_size):
            images_01 = F.interpolate(
                images_01,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        # Model-specific preprocessing happens here, after clean/adversarial
        # RGB tensors have been constructed by the shared evaluator.
        images = _normalize(images_01.to(self.device))
        with torch.inference_mode():
            image_features, patch_features = self.model.encode_image(
                images, list(self.features_list), DPAM_layer=self.dpam_layer
            )
            image_features = F.normalize(image_features.float(), dim=-1)
            logits = image_features @ self.text_features[0].t()
            scores = (logits / 0.07).softmax(dim=-1)[:, 1]
            maps = []
            for index, patch in enumerate(patch_features):
                if index not in self.feature_map_indices:
                    continue
                patch = F.normalize(patch.float(), dim=-1)
                similarity, _ = self.library.compute_similarity(
                    patch, self.text_features[0]
                )
                similarity = similarity[:, 1:, :]
                side = int(similarity.shape[1] ** 0.5)
                if side * side != similarity.shape[1]:
                    raise ValueError(
                        f"Patch-token count is not square: {similarity.shape[1]}"
                    )
                similarity = similarity.reshape(similarity.shape[0], side, side, 2)
                maps.append((similarity[..., 1] + 1.0 - similarity[..., 0]) / 2.0)
            if not maps:
                raise RuntimeError("No AnomalyCLIP feature maps were selected")
            anomaly_maps = torch.stack(maps, dim=0).sum(dim=0)
        return (
            scores.detach().cpu().numpy().astype(np.float32),
            anomaly_maps.detach().cpu().numpy().astype(np.float32),
        )

    def release(self) -> None:
        del self.text_features
        del self.prompt_learner
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

