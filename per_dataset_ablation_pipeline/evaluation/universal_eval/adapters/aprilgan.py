"""APRIL-GAN target adapter using the official repository and checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
OFFICIAL_BACKBONE = "ViT-L-14-336"
OFFICIAL_PRETRAINED = "openai"
OFFICIAL_FEATURE_LAYERS = (6, 12, 18, 24)


class _LinearLayer(nn.Module):
    """The repository's trainable patch projection layer."""

    def __init__(self, dim_in: int, dim_out: int, count: int) -> None:
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(count)])

    def forward(self, tokens: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [layer(value[:, 1:, :]) for layer, value in zip(self.fc, tokens)]


def _normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def _prepare_import(repository_root: str | Path):
    root = Path(repository_root).expanduser().resolve()
    required = (
        root / "open_clip" / "__init__.py",
        root / "open_clip" / "model_configs" / "ViT-L-14-336.json",
        root / "prompt_ensemble.py",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Expected the official VAND-APRIL-GAN repository at {root}; "
            f"missing: {missing}"
        )
    root_string = str(root)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)
    importlib.invalidate_caches()

    # APRIL-GAN vendors its modified OpenCLIP package under the generic
    # ``open_clip`` name. Do not accidentally reuse another model's package.
    for module_name in list(sys.modules):
        if module_name != "open_clip" and not module_name.startswith("open_clip."):
            continue
        module = sys.modules.get(module_name)
        module_path = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_path.startswith(root_string):
            sys.modules.pop(module_name, None)
    return root, importlib.import_module("open_clip")


def _torch_load(path: Path, *, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _encode_text_with_prompt_ensemble(
    model: nn.Module,
    objects: Sequence[str],
    tokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Match the official ``prompt_ensemble.py`` normal/anomaly prompts."""

    prompt_states = (
        (
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage",
        ),
        ("damaged {}", "broken {}", "{} with flaw", "{} with defect", "{} with damage"),
    )
    templates = (
        "a bad photo of a {}.",
        "a low resolution photo of the {}.",
        "a bad photo of the {}.",
        "a cropped photo of the {}.",
        "a bright photo of a {}.",
        "a dark photo of the {}.",
        "a photo of my {}.",
        "a photo of the cool {}.",
        "a close-up photo of a {}.",
        "a black and white photo of the {}.",
        "a bright photo of the {}.",
        "a cropped photo of a {}.",
        "a jpeg corrupted photo of a {}.",
        "a blurry photo of the {}.",
        "a photo of the {}.",
        "a good photo of the {}.",
        "a photo of one {}.",
        "a close-up photo of the {}.",
        "a photo of a {}.",
        "a low resolution photo of a {}.",
        "a photo of a large {}.",
        "a blurry photo of a {}.",
        "a jpeg corrupted photo of the {}.",
        "a good photo of a {}.",
        "a photo of the small {}.",
        "a photo of the large {}.",
        "a black and white photo of a {}.",
        "a dark photo of a {}.",
        "a photo of a cool {}.",
        "a photo of a small {}.",
        "there is a {} in the scene.",
        "there is the {} in the scene.",
        "this is a {} in the scene.",
        "this is the {} in the scene.",
        "this is one {} in the scene.",
    )
    result: dict[str, torch.Tensor] = {}
    for object_name in objects:
        state_features = []
        for states in prompt_states:
            sentences = [
                template.format(state.format(object_name))
                for state in states
                for template in templates
            ]
            embeddings = model.encode_text(tokenizer(sentences).to(device))
            embeddings = F.normalize(embeddings, dim=-1)
            state_features.append(F.normalize(embeddings.mean(dim=0), dim=0))
        result[object_name] = torch.stack(state_features, dim=1).to(device)
    return result


@register_adapter("aprilgan")
@register_adapter("april-gan")
class APRILGANAdapter(ModelAdapter):
    """Official zero-shot APRIL-GAN inference with OpenAI ViT-L/14@336px."""

    model_name = "APRIL-GAN"

    def __init__(
        self,
        *,
        repository_root: str,
        checkpoint_path: str,
        device: str = "cuda",
        image_size: int = 518,
        clip_model_name: str = OFFICIAL_BACKBONE,
        clip_pretrained: str = OFFICIAL_PRETRAINED,
        feature_layers: Sequence[int] = OFFICIAL_FEATURE_LAYERS,
        clip_cache_dir: str | None = None,
    ) -> None:
        if (
            clip_model_name != OFFICIAL_BACKBONE
            or clip_pretrained != OFFICIAL_PRETRAINED
        ):
            raise ValueError(
                "Official APRIL-GAN requires OpenAI ViT-L/14@336px "
                f"({OFFICIAL_BACKBONE!r}, pretrained={OFFICIAL_PRETRAINED!r})"
            )
        if image_size != 518:
            raise ValueError("Official APRIL-GAN evaluation requires image_size=518")
        normalized_layers = tuple(int(layer) for layer in feature_layers)
        if normalized_layers != OFFICIAL_FEATURE_LAYERS:
            raise ValueError(
                "Official APRIL-GAN requires feature layers "
                f"{OFFICIAL_FEATURE_LAYERS}"
            )

        root, open_clip = _prepare_import(repository_root)
        config_path = root / "open_clip" / "model_configs" / "ViT-L-14-336.json"
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_config = {
            "embed_dim": 768,
            "image_size": 336,
            "layers": 24,
            "width": 1024,
            "patch_size": 14,
        }
        actual_config = {
            "embed_dim": model_config.get("embed_dim"),
            **{
                name: model_config.get("vision_cfg", {}).get(name)
                for name in ("image_size", "layers", "width", "patch_size")
            },
        }
        if actual_config != expected_config:
            raise ValueError(
                "APRIL-GAN's ViT-L/14@336px config does not match the verified "
                f"official architecture: {actual_config}"
            )

        self.device = torch.device(device)
        self.image_size = image_size
        self.feature_layers = normalized_layers
        model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name,
            image_size,
            pretrained=clip_pretrained,
            device=self.device,
            cache_dir=clip_cache_dir,
        )
        model.eval().requires_grad_(False)
        self.model = model
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

        linearlayer = _LinearLayer(
            expected_config["width"], expected_config["embed_dim"], len(normalized_layers)
        ).to(self.device)
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"APRIL-GAN checkpoint not found: {checkpoint}")
        state = _torch_load(checkpoint, map_location="cpu")
        if not isinstance(state, dict) or "trainable_linearlayer" not in state:
            raise KeyError(
                f"APRIL-GAN checkpoint has no 'trainable_linearlayer' state: {checkpoint}"
            )
        linearlayer.load_state_dict(state["trainable_linearlayer"], strict=True)
        linearlayer.eval().requires_grad_(False)
        self.linearlayer = linearlayer
        self.text_prompts: dict[str, torch.Tensor] = {}

    def _text_features(self, categories: Sequence[str]) -> torch.Tensor:
        missing = sorted(set(categories) - self.text_prompts.keys())
        if missing:
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                self.text_prompts.update(
                    _encode_text_with_prompt_ensemble(
                        self.model, missing, self.tokenizer, self.device
                    )
                )
        return torch.stack([self.text_prompts[name] for name in categories], dim=0)

    def predict(self, images_01: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        raise ValueError(
            "APRIL-GAN needs one category per image; use predict_with_categories()"
        )

    def predict_with_categories(
        self, images_01: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if images_01.ndim != 4 or images_01.shape[1] != 3:
            raise ValueError("Model input must have shape [B, 3, H, W]")
        if len(categories) != len(images_01):
            raise ValueError("Categories must contain one entry per image")
        if images_01.shape[-2:] != (self.image_size, self.image_size):
            images_01 = F.interpolate(
                images_01,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        images = _normalize(images_01.to(self.device))
        text_features = self._text_features(categories)

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            image_features, patch_tokens = self.model.encode_image(
                images, self.feature_layers
            )
            image_features = F.normalize(image_features, dim=-1)
            image_logits = 100.0 * torch.einsum(
                "bd,bdc->bc", image_features, text_features
            )
            scores = image_logits.softmax(dim=-1)[:, 1]

            projected_tokens = self.linearlayer(patch_tokens)
            anomaly_maps = []
            for tokens in projected_tokens:
                tokens = F.normalize(tokens, dim=-1)
                logits = 100.0 * torch.einsum(
                    "bld,bdc->blc", tokens, text_features
                )
                side = int(np.sqrt(logits.shape[1]))
                if side * side != logits.shape[1]:
                    raise RuntimeError(
                        f"APRIL-GAN returned a non-square patch grid: {logits.shape[1]}"
                    )
                anomaly_map = F.interpolate(
                    logits.permute(0, 2, 1).reshape(-1, 2, side, side),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=True,
                ).softmax(dim=1)[:, 1]
                anomaly_maps.append(anomaly_map)
            maps = torch.stack(anomaly_maps, dim=0).sum(dim=0)

        return (
            scores.float().cpu().numpy().astype(np.float32),
            maps.float().cpu().numpy().astype(np.float32),
        )

    def release(self) -> None:
        self.text_prompts.clear()
        del self.linearlayer
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
