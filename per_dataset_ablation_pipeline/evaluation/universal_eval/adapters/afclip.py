"""AF-CLIP target adapter using the official repository and released weights."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
OFFICIAL_BACKBONE = "ViT-L/14@336px"


def _normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def _prepare_import(repository_root: str | Path):
    root = Path(repository_root).expanduser().resolve()
    required = (root / "clip" / "clip.py", root / "clip" / "model.py")
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Expected the official AF-CLIP repository at {root}; missing: {missing}"
        )
    root_string = str(root)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)
    importlib.invalidate_caches()

    # The official repository intentionally uses the top-level package name
    # ``clip``. Remove a package left by another model notebook before import.
    for module_name in list(sys.modules):
        if module_name != "clip" and not module_name.startswith("clip."):
            continue
        module = sys.modules.get(module_name)
        module_path = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_path.startswith(root_string):
            sys.modules.pop(module_name, None)
    return importlib.import_module("clip.clip")


def _torch_load(path: Path, *, map_location: torch.device):
    """Load the released module objects on both old and new PyTorch versions."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0 has no weights_only argument.
        return torch.load(path, map_location=map_location)


@register_adapter("afclip")
@register_adapter("af-clip")
class AFCLIPAdapter(ModelAdapter):
    """Official zero-shot AF-CLIP inference with ViT-L/14@336px."""

    model_name = "AF-CLIP"

    def __init__(
        self,
        *,
        repository_root: str,
        prompt_checkpoint_path: str,
        adaptor_checkpoint_path: str,
        device: str = "cuda",
        image_size: int = 518,
        clip_model_name: str = OFFICIAL_BACKBONE,
        clip_download_root: str = "",
        prompt_len: int = 12,
        feature_layers: tuple[int, ...] = (6, 12, 18, 24),
    ) -> None:
        if clip_model_name != OFFICIAL_BACKBONE:
            raise ValueError(
                f"Official AF-CLIP requires {OFFICIAL_BACKBONE!r}, got "
                f"{clip_model_name!r}"
            )
        if image_size != 518:
            raise ValueError("Official AF-CLIP evaluation requires image_size=518")

        self.device = torch.device(device)
        self.image_size = image_size
        clip_module = _prepare_import(repository_root)
        model, _ = clip_module.load(
            name=clip_model_name,
            jit=False,
            device=self.device,
            download_root=clip_download_root or None,
        )
        args = SimpleNamespace(
            prompt_len=prompt_len,
            feature_layers=list(feature_layers),
            memory_layers=list(feature_layers),
            alpha=0.1,
            fewshot=0,
        )
        model.insert(args=args, tokenizer=clip_module.tokenize, device=self.device)

        prompt_path = Path(prompt_checkpoint_path).expanduser().resolve()
        adaptor_path = Path(adaptor_checkpoint_path).expanduser().resolve()
        for label, path in (("prompt", prompt_path), ("adaptor", adaptor_path)):
            if not path.is_file():
                raise FileNotFoundError(f"AF-CLIP {label} checkpoint not found: {path}")

        prompt = _torch_load(prompt_path, map_location=self.device)
        prompt_tensor = torch.as_tensor(prompt).detach().to(self.device)
        model.state_prompt_embedding = torch.nn.Parameter(
            prompt_tensor, requires_grad=False
        )
        adaptor = _torch_load(adaptor_path, map_location=self.device)
        if not isinstance(adaptor, torch.nn.Module):
            raise TypeError(f"AF-CLIP adaptor checkpoint is not a module: {adaptor_path}")
        model.adaptor = adaptor.to(self.device)
        model.eval().requires_grad_(False)

        self.model = model
        self.args = args

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
        images = _normalize(images_01.to(self.device))
        with torch.inference_mode():
            scores, maps = self.model.detect_forward(images, self.args)
        return (
            scores.detach().cpu().numpy().astype(np.float32),
            maps[:, 0].detach().cpu().numpy().astype(np.float32),
        )

    def release(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
