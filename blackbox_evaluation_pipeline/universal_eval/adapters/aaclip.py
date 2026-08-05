"""AA-CLIP target adapter using the official repository and adapter weights."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DATASET_NAMES = {"mvtec": "MVTec", "visa": "VisA"}


def _normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def _prepare_import(repository_root: str | Path):
    root = Path(repository_root).expanduser().resolve()
    required = (
        root / "model" / "adapter.py",
        root / "model" / "clip.py",
        root / "forward_utils.py",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Expected the official AA-CLIP repository at {root}; missing: {missing}"
        )
    root_string = str(root)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)
    importlib.invalidate_caches()

    # AA-CLIP uses top-level module names that commonly collide with other repos.
    for module_name in list(sys.modules):
        if not (
            module_name in {"dataset", "forward_utils", "model", "utils"}
            or module_name.startswith(("dataset.", "model."))
        ):
            continue
        module = sys.modules.get(module_name)
        module_path = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_path.startswith(root_string):
            sys.modules.pop(module_name, None)

    adapter_module = importlib.import_module("model.adapter")
    clip_module = importlib.import_module("model.clip")
    forward_module = importlib.import_module("forward_utils")
    utils_module = importlib.import_module("utils")
    return adapter_module, clip_module, forward_module, utils_module


def _load_component_checkpoint(
    module: torch.nn.Module,
    checkpoint_path: str | Path,
    component: str,
) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AA-CLIP {component} checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or component not in checkpoint:
        raise KeyError(f"Checkpoint has no {component!r} state: {path}")
    module.load_state_dict(checkpoint[component])
    return path


@register_adapter("aaclip")
@register_adapter("aa-clip")
class AACLIPAdapter(ModelAdapter):
    """Paper-default AA-CLIP inference for MVTec AD and VisA categories."""

    model_name = "AA-CLIP"

    def __init__(
        self,
        *,
        repository_root: str,
        image_checkpoint_path: str,
        text_checkpoint_path: str | None = None,
        target_dataset: str,
        device: str = "cuda",
        image_size: int = 518,
        model_name: str = "ViT-L-14-336",
        seed: int = 111,
        text_adapt_weight: float = 0.1,
        image_adapt_weight: float = 0.1,
        text_adapt_until: int = 3,
        image_adapt_until: int = 6,
        levels: Sequence[int] = (6, 12, 18, 24),
        relu: bool = False,
    ) -> None:
        normalized_dataset = target_dataset.strip().lower()
        if normalized_dataset not in DATASET_NAMES:
            raise ValueError(
                f"AA-CLIP target_dataset must be one of {sorted(DATASET_NAMES)}"
            )
        if image_size != 518:
            raise ValueError("Paper-default AA-CLIP evaluation requires image_size=518")

        self.device = torch.device(device)
        self.image_size = image_size
        self.dataset_name = DATASET_NAMES[normalized_dataset]
        adapter_module, clip_module, forward_module, utils_module = _prepare_import(
            repository_root
        )
        utils_module.setup_seed(seed)

        clip_model = clip_module.create_model(
            model_name=model_name,
            img_size=image_size,
            device=self.device,
            pretrained="openai",
            require_pretrained=True,
        )
        clip_model.eval()
        model = adapter_module.AdaptedCLIP(
            clip_model=clip_model,
            text_adapt_weight=text_adapt_weight,
            image_adapt_weight=image_adapt_weight,
            text_adapt_until=text_adapt_until,
            image_adapt_until=image_adapt_until,
            levels=list(levels),
            relu=relu,
        ).to(self.device)
        _load_component_checkpoint(
            model.image_adapter, image_checkpoint_path, "image_adapter"
        )

        self.uses_text_adapter = bool(text_checkpoint_path)
        if self.uses_text_adapter:
            _load_component_checkpoint(
                model.text_adapter, str(text_checkpoint_path), "text_adapter"
            )
            text_model = model
        else:
            text_model = clip_model
        model.eval()
        model.requires_grad_(False)

        with torch.inference_mode():
            self.text_embeddings = forward_module.get_adapted_text_embedding(
                text_model, self.dataset_name, self.device
            )
            self.text_embeddings = {
                category: embedding.detach()
                for category, embedding in self.text_embeddings.items()
            }
        self.model = model
        self.forward_module = forward_module

    def predict(self, images_01: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        raise ValueError(
            "AA-CLIP needs one category per image; use predict_with_categories()"
        )

    def predict_with_categories(
        self, images_01: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if images_01.ndim != 4 or images_01.shape[1] != 3:
            raise ValueError("Model input must have shape [B, 3, H, W]")
        if len(categories) != len(images_01):
            raise ValueError("Categories must contain one entry per image")
        unknown = sorted(set(categories) - set(self.text_embeddings))
        if unknown:
            raise ValueError(
                f"Unknown {self.dataset_name} AA-CLIP categories: {unknown}"
            )
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
            patch_features, detection_features = self.model(images)
            batch_scores = torch.empty(
                len(images), dtype=torch.float32, device=self.device
            )
            batch_maps = torch.empty(
                (len(images), self.image_size, self.image_size),
                dtype=torch.float32,
                device=self.device,
            )
            for category in dict.fromkeys(categories):
                indices = [
                    index for index, value in enumerate(categories) if value == category
                ]
                index_tensor = torch.as_tensor(indices, device=self.device)
                text = self.text_embeddings[category]

                # These are the exact raw image scores used by official test.py.
                prediction = detection_features[index_tensor] @ text
                batch_scores[index_tensor] = (prediction[:, 1] + 1.0) / 2.0

                maps = []
                for features in patch_features:
                    maps.append(
                        self.forward_module.calculate_similarity_map(
                            features[index_tensor],
                            text,
                            self.image_size,
                            test=True,
                            domain="Industrial",
                        )
                    )
                batch_maps[index_tensor] = torch.cat(maps, dim=1).sum(dim=1)

        return (
            batch_scores.cpu().numpy().astype(np.float32),
            batch_maps.cpu().numpy().astype(np.float32),
        )

    def postprocess_image_scores(
        self,
        scores: np.ndarray,
        map_mins: np.ndarray,
        map_maxs: np.ndarray,
        categories: Sequence[str],
    ) -> np.ndarray:
        """Reproduce official metrics_eval image-score aggregation per category."""

        result = np.asarray(scores, dtype=np.float64).copy()
        map_mins = np.asarray(map_mins, dtype=np.float64)
        map_maxs = np.asarray(map_maxs, dtype=np.float64)
        categories_array = np.asarray(categories)
        if not (
            result.shape == map_mins.shape == map_maxs.shape == categories_array.shape
        ):
            raise ValueError("AA-CLIP score postprocessing inputs must have matching shapes")

        for category in dict.fromkeys(categories):
            selected = categories_array == category
            category_scores = result[selected]
            category_map_maxs = map_maxs[selected]
            pixel_min = float(map_mins[selected].min())
            pixel_max = float(category_map_maxs.max())

            # Match the conditionals and operations in official forward_utils.metrics_eval.
            if pixel_max != 1.0:
                if pixel_max == pixel_min:
                    raise ValueError(f"Constant AA-CLIP anomaly maps for {category!r}")
                category_map_maxs = (
                    category_map_maxs - pixel_min
                ) / (pixel_max - pixel_min)
            if float(category_scores.max()) != 1.0:
                score_min = float(category_scores.min())
                score_max = float(category_scores.max())
                if score_max == score_min:
                    raise ValueError(f"Constant AA-CLIP image scores for {category!r}")
                category_scores = (
                    category_scores - score_min
                ) / (score_max - score_min)
            result[selected] = 0.5 * category_map_maxs + 0.5 * category_scores
        return result.astype(np.float32)

    def release(self) -> None:
        del self.text_embeddings
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
