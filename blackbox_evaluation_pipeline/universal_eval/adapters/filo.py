"""FiLo target adapter using its official CLIP and Grounding DINO models."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OFFICIAL_BACKBONE = "ViT-L-14-336"


def _normalize(images: torch.Tensor, mean_values, std_values) -> torch.Tensor:
    mean = images.new_tensor(mean_values).view(1, 3, 1, 1)
    std = images.new_tensor(std_values).view(1, 3, 1, 1)
    return (images - mean) / std


def _prepare_import(repository_root: str | Path):
    root = Path(repository_root).expanduser().resolve()
    required = (
        root / "models" / "FiLo.py",
        root / "test.py",
        root
        / "models"
        / "GroundingDINO"
        / "groundingdino"
        / "config"
        / "GroundingDINO_SwinT_OGC.py",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Expected the official FiLo repository at {root}; missing: {missing}"
        )
    root_string = str(root)
    grounding_root_string = str(root / "models" / "GroundingDINO")
    # FiLo imports Grounding DINO both as ``models.GroundingDINO`` and through
    # its own absolute ``groundingdino`` imports. Expose both official roots.
    for import_root in (root_string, grounding_root_string):
        if import_root in sys.path:
            sys.path.remove(import_root)
        sys.path.insert(0, import_root)
    importlib.invalidate_caches()

    # FiLo uses the generic top-level package name ``models``.
    for module_name in list(sys.modules):
        if module_name != "models" and not module_name.startswith("models."):
            continue
        module = sys.modules.get(module_name)
        module_path = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_path.startswith(root_string):
            sys.modules.pop(module_name, None)

    filo_module = importlib.import_module("models.FiLo")
    slconfig_module = importlib.import_module(
        "models.GroundingDINO.groundingdino.util.slconfig"
    )
    builder_module = importlib.import_module(
        "models.GroundingDINO.groundingdino.models"
    )
    utils_module = importlib.import_module(
        "models.GroundingDINO.groundingdino.util.utils"
    )

    # Reuse the exact released caption filtering and phrase extraction helper.
    spec = importlib.util.spec_from_file_location("_official_filo_test", root / "test.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FiLo test.py from {root}")
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    return filo_module, slconfig_module, builder_module, utils_module, test_module


def _torch_load(path: Path, *, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _gaussian_blur_3x3_sigma4(value: torch.Tensor) -> torch.Tensor:
    """Match torchvision GaussianBlur(3, 4.0) without an eager torchvision import."""

    coordinates = torch.arange(-1, 2, device=value.device, dtype=value.dtype)
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * 4.0**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    channels = value.shape[1]
    weight = kernel_2d.expand(channels, 1, 3, 3)
    return F.conv2d(F.pad(value, (1, 1, 1, 1), mode="reflect"), weight, groups=channels)


def _category_name(value: str) -> str:
    return value.replace("_", " ")


@register_adapter("filo")
class FiLoAdapter(ModelAdapter):
    """Official zero-shot FiLo inference with location-aware anomaly prompts."""

    model_name = "FiLo"

    def __init__(
        self,
        *,
        repository_root: str,
        checkpoint_path: str,
        grounding_checkpoint_path: str,
        target_dataset: str,
        device: str = "cuda",
        image_size: int = 518,
        clip_model_name: str = OFFICIAL_BACKBONE,
        clip_pretrained: str = "openai",
        features_list: Sequence[int] = (6, 12, 18, 24),
        n_ctx: int = 12,
        grounding_config_path: str | None = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        area_threshold: float = 0.7,
    ) -> None:
        if clip_model_name != OFFICIAL_BACKBONE or clip_pretrained != "openai":
            raise ValueError(
                "Official FiLo requires OpenAI ViT-L/14@336px "
                f"({OFFICIAL_BACKBONE!r}, pretrained='openai')"
            )
        if image_size != 518:
            raise ValueError("Official FiLo evaluation requires image_size=518")
        normalized_dataset = target_dataset.strip().lower()
        if normalized_dataset not in {"mvtec", "visa"}:
            raise ValueError("FiLo target_dataset must be 'mvtec' or 'visa'")

        root = Path(repository_root).expanduser().resolve()
        (
            filo_module,
            slconfig_module,
            builder_module,
            grounding_utils,
            test_module,
        ) = _prepare_import(root)
        self.device = torch.device(device)
        self.image_size = image_size
        self.target_dataset = normalized_dataset
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.area_threshold = area_threshold
        self.test_module = test_module

        object_names = (
            filo_module.mvtec_obj_list
            if normalized_dataset == "mvtec"
            else filo_module.visa_obj_list
        )
        self.categories = {_category_name(name) for name in object_names}
        args = SimpleNamespace(
            clip_model=clip_model_name,
            clip_pretrained=clip_pretrained,
            image_size=image_size,
            features_list=list(features_list),
            n_ctx=n_ctx,
            device=str(self.device),
        )
        model = filo_module.FiLo(sorted(self.categories), args, str(self.device)).to(
            self.device
        )
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FiLo checkpoint not found: {checkpoint}")
        state = _torch_load(checkpoint, map_location="cpu")
        if not isinstance(state, dict) or "filo" not in state:
            raise KeyError(f"Checkpoint has no 'filo' state: {checkpoint}")
        model.load_state_dict(state["filo"], strict=False)
        model.eval().requires_grad_(False)
        self.model = model

        config_path = Path(grounding_config_path).expanduser().resolve() if grounding_config_path else (
            root
            / "models"
            / "GroundingDINO"
            / "groundingdino"
            / "config"
            / "GroundingDINO_SwinT_OGC.py"
        )
        grounding_checkpoint = Path(grounding_checkpoint_path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Grounding DINO config not found: {config_path}")
        if not grounding_checkpoint.is_file():
            raise FileNotFoundError(
                f"Grounding DINO checkpoint not found: {grounding_checkpoint}"
            )
        grounding_args = slconfig_module.SLConfig.fromfile(str(config_path))
        grounding_args.device = str(self.device)
        grounding_model = builder_module.build_model(grounding_args)
        grounding_state = _torch_load(grounding_checkpoint, map_location="cpu")
        if isinstance(grounding_state, dict) and "model" in grounding_state:
            grounding_state = grounding_state["model"]
        grounding_model.load_state_dict(
            grounding_utils.clean_state_dict(grounding_state), strict=False
        )
        grounding_model.to(self.device).eval().requires_grad_(False)
        self.grounding_model = grounding_model

    def _grounding_context(
        self, image_01: torch.Tensor, category: str
    ) -> tuple[list[str], torch.Tensor]:
        details = (
            self.test_module.mvtec_anomaly_detail_gpt
            if self.target_dataset == "mvtec"
            else self.test_module.visa_anomaly_detail_gpt
        )[category]
        caption = " .\n".join(self.test_module.anomaly_status_general + details)
        dino_image = _normalize(image_01, IMAGENET_MEAN, IMAGENET_STD)[0]
        boxes, phrases = self.test_module.get_grounding_output(
            self.grounding_model,
            dino_image,
            caption,
            self.box_threshold,
            self.text_threshold,
            category=category,
            device=str(self.device),
            area_thr=self.area_threshold,
        )

        boxes_xyxy = boxes.clone()
        boxes_xyxy *= self.image_size
        boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
        boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

        accepted = details + self.test_module.anomaly_status_general
        best_box = None
        best_score = 0.0
        for box, phrase in zip(boxes_xyxy, phrases):
            if not self.test_module.check_elements_in_array(accepted, phrase):
                continue
            match = re.search(r"\((.*?)\)", phrase)
            score = float(match.group(1)) if match else 0.0
            if score >= best_score:
                best_box = box
                best_score = score

        center = (
            ((best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2)
            if best_box is not None
            else (self.image_size // 2, self.image_size // 2)
        )
        positions = []
        for position, ((x1, y1), (x2, y2)) in self.test_module.location.items():
            if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                positions.append(position)
                break
        return positions, boxes_xyxy.cpu()

    def predict(self, images_01: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        raise ValueError("FiLo needs one category per image; use predict_with_categories()")

    def predict_with_categories(
        self, images_01: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if images_01.ndim != 4 or images_01.shape[1] != 3:
            raise ValueError("Model input must have shape [B, 3, H, W]")
        if len(categories) != len(images_01):
            raise ValueError("Categories must contain one entry per image")
        normalized_categories = [_category_name(value) for value in categories]
        unknown = sorted(set(normalized_categories) - self.categories)
        if unknown:
            raise ValueError(f"Unknown FiLo categories for {self.target_dataset}: {unknown}")
        if images_01.shape[-2:] != (self.image_size, self.image_size):
            images_01 = F.interpolate(
                images_01,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        images_01 = images_01.to(self.device)
        clip_images = _normalize(images_01, CLIP_MEAN, CLIP_STD)
        scores: list[float] = []
        maps: list[torch.Tensor] = []

        # Official test.py uses batch_size=1 and FiLo.forward selects prompts from
        # the first batch element. Per-image inference preserves that behavior.
        for index, category in enumerate(normalized_categories):
            image_01 = images_01[index : index + 1]
            positions, boxes = self._grounding_context(image_01, category)
            items = {"img": clip_images[index : index + 1], "cls_name": [category]}
            with torch.inference_mode():
                text_probabilities, anomaly_maps = self.model(
                    items, with_adapter=True, positions=positions
                )
                layer_maps = [
                    _gaussian_blur_3x3_sigma4(layer_map)[:, 1]
                    for layer_map in anomaly_maps
                ]
                anomaly_map = torch.stack(layer_maps).mean(dim=0)[0]
                score = 0.5 * (
                    text_probabilities.reshape(-1)[1] + anomaly_map.max()
                )

                box_mask = torch.zeros_like(anomaly_map, dtype=torch.bool)
                for box in boxes:
                    left, top, right, bottom = (int(value.item()) for value in box)
                    box_mask[top:bottom, left:right] = True
                anomaly_map = torch.where(box_mask, anomaly_map, anomaly_map * 0.7)
            scores.append(float(score.item()))
            maps.append(anomaly_map)

        return (
            np.asarray(scores, dtype=np.float32),
            torch.stack(maps).cpu().numpy().astype(np.float32),
        )

    def release(self) -> None:
        del self.grounding_model
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
