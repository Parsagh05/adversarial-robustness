"""MVTec AD discovery and resolution-aware tensor loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MVTecSample:
    index: int
    category: str
    defect_type: str
    image_path: Path
    mask_path: Optional[Path]
    label: int
    split: str = "test"

    @property
    def sample_id(self) -> str:
        return f"{self.category}/{self.defect_type}/{self.image_path.stem}"

    @property
    def protocol_id(self) -> str:
        """Split-qualified identifier used by held-out protocol manifests."""

        return f"{self.split}/{self.sample_id}"


def _image_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_mvtec(
    root: str,
    categories: Optional[Sequence[str]] = None,
    max_samples_per_category: Optional[int] = None,
) -> List[MVTecSample]:
    """Discover the official MVTec ``test`` split.

    ``test/good`` samples receive an all-zero mask. Defective samples are
    matched to ``ground_truth/<defect>/<stem>_mask.<ext>``.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"MVTec root does not exist: {root_path}")

    available = sorted(
        path.name for path in root_path.iterdir()
        if path.is_dir() and (path / "test").is_dir()
    )
    selected = list(categories) if categories is not None else available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"MVTec categories not found under {root_path}: {missing}")

    provisional: List[MVTecSample] = []
    for category in selected:
        category_samples: List[MVTecSample] = []
        test_root = root_path / category / "test"
        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            is_good = defect_dir.name.lower() == "good"
            for image_path in _image_files(defect_dir):
                mask_path = None
                if not is_good:
                    gt_dir = root_path / category / "ground_truth" / defect_dir.name
                    candidates = sorted(
                        path for path in gt_dir.glob(f"{image_path.stem}_mask.*")
                        if path.suffix.lower() in IMAGE_EXTENSIONS
                    )
                    if not candidates:
                        raise FileNotFoundError(
                            f"Ground-truth mask missing for {image_path} under {gt_dir}"
                        )
                    mask_path = candidates[0]
                category_samples.append(
                    MVTecSample(
                        index=-1,
                        category=category,
                        defect_type=defect_dir.name,
                        image_path=image_path,
                        mask_path=mask_path,
                        label=0 if is_good else 1,
                        split="test",
                    )
                )
        if max_samples_per_category is not None:
            # Deterministic, label-aware cap so smoke runs retain both classes.
            good = [sample for sample in category_samples if sample.label == 0]
            bad = [sample for sample in category_samples if sample.label == 1]
            half = max(1, max_samples_per_category // 2)
            category_samples = (good[:half] + bad[: max_samples_per_category - half])
        provisional.extend(category_samples)

    samples = [
        MVTecSample(
            index=index,
            category=sample.category,
            defect_type=sample.defect_type,
            image_path=sample.image_path,
            mask_path=sample.mask_path,
            label=sample.label,
            split=sample.split,
        )
        for index, sample in enumerate(provisional)
    ]
    if not samples:
        raise RuntimeError(f"No MVTec test samples found under {root_path}")
    return samples


def discover_mvtec_train_good(
    root: str,
    categories: Optional[Sequence[str]] = None,
) -> List[MVTecSample]:
    """Discover normal MVTec ``train/good`` images for universal fitting.

    These samples are deliberately kept separate from the indexed test set.
    They never participate in evaluation metrics or target prediction arrays.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"MVTec root does not exist: {root_path}")
    available = sorted(
        path.name
        for path in root_path.iterdir()
        if path.is_dir() and (path / "train" / "good").is_dir()
    )
    selected = list(categories) if categories is not None else available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(
            f"MVTec train/good categories not found under {root_path}: {missing}"
        )

    samples: List[MVTecSample] = []
    for category in selected:
        for image_path in _image_files(root_path / category / "train" / "good"):
            samples.append(
                MVTecSample(
                    index=-1,
                    category=category,
                    defect_type="good",
                    image_path=image_path,
                    mask_path=None,
                    label=0,
                    split="train",
                )
            )
    if not samples:
        raise RuntimeError(f"No MVTec train/good samples found under {root_path}")
    return samples


def group_by_category(samples: Sequence[MVTecSample]) -> Dict[str, List[MVTecSample]]:
    grouped: Dict[str, List[MVTecSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.category, []).append(sample)
    return grouped


def load_image_tensor(sample: MVTecSample, image_size: int) -> torch.Tensor:
    """Load RGB image as ``[3, H, W]`` float tensor in ``[0, 1]``."""

    image = Image.open(sample.image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return tensor.squeeze(0).clamp(0.0, 1.0)


def load_mask(sample: MVTecSample, image_size: int) -> np.ndarray:
    """Load a binary mask at the benchmark metric resolution."""

    if sample.mask_path is None:
        return np.zeros((image_size, image_size), dtype=np.uint8)
    mask = np.asarray(Image.open(sample.mask_path).convert("L"), dtype=np.uint8)
    tensor = torch.from_numpy((mask > 0).astype(np.float32))[None, None]
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return tensor[0, 0].numpy().astype(np.uint8)
