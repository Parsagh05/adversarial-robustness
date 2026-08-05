"""MVTec AD and VisA discovery with IDs matching the canonical manifest."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class EvaluationSample:
    dataset: str
    category: str
    defect_type: str
    image_path: Path
    mask_path: Path | None
    label: int
    split: str = "test"

    @property
    def protocol_id(self) -> str:
        prefix = "visa/" if self.dataset == "visa" else ""
        return (
            f"{self.split}/{prefix}{self.category}/{self.defect_type}/"
            f"{self.image_path.stem}"
        )


def _image_files(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return ()
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_mvtec(root: str | Path) -> list[EvaluationSample]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"MVTec root not found: {root_path}")
    samples: list[EvaluationSample] = []
    categories = sorted(
        path for path in root_path.iterdir() if (path / "test").is_dir()
    )
    for category_path in categories:
        for defect_path in sorted(path for path in (category_path / "test").iterdir() if path.is_dir()):
            is_normal = defect_path.name.lower() == "good"
            for image_path in _image_files(defect_path):
                mask_path = None
                if not is_normal:
                    mask_dir = category_path / "ground_truth" / defect_path.name
                    candidates = sorted(
                        path
                        for path in mask_dir.glob(f"{image_path.stem}_mask.*")
                        if path.suffix.lower() in IMAGE_EXTENSIONS
                    )
                    if not candidates:
                        raise FileNotFoundError(f"MVTec mask missing for {image_path}")
                    mask_path = candidates[0]
                samples.append(
                    EvaluationSample(
                        dataset="mvtec",
                        category=category_path.name,
                        defect_type=defect_path.name,
                        image_path=image_path,
                        mask_path=mask_path,
                        label=0 if is_normal else 1,
                    )
                )
    if not samples:
        raise RuntimeError(f"No MVTec test samples found under {root_path}")
    return samples


def discover_mvtec_train_normal(root: str | Path) -> list[EvaluationSample]:
    """Discover only MVTec's official ``train/good`` calibration images."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"MVTec root not found: {root_path}")
    samples: list[EvaluationSample] = []
    categories = sorted(
        path for path in root_path.iterdir() if (path / "train" / "good").is_dir()
    )
    for category_path in categories:
        for image_path in _image_files(category_path / "train" / "good"):
            samples.append(
                EvaluationSample(
                    dataset="mvtec",
                    category=category_path.name,
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


def _visa_manifest(root: Path) -> Path:
    for candidate in (
        root / "split_csv" / "1cls.csv",
        root / "split_csv" / "1cls.csv.csv",
        root / "1cls.csv",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"VisA split_csv/1cls.csv not found under {root}")


def _visa_path(root: Path, value: str) -> Path | None:
    value = value.strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else root / path


def _discover_visa_split(
    root: str | Path, split: str, *, normal_only: bool = False
) -> list[EvaluationSample]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"VisA root not found: {root_path}")
    manifest = _visa_manifest(root_path)
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [
            {str(k).strip().lower(): str(v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]
    required = {"object", "split", "label", "image"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"VisA manifest must contain {sorted(required)}: {manifest}")

    samples: list[EvaluationSample] = []
    for row in rows:
        if row["split"].lower() != split.lower():
            continue
        normal = row["label"].lower() in {"normal", "good", "0"}
        if normal_only and not normal:
            continue
        image_path = _visa_path(root_path, row["image"])
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(f"VisA image missing: {image_path}")
        mask_path = None if normal else _visa_path(root_path, row.get("mask", ""))
        if not normal and (mask_path is None or not mask_path.is_file()):
            raise FileNotFoundError(f"VisA mask missing: {mask_path}")
        samples.append(
            EvaluationSample(
                dataset="visa",
                category=row["object"],
                defect_type="normal" if normal else "anomaly",
                image_path=image_path,
                mask_path=mask_path,
                label=0 if normal else 1,
                split=split.lower(),
            )
        )
    samples.sort(key=lambda sample: sample.protocol_id)
    if not samples:
        label = "normal " if normal_only else ""
        raise RuntimeError(
            f"No VisA {label}{split} samples found under {root_path}"
        )
    return samples


def discover_visa(root: str | Path) -> list[EvaluationSample]:
    return _discover_visa_split(root, "test")


def discover_visa_train_normal(root: str | Path) -> list[EvaluationSample]:
    """Discover only normal images from VisA's official training split."""

    return _discover_visa_split(root, "train", normal_only=True)


def discover_dataset(
    name: str,
    *,
    mvtec_root: str | Path | None,
    visa_root: str | Path | None,
) -> list[EvaluationSample]:
    if name == "mvtec":
        if mvtec_root is None:
            raise ValueError("mvtec_root is required for MVTec evaluation")
        return discover_mvtec(mvtec_root)
    if name == "visa":
        if visa_root is None:
            raise ValueError("visa_root is required for VisA evaluation")
        return discover_visa(visa_root)
    raise ValueError(f"Unsupported target dataset: {name!r}")


def discover_calibration_dataset(
    name: str,
    *,
    mvtec_root: str | Path | None,
    visa_root: str | Path | None,
) -> list[EvaluationSample]:
    """Return normal training images without consulting test labels."""

    if name == "mvtec":
        if mvtec_root is None:
            raise ValueError("mvtec_root is required for MVTec calibration")
        return discover_mvtec_train_normal(mvtec_root)
    if name == "visa":
        if visa_root is None:
            raise ValueError("visa_root is required for VisA calibration")
        return discover_visa_train_normal(visa_root)
    raise ValueError(f"Unsupported calibration dataset: {name!r}")


def index_samples(samples: Iterable[EvaluationSample]) -> dict[str, EvaluationSample]:
    result: dict[str, EvaluationSample] = {}
    for sample in samples:
        if sample.protocol_id in result:
            raise ValueError(f"Duplicate dataset protocol ID: {sample.protocol_id}")
        result[sample.protocol_id] = sample
    return result


def load_image(sample: EvaluationSample, size: int) -> torch.Tensor:
    array = np.asarray(Image.open(sample.image_path).convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(array / 255.0).permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(size, size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return tensor[0].clamp_(0.0, 1.0)


def load_mask(sample: EvaluationSample, size: int) -> np.ndarray:
    if sample.mask_path is None:
        return np.zeros((size, size), dtype=np.uint8)
    mask = np.asarray(Image.open(sample.mask_path).convert("L"), dtype=np.uint8)
    tensor = torch.from_numpy((mask > 0).astype(np.float32))[None, None]
    return (
        F.interpolate(tensor, size=(size, size), mode="nearest")[0, 0]
        .numpy()
        .astype(np.uint8)
    )
