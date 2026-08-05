"""Stable adapter boundary for adding target anomaly-detection models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import torch


class ModelAdapter(ABC):
    """A model receives RGB tensors in [0, 1] and owns all preprocessing."""

    model_name: str

    @abstractmethod
    def predict(self, images_01: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return one anomaly score and one low-resolution map per image."""

    def predict_with_categories(
        self, images_01: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict a batch with category context when a model needs class prompts."""

        if len(categories) != len(images_01):
            raise ValueError("Categories must contain one entry per image")
        return self.predict(images_01)

    def postprocess_image_scores(
        self,
        scores: np.ndarray,
        map_mins: np.ndarray,
        map_maxs: np.ndarray,
        categories: Sequence[str],
    ) -> np.ndarray:
        """Apply optional split/category-level image-score aggregation."""

        del map_mins, map_maxs, categories
        return scores

    @abstractmethod
    def release(self) -> None:
        """Release model resources."""


AdapterType = TypeVar("AdapterType", bound=type[ModelAdapter])
_REGISTRY: dict[str, type[ModelAdapter]] = {}


def register_adapter(name: str) -> Callable[[AdapterType], AdapterType]:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Adapter name cannot be empty")

    def decorator(cls: AdapterType) -> AdapterType:
        if normalized in _REGISTRY:
            raise ValueError(f"Model adapter already registered: {normalized}")
        _REGISTRY[normalized] = cls
        return cls

    return decorator


def available_adapters() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_adapter(name: str, **kwargs: object) -> ModelAdapter:
    normalized = name.strip().lower()
    if normalized not in _REGISTRY:
        raise ValueError(
            f"Unknown model adapter {name!r}; available adapters: {available_adapters()}"
        )
    return _REGISTRY[normalized](**kwargs)
