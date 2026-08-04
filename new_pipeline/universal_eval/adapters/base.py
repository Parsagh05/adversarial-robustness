"""Stable adapter boundary for adding target anomaly-detection models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import torch


class ModelAdapter(ABC):
    """A model receives RGB tensors in [0, 1] and owns all preprocessing."""

    model_name: str

    @abstractmethod
    def predict(self, images_01: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return one anomaly score and one low-resolution map per image."""

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

