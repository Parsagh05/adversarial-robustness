"""Built-in model adapters."""

from .base import ModelAdapter, available_adapters, build_adapter, register_adapter
from . import anomalyclip as _anomalyclip  # register built-in adapter

__all__ = ["ModelAdapter", "available_adapters", "build_adapter", "register_adapter"]

