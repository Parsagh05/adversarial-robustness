"""Built-in model adapters."""

from .base import ModelAdapter, available_adapters, build_adapter, register_adapter
from . import aaclip as _aaclip  # register built-in adapter
from . import afclip as _afclip  # register built-in adapter
from . import anomalyclip as _anomalyclip  # register built-in adapter
from . import aprilgan as _aprilgan  # register built-in adapter
from . import filo as _filo  # register built-in adapter

__all__ = ["ModelAdapter", "available_adapters", "build_adapter", "register_adapter"]
