"""Diffulex model package that imports built-in models to trigger registration."""
from __future__ import annotations
import importlib
from pathlib import Path

# We define what is available, but we don't import it yet
_model_modules = [
    py_file.stem for py_file in Path(__file__).parent.glob("*.py")
    if py_file.stem not in {"auto_model", "__init__"}
]

def __getattr__(name):
    """Python 3.7+ magic method to load modules only when accessed."""
    if name in _model_modules:
        return importlib.import_module(f".{name}", __name__)
    if name == "AutoModelForDiffusionLM":
        from .auto_model import AutoModelForDiffusionLM
        return AutoModelForDiffusionLM
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = _model_modules + ["AutoModelForDiffusionLM"]