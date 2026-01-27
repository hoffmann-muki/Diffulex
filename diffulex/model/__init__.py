"""Diffulex model package that imports built-in models to trigger registration."""
from __future__ import annotations
import importlib
import sys
from pathlib import Path

# We define what is available, but we don't import it yet
_model_modules = [
    py_file.stem for py_file in Path(__file__).parent.glob("*.py")
    if py_file.stem not in {"auto_model", "__init__"}
]

# Force import of model modules to trigger their registration decorators
_import_errors = {}
for module_name in _model_modules:
    try:
        importlib.import_module(f".{module_name}", __name__)
    except Exception as e:
        _import_errors[module_name] = e
        print(
            f"Warning: Failed to import model module '{module_name}': {e}",
            file=sys.stderr,
            flush=True
        )

def __getattr__(name):
    """Python 3.7+ magic method to load modules only when accessed."""
    if name in _model_modules:
        if name in _import_errors:
            raise ImportError(f"Failed to import {name}: {_import_errors[name]}")
        return importlib.import_module(f".{name}", __name__)
    if name == "AutoModelForDiffusionLM":
        from .auto_model import AutoModelForDiffusionLM
        return AutoModelForDiffusionLM
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = _model_modules + ["AutoModelForDiffusionLM"]