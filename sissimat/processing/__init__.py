"""Reusable spectral-processing primitives."""

from .models import Spectrum
from .recipe import RecipeResult, apply_recipe, required_bindings, validate_recipe_document
from .transforms import (
    arpls_baseline,
    background_normalize,
    baseline_correct,
    crop,
    derivative,
    normalize,
    resample,
    savgol_smooth,
    to_absorbance,
    transmittance_warning,
)

__all__ = [
    "Spectrum",
    "RecipeResult",
    "apply_recipe",
    "arpls_baseline",
    "background_normalize",
    "baseline_correct",
    "crop",
    "derivative",
    "normalize",
    "resample",
    "required_bindings",
    "savgol_smooth",
    "to_absorbance",
    "transmittance_warning",
    "validate_recipe_document",
]
