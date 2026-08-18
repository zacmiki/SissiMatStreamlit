"""Validation and execution of reusable spectral-processing recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .models import Spectrum
from .transforms import (
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


SUPPORTED_OPERATIONS = {
    "crop",
    "resample",
    "background_normalization",
    "transmittance_to_absorbance",
    "savitzky_golay",
    "derivative",
    "baseline_correction",
    "normalize",
}


@dataclass(frozen=True)
class RecipeResult:
    spectrum: Spectrum
    applied_steps: tuple[dict[str, Any], ...]
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def validate_recipe_document(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return normalized steps or raise a descriptive validation error."""

    if not isinstance(document, Mapping):
        raise ValueError("Recipe must be a JSON object.")
    schema_version = document.get("schema_version", 1)
    if schema_version != 1:
        raise ValueError(f"Unsupported recipe schema version: {schema_version}")
    raw_steps = document.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("Recipe must contain a 'steps' list.")

    steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"Recipe step {index} must be an object.")
        operation = raw_step.get("operation")
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"Recipe step {index} uses unsupported operation: {operation}")
        parameters = raw_step.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError(f"Parameters for recipe step {index} must be an object.")
        steps.append({"operation": operation, "parameters": dict(parameters)})
    return steps


def required_bindings(steps_or_document: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> set[str]:
    """Return external spectrum roles required to execute a recipe."""

    steps = (
        validate_recipe_document(steps_or_document)
        if isinstance(steps_or_document, Mapping)
        else list(steps_or_document)
    )
    bindings: set[str] = set()
    for step in steps:
        if step.get("operation") == "background_normalization":
            bindings.add(str(step.get("parameters", {}).get("binding", "background")))
    return bindings


def apply_recipe(
    spectrum: Spectrum,
    steps_or_document: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    bindings: Mapping[str, Spectrum] | None = None,
) -> RecipeResult:
    """Apply a validated recipe to one spectrum."""

    steps = (
        validate_recipe_document(steps_or_document)
        if isinstance(steps_or_document, Mapping)
        else validate_recipe_document({"schema_version": 1, "steps": list(steps_or_document)})
    )
    available_bindings = dict(bindings or {})
    current = spectrum.cleaned()
    artifacts: dict[str, Any] = {}
    warnings: list[str] = []

    for index, step in enumerate(steps, start=1):
        operation = step["operation"]
        parameters = dict(step["parameters"])
        try:
            if operation == "crop":
                current = crop(current, parameters["lower"], parameters["upper"])
            elif operation == "resample":
                current = resample(
                    current,
                    parameters["spacing"],
                    lower=parameters.get("lower"),
                    upper=parameters.get("upper"),
                )
            elif operation == "background_normalization":
                binding_name = str(parameters.get("binding", "background"))
                if binding_name not in available_bindings:
                    raise ValueError(f"Missing required spectrum binding: {binding_name}")
                current = background_normalize(current, available_bindings[binding_name])
            elif operation == "transmittance_to_absorbance":
                percent = bool(parameters.get("percent", False))
                warning = transmittance_warning(current, percent=percent)
                if warning is not None:
                    warnings.append(f"Recipe step {index} ({operation}): {warning}")
                current = to_absorbance(current, percent=percent)
            elif operation == "savitzky_golay":
                current = savgol_smooth(
                    current,
                    int(parameters["window"]),
                    int(parameters["polynomial_order"]),
                )
            elif operation == "derivative":
                current = derivative(current, int(parameters["order"]))
            elif operation == "baseline_correction":
                method = str(parameters.pop("method"))
                current, baseline = baseline_correct(current, method, **parameters)
                artifacts["baseline"] = np.asarray(baseline)
                artifacts["pre_baseline_spectrum"] = current.with_data(y=current.y + baseline)
            elif operation == "normalize":
                method = str(parameters.pop("method"))
                current = normalize(current, method, **parameters)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Recipe step {index} ({operation}) failed: {error}") from error

    return RecipeResult(
        spectrum=current,
        applied_steps=tuple(steps),
        artifacts=artifacts,
        warnings=tuple(warnings),
    )
