"""Data models shared by the processing and user-interface layers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Spectrum:
    """One-dimensional spectrum with lightweight provenance metadata."""

    x: np.ndarray
    y: np.ndarray
    name: str = "Spectrum"
    x_unit: str = "cm⁻¹"
    y_label: str = "Intensity"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Spectrum coordinates and values must be one-dimensional.")
        if x.size != y.size:
            raise ValueError("Spectrum coordinates and values must have the same length.")
        if x.size < 3:
            raise ValueError("A spectrum must contain at least three points.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Spectrum contains NaN or infinite values.")
        object.__setattr__(self, "x", x.copy())
        object.__setattr__(self, "y", y.copy())
        object.__setattr__(self, "metadata", dict(self.metadata))

    def cleaned(self) -> "Spectrum":
        """Return an ascending spectrum with duplicate coordinates averaged."""

        order = np.argsort(self.x, kind="stable")
        x_sorted = self.x[order]
        y_sorted = self.y[order]
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        if unique_x.size == x_sorted.size:
            return replace(self, x=x_sorted, y=y_sorted)

        sums = np.zeros(unique_x.size, dtype=float)
        counts = np.zeros(unique_x.size, dtype=float)
        np.add.at(sums, inverse, y_sorted)
        np.add.at(counts, inverse, 1.0)
        return replace(self, x=unique_x, y=sums / counts)

    def with_data(
        self,
        *,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        name: str | None = None,
        y_label: str | None = None,
    ) -> "Spectrum":
        """Return a copy while preserving source metadata."""

        return replace(
            self,
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            name=self.name if name is None else name,
            y_label=self.y_label if y_label is None else y_label,
        )
