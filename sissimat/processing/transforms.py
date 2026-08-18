"""Numerical spectral-processing functions without Streamlit dependencies."""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

from .models import Spectrum


def crop(spectrum: Spectrum, lower: float, upper: float) -> Spectrum:
    """Restrict a spectrum to an inclusive coordinate interval."""

    spectrum = spectrum.cleaned()
    lower, upper = sorted((float(lower), float(upper)))
    mask = (spectrum.x >= lower) & (spectrum.x <= upper)
    if np.count_nonzero(mask) < 3:
        raise ValueError("The selected crop interval contains fewer than three points.")
    return spectrum.with_data(x=spectrum.x[mask], y=spectrum.y[mask])


def to_absorbance(spectrum: Spectrum, *, percent: bool = False) -> Spectrum:
    """Convert fractional or percentage transmittance to base-10 absorbance."""

    transmittance = spectrum.y / 100.0 if percent else spectrum.y
    if np.any(transmittance <= 0):
        raise ValueError("Transmittance must be greater than zero for absorbance conversion.")
    return spectrum.with_data(y=-np.log10(transmittance), y_label="Absorbance")


def transmittance_warning(spectrum: Spectrum, *, percent: bool = False) -> str | None:
    """Describe transmittance above unity without preventing absorbance conversion."""

    transmittance = spectrum.y / 100.0 if percent else spectrum.y
    above_unity = transmittance > 1.0 + 1e-9
    if not np.any(above_unity):
        return None

    count = int(np.count_nonzero(above_unity))
    percentage = count / transmittance.size * 100.0
    maximum = float(np.max(transmittance))
    minimum_absorbance = float(-np.log10(maximum))
    severity = (
        "Large transmittance overshoot"
        if maximum > 1.2 or percentage > 50.0
        else "Transmittance exceeds 1"
    )
    return (
        f"{severity} for {count} of {transmittance.size} points ({percentage:.1f}%); "
        f"maximum T = {maximum:.4g}, giving a minimum absorbance of "
        f"{minimum_absorbance:.4g} AU. Conversion continued; check the background, "
        "measurement drift, alignment, and the fractional/percentage setting."
    )


def resample(
    spectrum: Spectrum,
    spacing: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> Spectrum:
    """Linearly interpolate a spectrum onto an evenly spaced grid."""

    spectrum = spectrum.cleaned()
    spacing = float(spacing)
    if spacing <= 0:
        raise ValueError("Resampling spacing must be greater than zero.")
    start = spectrum.x[0] if lower is None else max(float(lower), spectrum.x[0])
    stop = spectrum.x[-1] if upper is None else min(float(upper), spectrum.x[-1])
    if start >= stop:
        raise ValueError("The resampling interval does not overlap the spectrum.")
    new_x = np.arange(start, stop + spacing * 0.5, spacing)
    if new_x.size < 3:
        raise ValueError("The requested spacing produces fewer than three points.")
    new_y = np.interp(new_x, spectrum.x, spectrum.y)
    return spectrum.with_data(x=new_x, y=new_y)


def savgol_smooth(spectrum: Spectrum, window: int, order: int) -> Spectrum:
    """Apply Savitzky–Golay smoothing."""

    window = int(window)
    order = int(order)
    if window < 3 or window % 2 == 0:
        raise ValueError("Savitzky–Golay window length must be an odd integer of at least 3.")
    if window > spectrum.y.size:
        raise ValueError("Savitzky–Golay window cannot exceed the spectrum length.")
    if order < 0 or order >= window:
        raise ValueError("Polynomial order must be non-negative and smaller than the window.")
    return spectrum.with_data(y=savgol_filter(spectrum.y, window, order))


def derivative(spectrum: Spectrum, order: int = 1) -> Spectrum:
    """Calculate a first or second numerical derivative on a nonuniform grid."""

    if order not in (1, 2):
        raise ValueError("Derivative order must be 1 or 2.")
    spectrum = spectrum.cleaned()
    values = spectrum.y.copy()
    for _ in range(order):
        values = np.gradient(values, spectrum.x, edge_order=2)
    return spectrum.with_data(
        y=values,
        y_label=f"{spectrum.y_label} derivative {order}",
    )


def normalize(
    spectrum: Spectrum,
    method: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> Spectrum:
    """Normalize by min–max range, vector norm, or absolute integrated area."""

    method = method.lower().replace("–", "-")
    x = spectrum.x
    y = spectrum.y
    if lower is not None or upper is not None:
        low = x.min() if lower is None else float(lower)
        high = x.max() if upper is None else float(upper)
        low, high = sorted((low, high))
        mask = (x >= low) & (x <= high)
        if np.count_nonzero(mask) < 2:
            raise ValueError("The normalization interval contains fewer than two points.")
    else:
        mask = np.ones(x.size, dtype=bool)

    region_x = x[mask]
    region_y = y[mask]
    if method == "min-max":
        minimum = np.min(region_y)
        scale = np.max(region_y) - minimum
        if np.isclose(scale, 0):
            raise ValueError("Cannot min–max normalize a constant spectrum.")
        result = (y - minimum) / scale
    elif method == "vector":
        scale = norm(region_y)
        if np.isclose(scale, 0):
            raise ValueError("Cannot vector-normalize a zero spectrum.")
        result = y / scale
    elif method == "area":
        scale = abs(trapezoid(region_y, region_x))
        if np.isclose(scale, 0):
            raise ValueError("Cannot area-normalize a spectrum with zero integrated area.")
        result = y / scale
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return spectrum.with_data(y=result, y_label=f"Normalized {spectrum.y_label}")


def polynomial_baseline(spectrum: Spectrum, degree: int = 2, edge_fraction: float = 0.1) -> np.ndarray:
    """Fit a polynomial baseline using points at both edges of the spectrum."""

    spectrum = spectrum.cleaned()
    degree = int(degree)
    edge_fraction = float(edge_fraction)
    if degree < 0 or degree > 8:
        raise ValueError("Polynomial baseline degree must be between 0 and 8.")
    if not 0 < edge_fraction <= 0.5:
        raise ValueError("Edge fraction must be between 0 and 0.5.")
    edge_count = max(degree + 1, int(np.ceil(spectrum.x.size * edge_fraction)))
    if edge_count * 2 > spectrum.x.size:
        edge_count = spectrum.x.size // 2
    indices = np.r_[0:edge_count, spectrum.x.size - edge_count : spectrum.x.size]
    coefficients = np.polyfit(spectrum.x[indices], spectrum.y[indices], degree)
    return np.polyval(coefficients, spectrum.x)


def rubberband_baseline(spectrum: Spectrum) -> np.ndarray:
    """Estimate a baseline by interpolating the spectrum's lower convex hull."""

    spectrum = spectrum.cleaned()
    hull: list[int] = []
    for index in range(spectrum.x.size):
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            cross = (
                (spectrum.x[second] - spectrum.x[first])
                * (spectrum.y[index] - spectrum.y[first])
                - (spectrum.y[second] - spectrum.y[first])
                * (spectrum.x[index] - spectrum.x[first])
            )
            if cross > 0:
                break
            hull.pop()
        hull.append(index)
    return np.interp(spectrum.x, spectrum.x[hull], spectrum.y[hull])


def arpls_baseline(
    spectrum: Spectrum,
    *,
    lam: float = 1e5,
    ratio: float = 1e-6,
    max_iterations: int = 30,
) -> np.ndarray:
    """Estimate a baseline using adaptive robust penalized least squares."""

    y = spectrum.y
    length = y.size
    if length < 3:
        raise ValueError("arPLS requires at least three points.")
    if lam <= 0 or ratio <= 0 or max_iterations < 1:
        raise ValueError("Invalid arPLS parameters.")

    diagonal = np.ones(length - 2)
    difference = sparse.spdiags(
        [diagonal, -2 * diagonal, diagonal], [0, -1, -2], length, length - 2
    )
    penalty = lam * difference.dot(difference.T)
    weights = np.ones(length)
    baseline = y.copy()

    for _ in range(int(max_iterations)):
        weight_matrix = sparse.spdiags(weights, 0, length, length)
        baseline = spsolve(weight_matrix + penalty, weights * y)
        residual = y - baseline
        negative = residual[residual < 0]
        if negative.size == 0:
            break
        mean = negative.mean()
        std = negative.std()
        if np.isclose(std, 0):
            break
        exponent = np.clip(2 * (residual - (2 * std - mean)) / std, -700, 700)
        new_weights = 1.0 / (1.0 + np.exp(exponent))
        criterion = norm(new_weights - weights) / max(norm(weights), np.finfo(float).eps)
        weights = new_weights
        if criterion <= ratio:
            break
    return np.asarray(baseline)


def baseline_correct(spectrum: Spectrum, method: str, **parameters: float) -> tuple[Spectrum, np.ndarray]:
    """Estimate and subtract a selected baseline."""

    spectrum = spectrum.cleaned()
    method = method.lower().replace("–", "-")
    if method == "arpls":
        baseline = arpls_baseline(spectrum, **parameters)
    elif method == "polynomial":
        baseline = polynomial_baseline(spectrum, **parameters)
    elif method == "rubber-band":
        baseline = rubberband_baseline(spectrum)
    else:
        raise ValueError(f"Unknown baseline method: {method}")
    return spectrum.with_data(y=spectrum.y - baseline), baseline


def background_normalize(spectrum: Spectrum, background: Spectrum) -> Spectrum:
    """Normalize a spectrum by an interpolated background as I / I₀."""

    spectrum = spectrum.cleaned()
    background = background.cleaned()
    tolerance = np.finfo(float).eps * max(abs(spectrum.x[0]), abs(spectrum.x[-1]), 1.0) * 10
    if background.x[0] > spectrum.x[0] + tolerance or background.x[-1] < spectrum.x[-1] - tolerance:
        raise ValueError("Background spectrum does not cover the complete processed range.")
    background_y = np.interp(spectrum.x, background.x, background.y)
    zero_tolerance = np.finfo(float).eps * max(float(np.max(np.abs(background_y))), 1.0) * 100
    if np.any(np.abs(background_y) <= zero_tolerance):
        raise ValueError("Background contains zero values; I / I₀ cannot be calculated.")
    return spectrum.with_data(
        y=spectrum.y / background_y,
        y_label="Background-normalized intensity (I / I₀)",
    )
