"""Interactive Spectral Processing Lab page."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sissimat.processing.io import load_spectra, preferred_dataset_label
from sissimat.processing.models import Spectrum
from sissimat.processing.recipe import required_bindings, validate_recipe_document
from sissimat.processing.transforms import (
    baseline_correct,
    background_normalize,
    crop,
    derivative,
    normalize,
    resample,
    savgol_smooth,
    to_absorbance,
    transmittance_warning,
)


@st.cache_data(show_spinner=False)
def _cached_load(file_bytes: bytes, filename: str) -> dict[str, Spectrum]:
    return load_spectra(file_bytes, filename)


def _load_uploaded(uploaded_file, key_prefix: str) -> Spectrum | None:
    if uploaded_file is None:
        return None
    try:
        spectra = _cached_load(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as error:
        st.error(f"Could not load {uploaded_file.name}: {error}")
        return None

    labels = list(spectra)
    selected = preferred_dataset_label(labels)
    if len(labels) > 1:
        selected = st.selectbox(
            "OPUS dataset",
            labels,
            index=labels.index(selected),
            key=f"{key_prefix}_dataset",
            help="SSC is selected by default because it is the computed Fourier-transform spectrum.",
        )
    return spectra[selected].cleaned()


def _spectrum_plot(
    original: Spectrum,
    processed: Spectrum,
    reverse_axis: bool,
    show_original: bool,
) -> go.Figure:
    figure = go.Figure()
    if show_original:
        figure.add_trace(
            go.Scatter(
                x=original.x,
                y=original.y,
                mode="lines",
                name="Original",
                line={"color": "rgba(160,160,160,0.75)", "width": 1.5},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=processed.x,
            y=processed.y,
            mode="lines",
            name="Processed",
            line={"color": "#ff4b4b", "width": 2.2},
        )
    )
    figure.update_layout(
        height=560,
        margin={"l": 40, "r": 20, "t": 35, "b": 40},
        hovermode="x unified",
        xaxis_title=f"Wavenumber ({processed.x_unit})",
        yaxis_title=processed.y_label,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    figure.update_xaxes(autorange="reversed" if reverse_axis else True)
    return figure


def _baseline_plot(before: Spectrum, baseline: np.ndarray, corrected: Spectrum, reverse_axis: bool) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=before.x, y=before.y, mode="lines", name="Before baseline"))
    figure.add_trace(
        go.Scatter(
            x=before.x,
            y=baseline,
            mode="lines",
            name="Estimated baseline",
            line={"color": "#ffa421", "dash": "dash"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=corrected.x,
            y=corrected.y,
            mode="lines",
            name="Corrected",
            line={"color": "#00c98d"},
        )
    )
    figure.update_layout(height=430, hovermode="x unified", xaxis_title=f"Wavenumber ({before.x_unit})")
    figure.update_xaxes(autorange="reversed" if reverse_axis else True)
    return figure


def _recipe_step(operation: str, **parameters) -> dict:
    return {"operation": operation, "parameters": parameters}


def _recipe_table(steps: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Step": range(1, len(steps) + 1),
            "Operation": [step["operation"] for step in steps],
            "Parameters": [json.dumps(step["parameters"], ensure_ascii=False) for step in steps],
        }
    )


EDITABLE_RECIPE_ORDER = [
    "crop",
    "resample",
    "background_normalization",
    "transmittance_to_absorbance",
    "savitzky_golay",
    "derivative",
    "baseline_correction",
    "normalize",
]


def _empty_recipe_control_state(source: Spectrum) -> dict[str, object]:
    """Return the widget state for an empty recipe on this spectrum."""

    minimum = float(source.x.min())
    maximum = float(source.x.max())
    median_spacing = float(np.median(np.diff(source.x)))
    max_window = source.x.size if source.x.size % 2 else source.x.size - 1
    default_window = min(11, max_window)
    identity = (source.name, source.x.size, minimum, maximum)
    return {
        "lab_enable_crop": False,
        "lab_crop_source_identity": identity,
        "lab_crop_bounds": (minimum, maximum),
        "lab_crop_lower": minimum,
        "lab_crop_upper": maximum,
        "lab_crop_slider": (minimum, maximum),
        "lab_enable_resample": False,
        "lab_resample_spacing": median_spacing,
        "lab_enable_background": False,
        "lab_representation": "Keep current values",
        "lab_enable_smoothing": False,
        "lab_smoothing_window": default_window,
        "lab_smoothing_order": min(3, default_window - 1),
        "lab_derivative": "None",
        "lab_baseline_method": "None",
        "lab_arpls_lam": 1e5,
        "lab_arpls_ratio": 1e-6,
        "lab_arpls_iterations": 30,
        "lab_polynomial_degree": 2,
        "lab_polynomial_edges": 0.10,
        "lab_normalization_method": "None",
        "lab_normalization_range": (minimum, maximum),
    }


def _editable_recipe_control_state(
    steps: list[dict], source: Spectrum
) -> tuple[dict[str, object], list[str]]:
    """Translate a canonical saved recipe into existing Processing Lab widgets."""

    operations = [step["operation"] for step in steps]
    if len(operations) != len(set(operations)):
        raise ValueError(
            "The full interface cannot edit recipes containing the same operation more than once."
        )
    positions = [EDITABLE_RECIPE_ORDER.index(operation) for operation in operations]
    if positions != sorted(positions):
        raise ValueError(
            "This recipe uses a custom step order that the current full interface cannot preserve."
        )

    state = _empty_recipe_control_state(source)
    warnings: list[str] = []
    source_minimum = float(source.x.min())
    source_maximum = float(source.x.max())
    median_spacing = float(np.median(np.diff(source.x)))
    max_window = source.x.size if source.x.size % 2 else source.x.size - 1

    for step in steps:
        operation = step["operation"]
        parameters = dict(step["parameters"])
        if operation == "crop":
            requested_lower, requested_upper = sorted(
                (float(parameters["lower"]), float(parameters["upper"]))
            )
            lower = max(source_minimum, requested_lower)
            upper = min(source_maximum, requested_upper)
            if lower >= upper:
                raise ValueError("The recipe crop range does not overlap this spectrum.")
            if (lower, upper) != (requested_lower, requested_upper):
                warnings.append("The crop range was clipped to the loaded spectrum limits.")
            state.update(
                {
                    "lab_enable_crop": True,
                    "lab_crop_lower": lower,
                    "lab_crop_upper": upper,
                    "lab_crop_slider": (lower, upper),
                }
            )
        elif operation == "resample":
            if parameters.get("lower") is not None or parameters.get("upper") is not None:
                raise ValueError(
                    "This recipe uses resampling limits that are not exposed by the full interface."
                )
            spacing = float(parameters["spacing"])
            minimum_spacing = max(
                median_spacing / 1000.0, np.finfo(float).eps
            )
            if spacing < minimum_spacing:
                raise ValueError(
                    "Recipe grid spacing is below the minimum exposed by the full interface."
                )
            state["lab_enable_resample"] = True
            state["lab_resample_spacing"] = spacing
        elif operation == "background_normalization":
            if str(parameters.get("binding", "background")) != "background":
                raise ValueError(
                    "The full interface supports only the standard 'background' binding."
                )
            state["lab_enable_background"] = True
        elif operation == "transmittance_to_absorbance":
            state["lab_representation"] = (
                "Percentage transmittance → absorbance"
                if bool(parameters.get("percent", False))
                else "Fractional transmittance or I / I₀ → absorbance"
            )
        elif operation == "savitzky_golay":
            window = int(parameters["window"])
            polynomial_order = int(parameters["polynomial_order"])
            if window < 3 or window > max_window or window % 2 == 0:
                raise ValueError(
                    f"Savitzky–Golay window must be an odd value from 3 to {max_window}."
                )
            if polynomial_order < 1 or polynomial_order >= window:
                raise ValueError("Invalid Savitzky–Golay polynomial order.")
            state["lab_enable_smoothing"] = True
            state["lab_smoothing_window"] = window
            state["lab_smoothing_order"] = polynomial_order
        elif operation == "derivative":
            order = int(parameters["order"])
            if order not in (1, 2):
                raise ValueError("Derivative order must be 1 or 2.")
            state["lab_derivative"] = (
                "First derivative" if order == 1 else "Second derivative"
            )
        elif operation == "baseline_correction":
            method = str(parameters.pop("method")).lower().replace("–", "-")
            method_labels = {
                "arpls": "arPLS",
                "polynomial": "Polynomial",
                "rubber-band": "Rubber-band",
            }
            if method not in method_labels:
                raise ValueError(f"Unsupported baseline method: {method}")
            state["lab_baseline_method"] = method_labels[method]
            if method == "arpls":
                lam = float(parameters.get("lam", 1e5))
                ratio = float(parameters.get("ratio", 1e-6))
                iterations = int(parameters.get("max_iterations", 30))
                if lam < 1.0 or ratio < 1e-12 or not 1 <= iterations <= 200:
                    raise ValueError(
                        "The arPLS parameters are outside the limits of the full interface."
                    )
                state["lab_arpls_lam"] = lam
                state["lab_arpls_ratio"] = ratio
                state["lab_arpls_iterations"] = iterations
            elif method == "polynomial":
                degree = int(parameters.get("degree", 2))
                edge_fraction = float(parameters.get("edge_fraction", 0.10))
                if not 0 <= degree <= 8 or not 0.02 <= edge_fraction <= 0.50:
                    raise ValueError(
                        "The polynomial baseline parameters are outside the limits "
                        "of the full interface."
                    )
                state["lab_polynomial_degree"] = degree
                state["lab_polynomial_edges"] = edge_fraction
        elif operation == "normalize":
            method = str(parameters.get("method", "")).lower().replace("–", "-")
            method_labels = {
                "min-max": "Min–max",
                "vector": "Vector",
                "area": "Area",
            }
            if method not in method_labels:
                raise ValueError(f"Unsupported normalization method: {method}")
            state["lab_normalization_method"] = method_labels[method]
            interval_minimum = (
                float(state["lab_crop_lower"])
                if state["lab_enable_crop"]
                else source_minimum
            )
            interval_maximum = (
                float(state["lab_crop_upper"])
                if state["lab_enable_crop"]
                else source_maximum
            )
            raw_lower = parameters.get("lower")
            raw_upper = parameters.get("upper")
            requested_lower, requested_upper = sorted(
                (
                    interval_minimum if raw_lower is None else float(raw_lower),
                    interval_maximum if raw_upper is None else float(raw_upper),
                )
            )
            lower = max(interval_minimum, requested_lower)
            upper = min(interval_maximum, requested_upper)
            if lower >= upper:
                raise ValueError(
                    "The recipe normalization interval does not overlap the processed range."
                )
            if (lower, upper) != (requested_lower, requested_upper):
                warnings.append(
                    "The normalization interval was clipped to the processed range."
                )
            state["lab_normalization_range"] = (lower, upper)

    return state, warnings


def _set_recipe_control_state(state: dict[str, object]) -> None:
    for key, value in state.items():
        st.session_state[key] = value


def _widget_default(key: str, value: object) -> dict[str, object]:
    """Supply a widget default only when session state has not pre-filled it."""

    return {} if key in st.session_state else {"value": value}


def _sync_crop_numbers_from_slider() -> None:
    lower, upper = st.session_state.lab_crop_slider
    st.session_state.lab_crop_lower = lower
    st.session_state.lab_crop_upper = upper


def _enable_background_from_upload() -> None:
    if st.session_state.get("processing_background_file") is not None:
        st.session_state.lab_enable_background = True


def _sync_crop_slider_from_numbers() -> None:
    minimum, maximum = st.session_state.lab_crop_bounds
    lower = min(max(float(st.session_state.lab_crop_lower), minimum), maximum)
    upper = min(max(float(st.session_state.lab_crop_upper), minimum), maximum)
    lower, upper = sorted((lower, upper))
    st.session_state.lab_crop_lower = lower
    st.session_state.lab_crop_upper = upper
    st.session_state.lab_crop_slider = (lower, upper)


def _initialize_crop_controls(source: Spectrum) -> None:
    minimum = float(source.x.min())
    maximum = float(source.x.max())
    identity = (source.name, source.x.size, minimum, maximum)
    if st.session_state.get("lab_crop_source_identity") != identity:
        st.session_state.lab_crop_source_identity = identity
        st.session_state.lab_crop_bounds = (minimum, maximum)
        st.session_state.lab_crop_lower = minimum
        st.session_state.lab_crop_upper = maximum
        st.session_state.lab_crop_slider = (minimum, maximum)


def render_spectral_processing_lab() -> None:
    st.title("🔬 Spectral Processing Lab")
    st.caption(
        "Build a reproducible processing recipe, compare it with the original spectrum, "
        "and export both the processed data and its parameters."
    )

    processing_mode = st.radio(
        "Processing mode",
        ["Single spectrum", "Batch/series"],
        horizontal=True,
        key="lab_processing_mode",
    )
    if processing_mode == "Batch/series":
        from sissimat.pages.batch_processing import render_batch_processing

        render_batch_processing()
        return

    uploaded = st.file_uploader(
        "Load an OPUS, CSV, TXT or DAT spectrum",
        key="processing_primary_file",
        help="For text files, the first two numeric columns are interpreted as x and y.",
    )
    source = _load_uploaded(uploaded, "processing_primary")
    if source is None:
        st.info("Load a spectrum to configure the processing recipe.")
        return

    metric_columns = st.columns(3)
    metric_columns[0].metric("Points", f"{source.x.size:,}")
    metric_columns[1].metric("Minimum", f"{source.x.min():.3f} {source.x_unit}")
    metric_columns[2].metric("Maximum", f"{source.x.max():.3f} {source.x_unit}")

    with st.expander("Load a saved recipe as an editable starting point"):
        imported_recipe = st.file_uploader(
            "Recipe JSON",
            type=["json"],
            key="single_imported_recipe",
        )
    imported_steps: list[dict] | None = None
    if imported_recipe is None:
        st.session_state.pop("lab_imported_recipe_identity", None)
    if imported_recipe is not None:
        try:
            recipe_bytes = imported_recipe.getvalue()
            recipe_document = json.loads(recipe_bytes.decode("utf-8"))
            imported_steps = validate_recipe_document(recipe_document)
            recipe_identity = (
                hashlib.sha256(recipe_bytes).hexdigest(),
                source.name,
                source.x.size,
                float(source.x.min()),
                float(source.x.max()),
            )
            if st.session_state.get("lab_imported_recipe_identity") != recipe_identity:
                control_state, import_warnings = _editable_recipe_control_state(
                    imported_steps, source
                )
                _set_recipe_control_state(control_state)
                st.session_state.lab_imported_recipe_identity = recipe_identity
                st.session_state.lab_imported_recipe_name = imported_recipe.name
                st.session_state.lab_imported_recipe_steps = imported_steps
                st.session_state.lab_import_warnings = import_warnings
                st.rerun()
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as error:
            imported_steps = None
            st.error(f"Recipe cannot be loaded into the editable interface: {error}")

    if imported_steps is not None:
        st.success(
            f"Loaded {imported_recipe.name}. The controls below are now editable."
        )
        action_columns = st.columns(2)
        if action_columns[0].button(
            "Reset to imported recipe",
            use_container_width=True,
            key="lab_reset_imported_recipe",
        ):
            control_state, import_warnings = _editable_recipe_control_state(
                imported_steps, source
            )
            _set_recipe_control_state(control_state)
            st.session_state.lab_import_warnings = import_warnings
            st.rerun()
        if action_columns[1].button(
            "Clear all processing steps",
            use_container_width=True,
            key="lab_clear_imported_recipe",
        ):
            _set_recipe_control_state(_empty_recipe_control_state(source))
            st.rerun()
        for warning in st.session_state.get("lab_import_warnings", []):
            st.warning(warning)
        if (
            "background" in required_bindings(imported_steps)
            and st.session_state.get("lab_enable_background", False)
        ):
            st.info(
                "This recipe requires background normalization. Select the actual "
                "background spectrum in section 2 before using or exporting it."
            )

    st.markdown("### Processing recipe")
    st.caption("Operations are applied from top to bottom in the order shown below.")

    with st.expander("1 · Range and grid", expanded=True):
        enable_crop = st.checkbox("Crop spectral range", key="lab_enable_crop")
        crop_range = (float(source.x.min()), float(source.x.max()))
        if enable_crop:
            _initialize_crop_controls(source)
            st.slider(
                "Range",
                min_value=float(source.x.min()),
                max_value=float(source.x.max()),
                key="lab_crop_slider",
                on_change=_sync_crop_numbers_from_slider,
            )
            range_columns = st.columns(2)
            number_step = max(float(np.median(np.diff(source.x))), np.finfo(float).eps)
            range_columns[0].number_input(
                f"Lower limit ({source.x_unit})",
                min_value=float(source.x.min()),
                max_value=float(source.x.max()),
                step=number_step,
                format="%.6f",
                key="lab_crop_lower",
                on_change=_sync_crop_slider_from_numbers,
            )
            range_columns[1].number_input(
                f"Upper limit ({source.x_unit})",
                min_value=float(source.x.min()),
                max_value=float(source.x.max()),
                step=number_step,
                format="%.6f",
                key="lab_crop_upper",
                on_change=_sync_crop_slider_from_numbers,
            )
            crop_range = (
                float(st.session_state.lab_crop_lower),
                float(st.session_state.lab_crop_upper),
            )
        median_spacing = float(np.median(np.diff(source.x)))
        enable_resample = st.checkbox("Resample onto an evenly spaced grid", key="lab_enable_resample")
        spacing = median_spacing
        if enable_resample:
            spacing = st.number_input(
                f"Grid spacing ({source.x_unit})",
                min_value=max(median_spacing / 1000.0, np.finfo(float).eps),
                format="%.6f",
                key="lab_resample_spacing",
                **_widget_default("lab_resample_spacing", median_spacing),
            )

    with st.expander(
        "2 · Background normalization (I / I₀)",
        expanded=imported_steps is not None,
    ):
        enable_background = st.checkbox(
            "Apply background normalization",
            key="lab_enable_background",
            help="Uncheck this to remove background normalization from the recipe.",
        )
        background_upload = st.file_uploader(
            "Optional background spectrum (I₀)",
            key="processing_background_file",
            help="The sample intensity I is divided point-by-point by the interpolated background I₀.",
            on_change=_enable_background_from_upload,
        )
        loaded_background = _load_uploaded(background_upload, "processing_background")
        background = loaded_background if enable_background else None
        if enable_background and loaded_background is None:
            st.info("Select the background spectrum required for I / I₀ normalization.")

    max_window = source.x.size if source.x.size % 2 else source.x.size - 1
    default_window = min(11, max_window)
    with st.expander(
        "3 · Representation, smoothing and derivatives",
        expanded=imported_steps is not None,
    ):
        representation = st.selectbox(
            "Convert current spectrum",
            [
                "Keep current values",
                "Fractional transmittance or I / I₀ → absorbance",
                "Percentage transmittance → absorbance",
            ],
            key="lab_representation",
            help="Absorbance conversion uses A = −log₁₀(T). Background normalization is applied first.",
        )
        enable_smoothing = st.checkbox("Savitzky–Golay smoothing", key="lab_enable_smoothing")
        smoothing_window = default_window
        smoothing_order = min(3, default_window - 1)
        if enable_smoothing:
            smooth_columns = st.columns(2)
            smoothing_window = smooth_columns[0].number_input(
                "Odd window length",
                min_value=3,
                max_value=max_window,
                step=2,
                key="lab_smoothing_window",
                **_widget_default("lab_smoothing_window", default_window),
            )
            smoothing_order = smooth_columns[1].number_input(
                "Polynomial order",
                min_value=1,
                max_value=max(1, int(smoothing_window) - 1),
                key="lab_smoothing_order",
                **_widget_default(
                    "lab_smoothing_order", min(3, int(smoothing_window) - 1)
                ),
            )
        derivative_order = st.selectbox(
            "Derivative",
            ["None", "First derivative", "Second derivative"],
            key="lab_derivative",
        )

    with st.expander(
        "4 · Baseline correction",
        expanded=imported_steps is not None,
    ):
        baseline_method = st.selectbox(
            "Baseline method",
            ["None", "arPLS", "Polynomial", "Rubber-band"],
            key="lab_baseline_method",
        )
        baseline_parameters: dict[str, float | int] = {}
        if baseline_method == "arPLS":
            baseline_columns = st.columns(3)
            baseline_parameters["lam"] = baseline_columns[0].number_input(
                "Smoothness λ",
                min_value=1.0,
                format="%.3e",
                key="lab_arpls_lam",
                **_widget_default("lab_arpls_lam", 1e5),
            )
            baseline_parameters["ratio"] = baseline_columns[1].number_input(
                "Convergence ratio",
                min_value=1e-12,
                format="%.3e",
                key="lab_arpls_ratio",
                **_widget_default("lab_arpls_ratio", 1e-6),
            )
            baseline_parameters["max_iterations"] = baseline_columns[2].number_input(
                "Maximum iterations",
                min_value=1,
                max_value=200,
                key="lab_arpls_iterations",
                **_widget_default("lab_arpls_iterations", 30),
            )
        elif baseline_method == "Polynomial":
            baseline_columns = st.columns(2)
            baseline_parameters["degree"] = baseline_columns[0].number_input(
                "Polynomial degree",
                min_value=0,
                max_value=8,
                key="lab_polynomial_degree",
                **_widget_default("lab_polynomial_degree", 2),
            )
            baseline_parameters["edge_fraction"] = baseline_columns[1].slider(
                "Fraction fitted at each edge",
                min_value=0.02,
                max_value=0.50,
                step=0.01,
                key="lab_polynomial_edges",
                **_widget_default("lab_polynomial_edges", 0.10),
            )

    with st.expander(
        "5 · Normalization",
        expanded=imported_steps is not None,
    ):
        normalization_method = st.selectbox(
            "Normalization method",
            ["None", "Min–max", "Vector", "Area"],
            key="lab_normalization_method",
        )
        normalization_range = crop_range
        if normalization_method != "None":
            normalization_range = st.slider(
                "Normalization interval",
                min_value=float(crop_range[0]),
                max_value=float(crop_range[1]),
                key="lab_normalization_range",
                **_widget_default(
                    "lab_normalization_range",
                    (float(crop_range[0]), float(crop_range[1])),
                ),
            )

    if enable_background and background is None:
        st.warning(
            "Processing preview is paused until a background spectrum is selected, "
            "or background normalization is disabled."
        )
        return

    current = source
    recipe: list[dict] = []
    processing_warnings: list[str] = []
    baseline_preview: tuple[Spectrum, np.ndarray, Spectrum] | None = None
    try:
        if enable_crop:
            current = crop(current, *crop_range)
            recipe.append(_recipe_step("crop", lower=float(crop_range[0]), upper=float(crop_range[1])))

        if enable_resample:
            current = resample(current, float(spacing))
            recipe.append(_recipe_step("resample", spacing=float(spacing)))

        if background is not None:
            current = background_normalize(current, background)
            recipe.append(
                _recipe_step(
                    "background_normalization",
                    binding="background",
                )
            )

        if representation != "Keep current values":
            percent = representation.startswith("Percentage")
            warning = transmittance_warning(current, percent=percent)
            if warning is not None:
                processing_warnings.append(warning)
            current = to_absorbance(current, percent=percent)
            recipe.append(_recipe_step("transmittance_to_absorbance", percent=percent))

        if enable_smoothing:
            current = savgol_smooth(current, int(smoothing_window), int(smoothing_order))
            recipe.append(
                _recipe_step(
                    "savitzky_golay",
                    window=int(smoothing_window),
                    polynomial_order=int(smoothing_order),
                )
            )

        if derivative_order != "None":
            selected_order = 1 if derivative_order == "First derivative" else 2
            current = derivative(current, selected_order)
            recipe.append(_recipe_step("derivative", order=selected_order))

        if baseline_method != "None":
            before_baseline = current
            current, estimated_baseline = baseline_correct(
                current,
                baseline_method,
                **baseline_parameters,
            )
            baseline_preview = (before_baseline, estimated_baseline, current)
            recipe.append(
                _recipe_step(
                    "baseline_correction",
                    method=baseline_method,
                    **baseline_parameters,
                )
            )

        if normalization_method != "None":
            current = normalize(
                current,
                normalization_method,
                lower=float(normalization_range[0]),
                upper=float(normalization_range[1]),
            )
            recipe.append(
                _recipe_step(
                    "normalize",
                    method=normalization_method,
                    lower=float(normalization_range[0]),
                    upper=float(normalization_range[1]),
                )
            )
    except (ValueError, RuntimeError) as error:
        st.error(f"Processing stopped: {error}")
        return

    for warning in processing_warnings:
        st.warning(warning)

    st.markdown("### Preview")
    preview_scope = st.radio(
        "Preview range",
        ["Full original spectrum", "Processed/cropped range only"],
        horizontal=True,
        key="lab_preview_scope",
        help="This changes only the displayed x-range; processing and exported data are unaffected.",
    )
    original_visibility = st.radio(
        "Original spectrum",
        ["Show original", "Hide original"],
        horizontal=True,
        key="lab_original_visibility",
        help="This changes only the main preview plot.",
    )
    reverse_axis = st.checkbox("Use conventional descending wavenumber axis", value=True, key="lab_reverse_axis")
    preview_source = source
    if preview_scope == "Processed/cropped range only":
        preview_source = crop(source, float(current.x.min()), float(current.x.max()))
    st.plotly_chart(
        _spectrum_plot(
            preview_source,
            current,
            reverse_axis,
            show_original=original_visibility == "Show original",
        ),
        width="stretch",
    )

    if baseline_preview is not None:
        with st.expander("Inspect baseline estimate"):
            st.plotly_chart(
                _baseline_plot(*baseline_preview, reverse_axis),
                width="stretch",
            )

    if recipe:
        recipe_frame = _recipe_table(recipe)
        st.dataframe(recipe_frame, width="stretch", hide_index=True)
    else:
        st.info("No processing operations are enabled; the exported values will match the loaded spectrum.")

    if imported_steps is not None:
        needs_background = (
            "background" in required_bindings(imported_steps)
            and enable_background
            and background is None
        )
        if needs_background:
            st.warning(
                "Imported recipe status: waiting for the required background spectrum."
            )
        elif recipe == imported_steps:
            st.success("Imported recipe status: unchanged.")
        else:
            st.info(
                "Imported recipe status: modified. The recipe downloaded below will "
                "contain the current edited parameters."
            )

    original_on_output_grid = np.interp(current.x, source.x, source.y)
    output_frame = pd.DataFrame(
        {
            f"wavenumber_{current.x_unit}": current.x,
            "original": original_on_output_grid,
            "processed": current.y,
        }
    )
    recipe_document = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source.name,
        "x_unit": current.x_unit,
        "output_label": current.y_label,
        "required_bindings": sorted(required_bindings(recipe)),
        "steps": recipe,
    }

    st.markdown("### Export")
    export_columns = st.columns(2)
    base_name = uploaded.name.rsplit(".", 1)[0] if uploaded is not None else "spectrum"
    export_columns[0].download_button(
        "Download processed CSV",
        data=output_frame.to_csv(index=False),
        file_name=f"{base_name}_processed.csv",
        mime="text/csv",
        width="stretch",
    )
    export_columns[1].download_button(
        "Download processing recipe",
        data=json.dumps(recipe_document, indent=2, ensure_ascii=False),
        file_name=f"{base_name}_recipe.json",
        mime="application/json",
        width="stretch",
    )
