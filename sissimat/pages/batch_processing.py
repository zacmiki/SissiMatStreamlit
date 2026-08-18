"""Batch/series recipe runner for the Spectral Processing Lab."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sissimat.processing.io import load_spectra, preferred_dataset_label
from sissimat.processing.recipe import apply_recipe, required_bindings, validate_recipe_document


@st.cache_data(show_spinner=False)
def _cached_load(file_bytes: bytes, filename: str):
    return load_spectra(file_bytes, filename)


def _load_collection(uploaded_files) -> tuple[dict[str, dict[str, Any]], list[str]]:
    loaded: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for uploaded in uploaded_files or []:
        try:
            file_bytes = uploaded.getvalue()
            spectra = _cached_load(file_bytes, uploaded.name)
            label = preferred_dataset_label(list(spectra))
            loaded[uploaded.name] = {
                "spectrum": spectra[label].cleaned(),
                "dataset": label,
                "sha256": hashlib.sha256(file_bytes).hexdigest(),
            }
        except Exception as error:
            errors.append(f"{uploaded.name}: {error}")
    return loaded, errors


def _canonical_stem(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"(^|[_\-\s])(sample|background|bkg|bg|reference|ref)($|[_\-\s])", "_", stem)
    return re.sub(r"[^a-z0-9]+", "", stem)


def _filename_mapping(sample_names: list[str], background_names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for sample_name in sample_names:
        sample_stem = _canonical_stem(sample_name)
        if not background_names:
            mapping[sample_name] = ""
            continue
        mapping[sample_name] = max(
            background_names,
            key=lambda name: SequenceMatcher(None, sample_stem, _canonical_stem(name)).ratio(),
        )
    return mapping


def _acquisition_time(record: dict[str, Any]) -> pd.Timestamp | None:
    metadata = record["spectrum"].metadata
    date = metadata.get("DAT") or metadata.get("date")
    time = metadata.get("TIM") or metadata.get("time")
    if not date:
        return None
    parsed = pd.to_datetime(f"{date} {time or ''}", errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed


def _time_mapping(
    samples: dict[str, dict[str, Any]],
    backgrounds: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], list[str]]:
    background_times = {
        name: timestamp
        for name, record in backgrounds.items()
        if (timestamp := _acquisition_time(record)) is not None
    }
    warnings: list[str] = []
    mapping: dict[str, str] = {}
    if not background_times:
        return ({name: "" for name in samples}, ["No background acquisition timestamps could be read."])
    for sample_name, record in samples.items():
        timestamp = _acquisition_time(record)
        if timestamp is None:
            mapping[sample_name] = ""
            warnings.append(f"No acquisition timestamp found for {sample_name}.")
            continue
        mapping[sample_name] = min(
            background_times,
            key=lambda name: abs((background_times[name] - timestamp).total_seconds()),
        )
    return mapping, warnings


def _mapping_editor(
    initial_mapping: dict[str, str],
    background_names: list[str],
    key_suffix: str,
) -> dict[str, str]:
    frame = pd.DataFrame(
        {
            "Sample": list(initial_mapping),
            "Background": [initial_mapping[name] for name in initial_mapping],
        }
    )
    edited = st.data_editor(
        frame,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        key=f"batch_mapping_{key_suffix}",
        column_config={
            "Sample": st.column_config.TextColumn(disabled=True),
            "Background": st.column_config.SelectboxColumn(
                options=[""] + background_names,
                required=False,
            ),
        },
    )
    return dict(zip(edited["Sample"], edited["Background"]))


def _safe_stem(filename: str) -> str:
    path = Path(filename)
    source_name = path.stem if path.suffix.lower() in {".csv", ".txt", ".dat"} else path.name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_name).strip("._")
    return stem or "spectrum"


def _preflight_message(
    sample_record: dict[str, Any],
    recipe_steps: list[dict[str, Any]],
    background_record: dict[str, Any] | None,
) -> str:
    messages: list[str] = []
    if sample_record["dataset"].split(" [", 1)[0].upper() != "SSC":
        messages.append(f"SSC unavailable; using {sample_record['dataset']}")
    sample = sample_record["spectrum"]
    lower = float(sample.x.min())
    upper = float(sample.x.max())
    for step in recipe_steps:
        if step["operation"] == "crop":
            lower = max(lower, float(step["parameters"]["lower"]))
            upper = min(upper, float(step["parameters"]["upper"]))
            break
    if lower >= upper:
        messages.append("Recipe crop does not overlap sample")
    if background_record is not None:
        background = background_record["spectrum"]
        if background.x.min() > lower or background.x.max() < upper:
            messages.append("Background does not cover processed range")
        overlap = background.y[(background.x >= lower) & (background.x <= upper)]
        if overlap.size and np.any(np.isclose(overlap, 0.0)):
            messages.append("Background contains zero values")
    return "; ".join(messages) if messages else "Ready"


def _build_archive(
    successful: dict[str, Any],
    manifest: pd.DataFrame,
    recipe_document: dict[str, Any],
    log_lines: list[str],
) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, result in successful.items():
            spectrum = result.spectrum
            frame = pd.DataFrame(
                {
                    f"wavenumber_{spectrum.x_unit}": spectrum.x,
                    "processed": spectrum.y,
                }
            )
            archive.writestr(
                f"processed/{_safe_stem(filename)}_processed.csv",
                frame.to_csv(index=False),
            )
        archive.writestr("manifest.csv", manifest.to_csv(index=False))
        archive.writestr("recipe.json", json.dumps(recipe_document, indent=2, ensure_ascii=False))
        archive.writestr("processing_log.txt", "\n".join(log_lines) + "\n")
    return buffer.getvalue()


def _overlay_figure(successful: dict[str, Any]) -> go.Figure:
    figure = go.Figure()
    for filename, result in successful.items():
        figure.add_trace(
            go.Scatter(x=result.spectrum.x, y=result.spectrum.y, mode="lines", name=filename)
        )
    figure.update_layout(
        height=580,
        hovermode="x unified",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Processed intensity",
    )
    figure.update_xaxes(autorange="reversed")
    return figure


def _waterfall_figure(successful: dict[str, Any]) -> go.Figure:
    figure = go.Figure()
    spectra = [result.spectrum for result in successful.values()]
    spans = [float(np.ptp(spectrum.y)) for spectrum in spectra]
    positive_spans = [span for span in spans if span > 0]
    offset = float(np.median(positive_spans)) * 1.2 if positive_spans else 1.0
    for index, (filename, result) in enumerate(successful.items()):
        figure.add_trace(
            go.Scatter(
                x=result.spectrum.x,
                y=result.spectrum.y + index * offset,
                mode="lines",
                name=filename,
            )
        )
    figure.update_layout(
        height=620,
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Processed intensity + offset",
    )
    figure.update_xaxes(autorange="reversed")
    return figure


def _heatmap_figure(successful: dict[str, Any]) -> go.Figure | None:
    items = list(successful.items())
    common_min = max(result.spectrum.x.min() for _, result in items)
    common_max = min(result.spectrum.x.max() for _, result in items)
    if common_min >= common_max:
        return None
    reference_x = items[0][1].spectrum.x
    common_x = reference_x[(reference_x >= common_min) & (reference_x <= common_max)]
    if common_x.size < 3:
        return None
    values = np.vstack(
        [np.interp(common_x, result.spectrum.x, result.spectrum.y) for _, result in items]
    )
    figure = go.Figure(
        go.Heatmap(
            x=common_x,
            y=[filename for filename, _ in items],
            z=values,
            colorscale="Viridis",
            colorbar={"title": "Intensity"},
        )
    )
    figure.update_layout(height=max(420, 35 * len(items)), xaxis_title="Wavenumber (cm⁻¹)")
    figure.update_xaxes(autorange="reversed")
    return figure


def render_batch_processing() -> None:
    st.subheader("Batch/series recipe runner")
    st.caption("Load a saved recipe, assign any required backgrounds, and process an entire series.")

    recipe_upload = st.file_uploader("Processing recipe (JSON)", type=["json"], key="batch_recipe")
    if recipe_upload is None:
        st.info("Export a recipe from Single spectrum mode, then load it here.")
        return
    try:
        recipe_document = json.loads(recipe_upload.getvalue().decode("utf-8"))
        steps = validate_recipe_document(recipe_document)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Invalid recipe: {error}")
        return

    st.dataframe(
        pd.DataFrame(
            {
                "Step": range(1, len(steps) + 1),
                "Operation": [step["operation"] for step in steps],
                "Parameters": [json.dumps(step["parameters"], ensure_ascii=False) for step in steps],
            }
        ),
        hide_index=True,
        width="stretch",
    )

    sample_uploads = st.file_uploader(
        "Sample spectra",
        accept_multiple_files=True,
        key="batch_samples",
    )
    samples, sample_errors = _load_collection(sample_uploads)
    for error in sample_errors:
        st.warning(error)
    if not samples:
        st.info("Load two or more sample spectra to run a series.")
        return

    required = required_bindings(steps)
    unsupported_bindings = required - {"background"}
    if unsupported_bindings:
        st.error(f"Unsupported recipe bindings: {', '.join(sorted(unsupported_bindings))}")
        return
    backgrounds: dict[str, dict[str, Any]] = {}
    mapping: dict[str, str] = {name: "" for name in samples}
    mapping_warnings: list[str] = []
    if "background" in required:
        background_uploads = st.file_uploader(
            "Background spectra (I₀)",
            accept_multiple_files=True,
            key="batch_backgrounds",
        )
        backgrounds, background_errors = _load_collection(background_uploads)
        for error in background_errors:
            st.warning(error)
        if not backgrounds:
            st.info("This recipe requires at least one background spectrum.")
            return

        strategy = st.radio(
            "Background assignment",
            ["One shared background", "Pair by filename", "Nearest acquisition time", "Manual"],
            horizontal=True,
            key="batch_background_strategy",
        )
        sample_names = list(samples)
        background_names = list(backgrounds)
        if strategy == "One shared background":
            selected_background = st.selectbox("Shared background", background_names)
            mapping = {name: selected_background for name in sample_names}
        else:
            if strategy == "Pair by filename":
                initial = _filename_mapping(sample_names, background_names)
            elif strategy == "Nearest acquisition time":
                initial, mapping_warnings = _time_mapping(samples, backgrounds)
            else:
                initial = {name: "" for name in sample_names}
            mapping_signature = hashlib.sha1(
                (strategy + "|" + "|".join(sample_names) + "|" + "|".join(background_names)).encode()
            ).hexdigest()[:12]
            mapping = _mapping_editor(initial, background_names, mapping_signature)
        for warning in mapping_warnings:
            st.warning(warning)

    preflight = pd.DataFrame(
        [
            {
                "Sample": name,
                "Dataset": record["dataset"],
                "Background": mapping.get(name, "") or "—",
                "Points": record["spectrum"].x.size,
                "Minimum": float(record["spectrum"].x.min()),
                "Maximum": float(record["spectrum"].x.max()),
                "Preflight": _preflight_message(
                    record,
                    steps,
                    backgrounds.get(mapping.get(name, "")),
                ),
            }
            for name, record in samples.items()
        ]
    )
    st.markdown("#### Preflight")
    st.dataframe(preflight, hide_index=True, width="stretch")

    missing_mappings = [name for name in samples if "background" in required and not mapping.get(name)]
    if missing_mappings:
        st.error(f"Assign a background to: {', '.join(missing_mappings)}")
        return

    run_signature = hashlib.sha256(
        json.dumps(
            {
                "recipe": recipe_document,
                "samples": {name: record["sha256"] for name, record in samples.items()},
                "backgrounds": {name: record["sha256"] for name, record in backgrounds.items()},
                "mapping": mapping,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if st.session_state.get("batch_run_signature") != run_signature:
        st.session_state.pop("batch_run_results", None)

    if st.button("Run recipe on series", type="primary", width="stretch"):
        successful: dict[str, Any] = {}
        manifest_rows: list[dict[str, Any]] = []
        log_lines = [f"Batch started: {datetime.now(timezone.utc).isoformat()}"]
        progress = st.progress(0)
        for index, (filename, record) in enumerate(samples.items(), start=1):
            background_name = mapping.get(filename, "")
            bindings = (
                {"background": backgrounds[background_name]["spectrum"]}
                if background_name
                else {}
            )
            try:
                result = apply_recipe(record["spectrum"], recipe_document, bindings=bindings)
                successful[filename] = result
                status = "Success with warning" if result.warnings else "Success"
                message = " | ".join(result.warnings)
                log_lines.append(f"SUCCESS {filename}")
                for warning in result.warnings:
                    log_lines.append(f"WARNING {filename}: {warning}")
                output_points = result.spectrum.x.size
            except Exception as error:
                status = "Failed"
                message = str(error)
                output_points = 0
                log_lines.append(f"FAILED {filename}: {error}")
            manifest_rows.append(
                {
                    "sample_file": filename,
                    "sha256": record["sha256"],
                    "opus_dataset": record["dataset"],
                    "background_file": background_name,
                    "status": status,
                    "message": message,
                    "output_points": output_points,
                }
            )
            progress.progress(index / len(samples))
        manifest = pd.DataFrame(manifest_rows)
        archive = _build_archive(successful, manifest, recipe_document, log_lines)
        st.session_state.batch_run_signature = run_signature
        st.session_state.batch_run_results = {
            "successful": successful,
            "manifest": manifest,
            "archive": archive,
        }

    batch_results = st.session_state.get("batch_run_results")
    if not batch_results:
        return
    successful = batch_results["successful"]
    manifest = batch_results["manifest"]
    st.markdown("#### Results")
    st.dataframe(manifest, hide_index=True, width="stretch")
    warning_count = int((manifest["status"] == "Success with warning").sum())
    if warning_count:
        st.warning(
            f"{warning_count} spectrum{' was' if warning_count == 1 else ' spectra were'} "
            "processed successfully with transmittance warnings. See the Message column "
            "and processing log for details."
        )
    if successful:
        if len(successful) > 40:
            st.warning("Plots show the first 40 successful spectra to keep the page responsive.")
            plotted = dict(list(successful.items())[:40])
        else:
            plotted = successful
        overlay_tab, waterfall_tab, heatmap_tab = st.tabs(["Overlay", "Waterfall", "Heat map"])
        with overlay_tab:
            st.plotly_chart(_overlay_figure(plotted), width="stretch")
        with waterfall_tab:
            st.plotly_chart(_waterfall_figure(plotted), width="stretch")
        with heatmap_tab:
            heatmap = _heatmap_figure(plotted)
            if heatmap is None:
                st.warning("The processed spectra do not share enough spectral range for a heat map.")
            else:
                st.plotly_chart(heatmap, width="stretch")
    st.download_button(
        "Download processed series as ZIP",
        data=batch_results["archive"],
        file_name="sissimat_processed_series.zip",
        mime="application/zip",
        width="stretch",
    )
