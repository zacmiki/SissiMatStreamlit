"""Load OPUS and common two-column text spectra."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .models import Spectrum


def _load_opus(file_bytes: bytes, filename: str) -> dict[str, Spectrum] | None:
    try:
        import opusFC
    except ImportError:
        return None

    with TemporaryDirectory(prefix="sissimat_opus_") as directory:
        path = Path(directory) / Path(filename).name
        path.write_bytes(file_bytes)
        try:
            if not opusFC.isOpusFile(str(path)):
                return None
            contents = opusFC.listContents(str(path))
            spectra: dict[str, Spectrum] = {}
            for index, descriptor in enumerate(contents):
                data = opusFC.getOpusData(str(path), descriptor)
                dataset_name = str(descriptor[0])
                label = dataset_name if dataset_name not in spectra else f"{dataset_name} [{index + 1}]"
                parameters = dict(getattr(data, "parameters", {}) or {})
                spectra[label] = Spectrum(
                    x=np.asarray(data.x),
                    y=np.asarray(data.y),
                    name=f"{filename} · {label}",
                    metadata={"source_file": filename, "opus_dataset": dataset_name, **parameters},
                )
            return spectra
        except Exception:
            return None


def _load_text(file_bytes: bytes, filename: str) -> dict[str, Spectrum]:
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    candidates: list[pd.DataFrame] = []
    for separator in (None, ",", ";", "\t", r"\s+"):
        try:
            frame = pd.read_csv(
                StringIO(text),
                sep=separator,
                engine="python",
                comment="#",
                header=None,
                on_bad_lines="skip",
            )
            candidates.append(frame)
        except (pd.errors.ParserError, ValueError):
            continue

    best_x: np.ndarray | None = None
    best_y: np.ndarray | None = None
    for frame in candidates:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        usable = [column for column in numeric.columns if numeric[column].notna().sum() >= 3]
        if len(usable) < 2:
            continue
        pair = numeric[usable[:2]].dropna()
        if best_x is None or pair.shape[0] > best_x.size:
            best_x = pair.iloc[:, 0].to_numpy(dtype=float)
            best_y = pair.iloc[:, 1].to_numpy(dtype=float)

    if best_x is None or best_y is None:
        raise ValueError("Could not find two numeric columns in the uploaded text file.")
    return {
        "Data": Spectrum(
            x=best_x,
            y=best_y,
            name=filename,
            metadata={"source_file": filename, "format": "text"},
        )
    }


def load_spectra(file_bytes: bytes, filename: str) -> dict[str, Spectrum]:
    """Load every spectrum in an OPUS file or a two-column text file."""

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")
    opus_spectra = _load_opus(file_bytes, filename)
    if opus_spectra:
        return opus_spectra
    return _load_text(file_bytes, filename)


def preferred_dataset_label(labels: list[str]) -> str:
    """Prefer the computed single-channel spectrum (SSC) for OPUS workflows."""

    if not labels:
        raise ValueError("No spectral datasets are available.")
    for label in labels:
        dataset_name = label.split(" [", 1)[0].upper()
        if dataset_name == "SSC":
            return label
    return labels[0]
