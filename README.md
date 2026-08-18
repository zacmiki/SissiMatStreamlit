# SISSI - Materials Infrared Utilities

Streamlit utilities for routine infrared-spectroscopy workflows at the SISSI-Mat beamline.

## Current application areas

- IR unit conversion
- OPUS inspection, plotting, conversion, and averaging
- Quick sample/background pre-analysis
- Reproducible spectral processing with CSV and JSON recipe export/replay
- Batch recipe processing with background mapping, quality-control previews, and ZIP export
- Diamond-anvil-cell and ruby-fluorescence utilities

## Project structure

New development lives in the `sissimat` package:

```text
sissimat/
├── pages/          # Single-spectrum and batch Streamlit workflows
└── processing/     # Tested numerical, file-loading, and recipe functions
tests/              # Unit tests for processing behavior
streamlit_app.py    # Application entry point and navigation
```

Legacy pages remain at repository root while they are migrated incrementally. Keeping numerical
functions independent of Streamlit allows them to be tested and reused by multiple workflows.

## Installation

- Set up an enviroment:

```
python -m venv .env
```

- Install requirements:

```
pip3 install -r requirements.txt
```

- Run the app:

```
streamlit run streamlit_app.py
```

- Run the processing tests:

```
python -m unittest discover -v
```
