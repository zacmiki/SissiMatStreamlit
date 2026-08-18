import unittest
import zipfile
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sissimat.pages.batch_processing import (
    _build_archive,
    _filename_mapping,
    _preflight_message,
    _safe_stem,
)
from sissimat.processing.models import Spectrum


class BatchProcessingTests(unittest.TestCase):
    def test_filename_pairing_uses_matching_run_stems(self):
        mapping = _filename_mapping(
            ["run1_sample.0", "run2_sample.0"],
            ["run2_background.0", "run1_background.0"],
        )

        self.assertEqual(mapping["run1_sample.0"], "run1_background.0")
        self.assertEqual(mapping["run2_sample.0"], "run2_background.0")

    def test_opus_numeric_extension_is_preserved_in_output_stem(self):
        self.assertEqual(_safe_stem("sample.17"), "sample.17")
        self.assertEqual(_safe_stem("sample.csv"), "sample")

    def test_preflight_warns_when_ssc_or_background_coverage_is_missing(self):
        sample = {
            "dataset": "SIFG",
            "spectrum": Spectrum(np.linspace(1000, 1100, 11), np.ones(11)),
        }
        background = {
            "dataset": "SSC",
            "spectrum": Spectrum(np.linspace(1020, 1080, 7), np.ones(7)),
        }

        message = _preflight_message(sample, [], background)

        self.assertIn("SSC unavailable", message)
        self.assertIn("does not cover", message)

    def test_archive_contains_unique_opus_outputs_and_manifest(self):
        x = np.linspace(1000, 1010, 3)
        successful = {
            "sample.0": SimpleNamespace(spectrum=Spectrum(x, np.ones(3))),
            "sample.1": SimpleNamespace(spectrum=Spectrum(x, np.ones(3) * 2)),
        }
        manifest = pd.DataFrame([{"sample_file": "sample.0", "status": "Success"}])

        archive_bytes = _build_archive(
            successful,
            manifest,
            {"schema_version": 1, "steps": []},
            ["SUCCESS sample.0", "SUCCESS sample.1"],
        )

        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            names = set(archive.namelist())
        self.assertIn("processed/sample.0_processed.csv", names)
        self.assertIn("processed/sample.1_processed.csv", names)
        self.assertIn("manifest.csv", names)
        self.assertIn("recipe.json", names)


if __name__ == "__main__":
    unittest.main()
