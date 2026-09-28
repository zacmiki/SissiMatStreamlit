import sys
import unittest
import zipfile
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


# The conversion helpers do not need a running Streamlit installation.
try:
    import streamlit  # noqa: F401
except ImportError:
    sys.modules["streamlit"] = SimpleNamespace()

import ConvertFiles


class UploadedFileStub:
    def __init__(self, name, contents):
        self.name = name
        self._contents = contents

    def getvalue(self):
        return self._contents


class ConvertFilesTests(unittest.TestCase):
    def test_safe_archive_path_makes_windows_separators_portable(self):
        self.assertEqual(
            ConvertFiles._safe_archive_path(r"experiment\run_1\sample.0"),
            "experiment/run_1/sample.0",
        )

    def test_safe_archive_path_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            ConvertFiles._safe_archive_path("experiment/../outside.0")

    def test_folder_conversion_preserves_subfolders_and_reports_invalid_files(self):
        uploads = [
            UploadedFileStub("experiment/run_1/sample.0", b"valid"),
            UploadedFileStub("experiment/notes.txt", b"invalid"),
        ]

        def is_opus_file(path):
            with open(path, "rb") as source:
                return source.read() == b"valid"

        spectrum = SimpleNamespace(
            x=np.array([1000.0, 1001.0]),
            y=np.array([0.5, 0.6]),
        )
        with (
            patch.object(ConvertFiles.opusFC, "isOpusFile", side_effect=is_opus_file),
            patch.object(ConvertFiles.opusFC, "listContents", return_value=[("AB",)]),
            patch.object(ConvertFiles.opusFC, "getOpusData", return_value=spectrum),
        ):
            archive_buffer, invalid_files = ConvertFiles._convert_uploaded_files(uploads)

        with zipfile.ZipFile(BytesIO(archive_buffer.getvalue())) as archive:
            self.assertEqual(
                archive.namelist(),
                ["experiment/run_1/sample.0.AB.txt"],
            )
        self.assertEqual(invalid_files, ["experiment/notes.txt"])


if __name__ == "__main__":
    unittest.main()
