import streamlit as st
import os
import opusFC
import numpy as np
import zipfile
from io import BytesIO
from pathlib import PurePosixPath
from tempfile import TemporaryDirectory


def _safe_archive_path(filename):
    """Return a safe, portable relative path for an uploaded file."""
    normalized = str(filename).replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"Unsafe uploaded path: {filename}")

    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts:
        raise ValueError("An uploaded file has no filename.")
    return "/".join(parts)


def _safe_suffix(value):
    """Keep an OPUS dataset label from creating extra ZIP directories."""
    suffix = str(value).replace("/", "_").replace("\\", "_").strip()
    return suffix or "DATA"


def _add_converted_opus(zip_file, source_path, archive_source_path):
    """Convert one OPUS file into one or more text members in a ZIP."""
    if not opusFC.isOpusFile(source_path):
        return False

    for dataset in opusFC.listContents(source_path):
        data = opusFC.getOpusData(source_path, dataset)
        suffix = _safe_suffix(dataset[0])
        output_path = f"{archive_source_path}.{suffix}.txt"
        spectrum = np.column_stack((data.x, data.y))

        with zip_file.open(output_path, "w") as output_file:
            np.savetxt(output_file, spectrum, delimiter=",", fmt="%f")
    return True


def _convert_uploaded_files(uploaded_files):
    """Convert browser uploads while retaining their relative folder paths."""
    zip_buffer = BytesIO()
    invalid_files = []

    with TemporaryDirectory(prefix="sissimat_upload_") as temp_directory:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for index, uploaded_file in enumerate(uploaded_files):
                archive_source_path = _safe_archive_path(uploaded_file.name)

                # Use an app-controlled temporary filename. UploadedFile.name is
                # deliberately retained only as a member name inside the ZIP.
                upload_directory = os.path.join(temp_directory, str(index))
                os.mkdir(upload_directory)
                temp_path = os.path.join(
                    upload_directory, PurePosixPath(archive_source_path).name
                )
                with open(temp_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.getvalue())

                if not _add_converted_opus(zip_file, temp_path, archive_source_path):
                    invalid_files.append(archive_source_path)

    zip_buffer.seek(0)
    return zip_buffer, invalid_files


def _show_invalid_files(invalid_files):
    if invalid_files:
        st.warning(
            "The following files were ignored because they are not valid OPUS files: "
            f"{', '.join(invalid_files)}"
        )


def convert_opus_files_in_directory():
    st.title("🌈️ SISSI-Mat File Utilities")
    st.divider()
    st.header("OPUS File Batch Converter")
    st.write("Choose a method below to convert OPUS files into text files and download as a ZIP.")

    tab1, tab2 = st.tabs(["Upload Files", "Upload Folder"])

    # --- Tab 1: Upload individual files ---
    with tab1:
        uploaded_files = st.file_uploader(
            "Upload OPUS files", accept_multiple_files=True, key="upload"
        )

        if uploaded_files:
            zip_buffer, invalid_files = _convert_uploaded_files(uploaded_files)
            _show_invalid_files(invalid_files)

            st.download_button(
                label="Download Converted Files as ZIP",
                data=zip_buffer,
                file_name="converted_files.zip",
                mime="application/zip",
                on_click="ignore",
            )

    # --- Tab 2: Upload a complete folder recursively from the browser ---
    with tab2:
        st.caption(
            "Choose or drag in a folder. Files in all subfolders are included, "
            "and the same folder structure is kept in the downloaded ZIP."
        )
        folder_files = st.file_uploader(
            "Upload a folder containing OPUS files",
            accept_multiple_files="directory",
            key="folder_upload",
        )

        if folder_files:
            st.info(f"Received {len(folder_files)} file(s).")
            zip_buffer, invalid_files = _convert_uploaded_files(folder_files)
            _show_invalid_files(invalid_files)

            st.download_button(
                label="Download Converted Folder as ZIP",
                data=zip_buffer,
                file_name="converted_experiment.zip",
                mime="application/zip",
                on_click="ignore",
            )

if __name__ == "__main__":
    convert_opus_files_in_directory()
