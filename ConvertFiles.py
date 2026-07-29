import streamlit as st
import os
import opusFC
import numpy as np
import zipfile
from io import BytesIO


def convert_opus_files_in_directory():
    st.title("🌈️ SISSI-Mat File Utilities")
    st.divider()
    st.header("OPUS File Batch Converter")
    st.write("Choose a method below to convert OPUS files into text files and download as a ZIP.")

    tab1, tab2 = st.tabs(["Upload Files", "Scan Local Folder"])

    # --- Tab 1: Upload individual files ---
    with tab1:
        uploaded_files = st.file_uploader("Upload OPUS files", accept_multiple_files=True, key="upload")

        if uploaded_files:
            zip_buffer = BytesIO()
            invalid_files = []

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for uploaded_file in uploaded_files:
                    st.write(f"Processing: {uploaded_file.name}")

                    temp_filename = f"/tmp/{uploaded_file.name}"
                    with open(temp_filename, "wb") as temp_file:
                        temp_file.write(uploaded_file.read())

                    if opusFC.isOpusFile(temp_filename):
                        dbs = opusFC.listContents(temp_filename)

                        for sets in dbs:
                            data = opusFC.getOpusData(temp_filename, sets)
                            suffix = sets[0]

                            txt_filename = f"{uploaded_file.name}.{suffix}.txt"
                            spectrum = np.column_stack((data.x, data.y))

                            with zip_file.open(txt_filename, "w") as output_file:
                                np.savetxt(output_file, spectrum, delimiter=',', fmt='%f')
                    else:
                        invalid_files.append(uploaded_file.name)

            zip_buffer.seek(0)

            if invalid_files:
                st.warning(
                    f"The following files were ignored because they are not valid OPUS files: "
                    f"{', '.join(invalid_files)}"
                )

            st.download_button(
                label="Download Converted Files as ZIP",
                data=zip_buffer,
                file_name="converted_files.zip",
                mime="application/zip"
            )

    # --- Tab 2: Scan a local folder recursively ---
    with tab2:
        folder_path = st.text_input(
            "Root folder containing OPUS files (subfolders scanned recursively):",
            placeholder="e.g. /path/to/experiment"
        )

        if folder_path:
            if not os.path.isdir(folder_path):
                st.error("The provided path does not exist or is not a directory.")
            else:
                # Gather all OPUS files recursively
                opus_files = []
                for root, dirs, files in os.walk(folder_path):
                    for fname in files:
                        full_path = os.path.join(root, fname)
                        if opusFC.isOpusFile(full_path):
                            opus_files.append(full_path)

                if not opus_files:
                    st.warning("No valid OPUS files found in the specified folder or its subfolders.")
                else:
                    st.info(f"Found {len(opus_files)} OPUS file(s). Click below to start conversion.")

                    if st.button("Convert All OPUS Files", type="primary"):
                        zip_buffer = BytesIO()
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        invalid_files = []

                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for i, full_path in enumerate(opus_files):
                                rel_path = os.path.relpath(full_path, folder_path)
                                status_text.text(f"Processing: {rel_path}")

                                if opusFC.isOpusFile(full_path):
                                    dbs = opusFC.listContents(full_path)

                                    for sets in dbs:
                                        data = opusFC.getOpusData(full_path, sets)
                                        suffix = sets[0]

                                        txt_rel_path = f"{rel_path}.{suffix}.txt"
                                        spectrum = np.column_stack((data.x, data.y))

                                        with zip_file.open(txt_rel_path, "w") as output_file:
                                            np.savetxt(output_file, spectrum, delimiter=',', fmt='%f')
                                else:
                                    invalid_files.append(rel_path)

                                progress_bar.progress((i + 1) / len(opus_files))

                        zip_buffer.seek(0)
                        status_text.text("Done!")

                        if invalid_files:
                            st.warning(
                                f"The following files were ignored (not valid OPUS): "
                                f"{', '.join(invalid_files)}"
                            )

                        st.download_button(
                            label="Download Converted Files as ZIP",
                            data=zip_buffer,
                            file_name="converted_experiment.zip",
                            mime="application/zip"
                        )


if __name__ == "__main__":
    convert_opus_files_in_directory()
