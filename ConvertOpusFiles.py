import streamlit as st
import os
import opusFC
import numpy as np
import zipfile
from io import BytesIO

# Function to get a list of all OPUS files in a directory and its subdirectories
def get_opus_files(dir_name):
    opus_files = []
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if opusFC.isOpusFile(os.path.join(root, file)):
                opus_files.append(os.path.join(root, file))
    return opus_files

# Function to convert OPUS files and save as text files in a ZIP archive with logging
def convert_opus_files_in_directory():
    st.title("🌈️ SISSI-Mat File Utilities")
    st.divider()
    st.header("OPUS File Batch Converter")
    st.write("This tool will SCAN for ALL OPUS FIles starting from the directory you input manually")
    st.write("Then it will convert ALL data and prepare a ZIP archive with all the contents.")

    # Input for directory path
    dir_name = st.text_input("Enter the directory path containing OPUS files:")

    if dir_name and os.path.isdir(dir_name):
        # Find all OPUS files in the specified directory and subdirectories
        opus_files = get_opus_files(dir_name)

        if opus_files:
            # Prepare an in-memory ZIP file to store converted files
            zip_buffer = BytesIO()

            # Display a placeholder for logging
            log_placeholder = st.empty()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for opus_file in opus_files:
                    # Update log for each file being processed
                    log_placeholder.text(f"Processing file: {opus_file}")
                    
                    dbs = opusFC.listContents(opus_file)
                    data_sets = len(dbs)

                    for sets in range(data_sets):
                        data = opusFC.getOpusData(opus_file, dbs[sets])

                        # Define filename for each dataset
                        for item in dbs:
                            suffix = item[0]
                            base_filename = os.path.basename(opus_file)
                            txt_filename = f"{base_filename}.{suffix}.txt"
                            spectrum = np.column_stack((data.x, data.y))

                            # Save to ZIP
                            with zip_file.open(txt_filename, "w") as output_file:
                                np.savetxt(output_file, spectrum, delimiter=',', fmt='%f')

            # Clear the log once processing is complete
            log_placeholder.text("Processing complete. Download File prepared")

            # Provide ZIP file for download
            zip_buffer.seek(0)
            st.download_button(
                label="Download Converted Files as ZIP",
                data=zip_buffer,
                file_name="converted_files.zip",
                mime="application/zip"
            )
        else:
            st.write("No OPUS files found in the specified directory.")
    elif dir_name:
        st.write("The specified path is not a valid directory. Please enter a valid directory path.")

# Run the function in Streamlit
#convert_opus_files_in_directory()
