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

def convert_opus_files_in_directory():    
    st.title("🌈️ SISSI-Mat File Utilities")
    st.divider()
    st.header("OPUS File Batch Converter")
    st.write("Upload OPUS files to convert them into text files and download as a ZIP.")
    
    uploaded_files = st.file_uploader("Upload OPUS files", accept_multiple_files=True, type=["0", "spa", "spc", "ir"])
    
    if uploaded_files:
        zip_buffer = BytesIO()
    
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for uploaded_file in uploaded_files:
                st.write(f"Processing: {uploaded_file.name}")
    
                # Save the uploaded file to a temporary location
                temp_filename = f"/tmp/{uploaded_file.name}"
                with open(temp_filename, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
    
                # Use the file path instead of BytesIO
                if opusFC.isOpusFile(temp_filename):
                    dbs = opusFC.listContents(temp_filename)
                    dataSets = len(dbs)
    
                    for index, sets in enumerate(dbs):
                        data = opusFC.getOpusData(temp_filename, sets)
                        suffix = sets[0]
    
                        txt_filename = f"{uploaded_file.name}.{suffix}.txt"
                        spectrum = np.column_stack((data.x, data.y))
                        
                        # Convert the numpy array to a CSV string
                        output = io.StringIO()
                        np.savetxt(output, spectrum, delimiter=',', fmt='%s')
                        csv_string = output.getvalue()
    
                        with zip_file.open(txt_filename, "w") as output_file:
                            np.savetxt(output_file, spectrum, delimiter=',', fmt='%f')
                else:
                    st.error(f"Error: {uploaded_file.name} is not a valid OPUS file.")
    
        zip_buffer.seek(0)
        st.download_button(
            label="Download Converted Files as ZIP",
            data=zip_buffer,
            file_name="converted_files.zip",
            mime="application/zip"
        )
