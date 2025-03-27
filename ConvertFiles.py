import streamlit as st
import os
import opusFC
import numpy as np
import zipfile
from io import BytesIO, StringIO

def convert_opus_files_in_directory():    
    st.title("🌈️ SISSI-Mat File Utilities")
    st.divider()
    st.header("OPUS File Batch Converter")
    st.write("Upload OPUS files to convert them into text files and download as a ZIP.")
    
    uploaded_files = st.file_uploader("Upload OPUS files", accept_multiple_files=True)
    
    if uploaded_files:
        zip_buffer = BytesIO()
        invalid_files = []
    
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for uploaded_file in uploaded_files:
                st.write(f"Processing: {uploaded_file.name}")
    
                # Save the uploaded file to a temporary location
                temp_filename = f"/tmp/{uploaded_file.name}"
                with open(temp_filename, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
    
                # Check if it's a valid OPUS file
                if opusFC.isOpusFile(temp_filename):
                    dbs = opusFC.listContents(temp_filename)
    
                    for sets in dbs:
                        data = opusFC.getOpusData(temp_filename, sets)
                        suffix = sets[0]
    
                        txt_filename = f"{uploaded_file.name}.{suffix}.txt"
                        spectrum = np.column_stack((data.x, data.y))
                        
                        # Convert the numpy array to CSV format
                        output = StringIO()
                        np.savetxt(output, spectrum, delimiter=',', fmt='%s')
                        csv_string = output.getvalue()
    
                        with zip_file.open(txt_filename, "w") as output_file:
                            np.savetxt(output_file, spectrum, delimiter=',', fmt='%f')
                else:
                    invalid_files.append(uploaded_file.name)
    
        zip_buffer.seek(0)
    
        if invalid_files:
            st.warning(f"The following files were ignored because they are not valid OPUS files: {', '.join(invalid_files)}")
    
        st.download_button(
            label="Download Converted Files as ZIP",
            data=zip_buffer,
            file_name="converted_files.zip",
            mime="application/zip"
        )

convert_opus_files_in_directory()
