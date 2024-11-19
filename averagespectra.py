import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from ringparameters import *
from sissi_util import loadSSC
from OpusGraher import opusgrapher
import matplotlib.pyplot as plt
import numpy as np
import io 

def load_opus_data(file_path):
    # Implement the function to load data from OPUS file
    # Assuming it returns a dictionary with 'x' and 'y' keys
    try:
        data = loadSSC(file_path)  # Update this according to your function's actual return type
        if data is not None and hasattr(data, 'x') and hasattr(data, 'y'):
            return data
        else:
            return None
    except Exception as e:
        st.error(f"Error loading OPUS data from {file_path}: {e}")
        return None

def get_elettra_status():
    if st.button(f"""\n\n## GET ELETTRA STATUS"""):
        st.markdown(f"<h1 style = 'text-align: center; color: grey;'>Machine Status<br> {get_machine_status()}</h1>", 
                    unsafe_allow_html = True
        )
    
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"\n##### Ring Energy = {get_energy_value()}")
        with col2:
            st.info(f"\n##### Ring Current = {get_current_value()}")
        

def averaging():
    st.session_state.fileloaded = ""
    fileloaded = ""
    
    with st.form("my-form", clear_on_submit=True):
         path = st.file_uploader("Choose the OPUS Files to Average\n  :red[The files must have the same number of datapoints]", accept_multiple_files=True, label_visibility="visible")
         
         submitted = st.form_submit_button("submit")

    if path:
        y_values_sum = None
        x_values = None
        num_files = 0
        
        for uploaded_file in path:
            if uploaded_file is not None:
                file_name = uploaded_file.name
                
                st.session_state.fileloaded = file_name
                file_extension = file_name.split(".")[-1]
                if file_extension.isdigit():
                    with open("temp.opus", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        data = load_opus_data("temp.opus")
                        if data is not None:
                            if y_values_sum is None:
                                # Initialize the sum with the first file's y-values
                                y_values_sum = np.zeros_like(data.y)
                                x_values = data.x
                            y_values_sum += data.y
                            num_files += 1
                            plt.plot(data.x, data.y, label=f'File {num_files}')
                        else:
                            st.error(f"Failed to load data from {file_name}.")
                else:
                    st.write("Please choose files with integer extensions.")
            else:
                st.write("Please choose only OPUS files.")
    
        if num_files > 0:
            # Compute the average
            averaged_spectrum = y_values_sum / num_files
        
            st.write("Averaged Spectrum:")
            #st.write(averaged_spectrum)
        
            # Plot the averaged spectrum
            plt.plot(x_values, averaged_spectrum, label='Averaged Spectrum', linewidth=2, color='black')
            plt.xlabel('X-axis label')
            plt.ylabel('Y-axis label')
            plt.title('Averaged Spectrum')
            plt.legend()
            plt.grid()
            st.pyplot(plt)
            
            fileaveraged = st.session_state.fileloaded + "_avg.txt"
            
            spectrum = np.column_stack((x_values, averaged_spectrum))
            output = io.StringIO()
            np.savetxt(output, spectrum, delimiter=',', fmt='%s')
            csv_string = output.getvalue()
            fileaveraged = file_name.split('.')[0]
            
            st.download_button(
                label=f"Download Averaged File",
                data=csv_string,
                file_name=fileaveraged,
                mime="text/csv",
                key=f"download_averaged"
            )
            
             
        else:
            st.write("No valid OPUS files uploaded.")

def averagespectrapage():
    st.title(":rainbow[Averaging Utility]")
    st.subheader("Pick a series of equal OPUS Spectra to load them then click the -Submit- Button")
    
    st.markdown(
    f"<h3 style = 'text-align: center; color: yellow;'> Choose a series of n equal spectra to sum and average them </h3>", 
    unsafe_allow_html = True)
    #get_elettra_status()
    
    averaging()
    
