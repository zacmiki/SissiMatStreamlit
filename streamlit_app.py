import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

#from irconverters import page1
from converters import page1
from dacutilities import page2
from OpusGraher import graphopus
from Opusinspector import inspectopusfile
from averagespectra import averagespectrapage, get_elettra_status
from fitruby_ls import rubyfitls

from preanalysis import online_analysis

from fitruby import rubyfit
from fitruby_voigt import rubyfit_v

#from ConvertOpusFiles import convert_opus_files_in_directory
from ConvertFiles import convert_opus_files_in_directory

from ringparameters import *

# Set up the sidebar.  -  SIDEBAR ----------- SIDEBAR ------------ SIDEBAR OPTIONS
#st.set_page_config(layout="wide")

st.sidebar.title("🌈 SISSI IR Utilities")
st.sidebar.caption("By Zac")
st.sidebar.write("Please select an option from the sidebar.")

# Main app logic.   ------ MAIN APP LOGIC --- HANDLING OF THE MENU
def main():
    selected_option = st.sidebar.selectbox(
        "Select an option",
        [
            "IR Converters",
            "Online Basic Data Analysis",
            "Graph Opus File",
            "Opus File Inspector",
            #"OPUS Spectra Averaging",
            "DAC Utilities",
            "Fit Ruby",
            "Convert OPUS Files"
        ],
    )
    
    if selected_option == "IR Converters":
        page1()
    elif selected_option == "DAC Utilities":
        page2()
    elif selected_option == "Graph Opus File":
        graphopus()
    elif selected_option == "Opus File Inspector":
        inspectopusfile()
    #elif selected_option == "OPUS Spectra Averaging":
        #averagespectrapage()
    elif selected_option == "Fit Ruby":
        rubyfitls()
    elif selected_option == "Convert OPUS Files":
        convert_opus_files_in_directory()
    elif "Online Basic Data Analysis":
        online_analysis()
        
    
    with st.sidebar:
        get_elettra_status()        
        st.divider()

if __name__ == "__main__":
    main()
