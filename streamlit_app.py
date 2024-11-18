import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from irconverters import page1
from dacutilities import page2
from averagespectra import averagespectrapage, get_elettra_status
from fitruby import rubyfit
from fitruby_ls import rubyfitls
from fitruby_voigt import rubyfit_v
from OpusGraher import graphopus
from ConvertOpusFiles import convert_opus_files_in_directory
from Opusinspector import inspectopusfile
from ringparameters import *

# Set up the sidebar.  -  SIDEBAR ----------- SIDEBAR ------------ SIDEBAR OPTIONS
#st.set_page_config(layout="wide")

st.sidebar.title("🌈SISSI IR Utilities")
st.sidebar.caption("By Zac")
st.sidebar.write("Please select an option from the sidebar.")

# Main app logic.   ------ MAIN APP LOGIC --- HANDLING OF THE MENU
def main():
    selected_option = st.sidebar.selectbox(
        "Select an option",
        [
            "DAC Utilities",
            "IR Converters",
            "Graph Opus File",
            "Opus File Inspector",
            "Multiple Spectra Averaging",
            "Fit Ruby",
            "Convert All OPUS Files"
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
    elif selected_option == "OPUS Spectra Averaging":
        averagespectrapage()
    elif selected_option == "Fit Ruby":
        rubyfitls()
    elif selected_option == "Convert All OPUS Files":
        convert_opus_files_in_directory()
    
    with st.sidebar:
        get_elettra_status()


if __name__ == "__main__":
    main()
