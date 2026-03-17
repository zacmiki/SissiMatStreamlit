import streamlit as st

from converters import page1
from dacutilities import page2
from OpusGraher import graphopus
from Opusinspector import inspectopusfile
from averagespectra import averagespectrapage, get_elettra_status
from fitruby_ls import rubyfitls

from preanalysis import online_analysis

from ConvertFiles import convert_opus_files_in_directory

from ringparameters import *

# Set up the sidebar.  -  SIDEBAR ----------- SIDEBAR ------------ SIDEBAR OPTIONS
# st.set_page_config(layout="wide")

st.sidebar.title("🌈 SISSI IR Utilities")
st.sidebar.caption("By Zac")


# Main app logic.   ------ MAIN APP LOGIC --- HANDLING OF THE MENU
def main():
    # Initialize session state for page selection
    if "current_page" not in st.session_state:
        st.session_state.current_page = "IR Converters"

    # Section: IR Converters
    st.sidebar.markdown("### 🌈 IR Converters")
    if st.sidebar.button("IR Converters", use_container_width=True):
        st.session_state.current_page = "IR Converters"

    # Section: FTIR OPUS Analysis
    st.sidebar.markdown("### 📂 FTIR OPUS Analysis")
    if st.sidebar.button("OPUS File Grapher", use_container_width=True):
        st.session_state.current_page = "OPUS File Grapher"
    if st.sidebar.button("OPUS File Inspector", use_container_width=True):
        st.session_state.current_page = "OPUS File Inspector"
    if st.sidebar.button("OPUS File Converter", use_container_width=True):
        st.session_state.current_page = "OPUS File Converter"
    if st.sidebar.button("OPUS Spectra Averaging", use_container_width=True):
        st.session_state.current_page = "OPUS Spectra Averaging"
    if st.sidebar.button("Online Basic Data Analysis", use_container_width=True):
        st.session_state.current_page = "Online Basic Data Analysis"

    # Section: DAC Tools
    st.sidebar.markdown("### 💎 DAC Tools")
    if st.sidebar.button("DAC Utilities", use_container_width=True):
        st.session_state.current_page = "DAC Utilities"
    if st.sidebar.button("Fit Ruby", use_container_width=True):
        st.session_state.current_page = "Fit Ruby"

    st.sidebar.markdown("---")

    # Get current selection
    selected_option = st.session_state.current_page

    if selected_option == "IR Converters":
        page1()
    elif selected_option == "DAC Utilities":
        page2()
    elif selected_option == "OPUS File Grapher":
        graphopus()
    elif selected_option == "OPUS File Inspector":
        inspectopusfile()
    elif selected_option == "OPUS File Converter":
        convert_opus_files_in_directory()
    elif selected_option == "OPUS Spectra Averaging":
        averagespectrapage()
    elif selected_option == "Fit Ruby":
        rubyfitls()
    elif selected_option == "Online Basic Data Analysis":
        online_analysis()

    with st.sidebar:
        get_elettra_status()
        st.divider()


if __name__ == "__main__":
    main()
