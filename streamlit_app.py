import streamlit as st

st.set_page_config(
    page_title="SISSI-Mat IR Utilities",
    page_icon="🌈",
    layout="wide",
)

from converters import page1
from dacutilities import page2
from OpusGraher import graphopus
from Opusinspector import inspectopusfile
from averagespectra import averagespectrapage, get_elettra_status
from fitruby_ls import rubyfitls
from preanalysis import online_analysis
from ConvertFiles import convert_opus_files_in_directory
from sissimat.pages.spectral_processing import render_spectral_processing_lab

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
    st.sidebar.markdown("### 🌈 IR Utilities")
    if st.sidebar.button("IR Converters", use_container_width=True):
        st.session_state.current_page = "IR Converters"

    # Section: OPUS files
    st.sidebar.markdown("### 📂 OPUS Files")
    if st.sidebar.button("OPUS File Grapher", use_container_width=True):
        st.session_state.current_page = "OPUS File Grapher"
    if st.sidebar.button("OPUS File Inspector", use_container_width=True):
        st.session_state.current_page = "OPUS File Inspector"
    if st.sidebar.button("OPUS File Converter", use_container_width=True):
        st.session_state.current_page = "OPUS File Converter"
    if st.sidebar.button("OPUS Spectra Averaging", use_container_width=True):
        st.session_state.current_page = "OPUS Spectra Averaging"

    # Section: Data processing
    st.sidebar.markdown("### 🔬 Data Processing")
    if st.sidebar.button("Quick Pre-analysis", use_container_width=True):
        st.session_state.current_page = "Quick Pre-analysis"
    if st.sidebar.button("Spectral Processing Lab", use_container_width=True):
        st.session_state.current_page = "Spectral Processing Lab"

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
    elif selected_option in ("Quick Pre-analysis", "Online Basic Data Analysis"):
        online_analysis()
    elif selected_option == "Spectral Processing Lab":
        render_spectral_processing_lab()

    with st.sidebar:
        get_elettra_status()
        st.divider()


if __name__ == "__main__":
    main()
