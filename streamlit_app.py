import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from irconverters import page1
from dacutilities import page2
from OpusGraher import graphopus
from Opusinspector import inspectopusfile

# Set up the sidebar.  -  SIDEBAR ----------- SIDEBAR ------------ SIDEBAR OPTIONS
st.sidebar.title("INFRARED Utilities")
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
            "Opus File Inspector"
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
        

if __name__ == "__main__":
    main()
