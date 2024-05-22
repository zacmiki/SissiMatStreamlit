import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from irconverters import page1
from dacutilities import page2

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
        ],
    )
    
    if selected_option == "IR Converters":
        page1()
        
    else:
    
        if selected_option == "DAC Utilities":
            page2()
            #andre alla pagina delle DAC Utilities

if __name__ == "__main__":
    main()


