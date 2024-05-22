import streamlit as st


# Page Visualization ----------------------------------

def page1():
    import pandas as pd
    import numpy as np
    from sissi_util import DACTemp, DACPress
        
    # Filter the DataFrame to select the first 20 elements where 'SD' is not NaN
    # --------- PAGE LAYOUT
    
    st.title("🌈️ Infrared Spectroscopy Utils")
    st.subheader("Useful Infrared Converters")
    st.divider()
    
    st.subheader("1 - Pressure in GPa from Ruby Line Position")
    #st.text("Pressure in GPa from Ruby Fluorescence Line Position")

    st.markdown(f"#### :red[Ruby Position in nm] ####")
    
    linepos = st.number_input("", step = 0.01 , format="%.2f", value = 694.19)
    st.markdown(f"#### :red[Corresponding Pressure =]  {DACPress(linepos)} GPa ####")
    
    
    df = pd.DataFrame({
        'first column': [1,2,3,4],
        'second column': [10,20,30,40]
    })
    df
    
