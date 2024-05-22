import streamlit as st


# Page Visualization ----------------------------------

def page1():
    import pandas as pd
    import numpy as np
    from sissi_util import thz2wn, en2wn, wl2wn, wn2en
        
    # Filter the DataFrame to select the first 20 elements where 'SD' is not NaN
    # --------- PAGE LAYOUT
    
    st.title("🌈️ Infrared Spectroscopy Utils")
    st.subheader("Converters for IR Spectroscopy")
    st.divider()
    
    with st.container():
        left_column, right_column = st.columns(2)
        
        with left_column:

            st.markdown("##### :red[THz to cm-1]")
            thz = st.number_input("", value=1.0, key="thz2wn", label_visibility="collapsed")
            wn = thz2wn(thz)
            st.markdown(f"##### {thz} Thz = **{wn:.0f} cm-1**")

            st.markdown("##### :red[Wavelength [nm] to cm-1]")
            wl = st.number_input("", value=1.0, key="wl2wn", label_visibility="collapsed")
            wn3 = wl2wn(wl)
            st.markdown(f"##### {wl} nm = **{wn3:.0f} cm-1**")
            
            
            
        with right_column:
            
            st.markdown("##### :red[meV to cm-1]")
            ev = st.number_input("", value=1.0, key="ev2wn", label_visibility="collapsed")
            wn2 = en2wn(ev)
            st.markdown(f"##### {ev} meV = **{wn2:.0f} cm-1**")
            
            st.markdown("##### :red[cm-1 to meV]")
            wn4 = st.number_input("", value=1.0, key="wn2en", label_visibility="collapsed")
            en3 = wn2en(wn4)
            st.markdown(f"##### {wn4} cm-1 = **{en3:.0f} meV**")
            
            
    st.divider()
    st.header(":rainbow[Cool comments and notes underneath]")
            
            
