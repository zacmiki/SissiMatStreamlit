import streamlit as st
import pandas as pd
from sissi_util import DACTemp, DACPress
import numpy as np
import plotly.graph_objects as go



def page2():
    st.title("💎️ DAC Utilities")
    st.divider()

    left_column, right_column = st.columns(2)

    # Initialize DataFrames in session state if they don't exist
    if 'pressure_df' not in st.session_state:
        st.session_state.pressure_df = pd.DataFrame(columns=['RubyLine', 'Press', 'DAC'])
    if 'temp_df' not in st.session_state:
        st.session_state.temp_df = pd.DataFrame(columns=['linepos', 'temperature'])

    # You can use a column just like st.sidebar:
    with left_column:
        st.markdown("##### Pressure from ruby line pos [nm]")
        st.caption("at ambient T")
        st.markdown(f"##### :red[Ruby line pos in nm]")
        rubypos = st.number_input("", step=0.01, format="%.2f", value=694.19, key="DACPress", label_visibility="collapsed")
        pressure = DACPress(rubypos)
        st.markdown(f"###### :red[Corresponding Pressure =] {pressure:.3f} GPa")

        # Add a button to save the pressure value
        if st.button("Add Pressure to DataFrame", key="add_pressure"):
            new_row = pd.DataFrame({'RubyLine': [rubypos], 'Press': [pressure], 'DAC': [0]})
            st.session_state.pressure_df = pd.concat([st.session_state.pressure_df, new_row], ignore_index=True)
            st.success("Pressure value added to DataFrame")

        # Display the editable DataFrame
        pressure_df = st.data_editor(st.session_state.pressure_df, key='pressure_editor')
        st.session_state.pressure_df = pressure_df
        #st.data_editor(st.session_state.pressure_df, key='pressure_editor')

    with right_column:
        st.markdown("##### Temp from ruby line pos in [nm]")
        st.caption("at ambient P")
        st.markdown(f"##### :red[Ruby line pos in nm] ####")
        line4temp = st.number_input("", step=0.01, format="%.2f", value=694.19, key="DACTemp", label_visibility="collapsed")
        temperature = DACTemp(line4temp)
        st.markdown(f"###### :red[Corresponding Temperature =] {temperature:.2f} K")

        # Add a button to save the temperature value
        if st.button("Add Temperature to DataFrame", key="add_temp"):
            new_row = pd.DataFrame({'linepos': [line4temp], 'temperature': [temperature]})
            st.session_state.temp_df = pd.concat([st.session_state.temp_df, new_row], ignore_index=True)
            st.success("Temperature value added to DataFrame")

        # Display the editable DataFrame
        st.data_editor(st.session_state.temp_df, key='temp_editor')
    
    st.divider()
    graphandfitruby()
    

def graphandfitruby():
    st.subheader(":rainbow[Ruby Fluorescence file: Graph & Fitting  ]")        
    uploaded_file = st.file_uploader("Choose The Ruby (.txt) file", type = ['txt'])
    
    if uploaded_file is not None:
        data = load_file(uploaded_file)
        x=data[:,0]
        y=data[:,1]
        
        st.markdown(f"\n##### 💎 File Loaded: :red[{uploaded_file.name}]")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[:,0], y=data[:,1], mode='lines'))
        
        fig.update_layout(
            title = uploaded_file.name,
            margin=dict(l=40, r=40, t=40, b=40),

            #autosize = False,
            height = 600,
            xaxis_title = "nm",
            yaxis_title = "Counts",
            font=dict(
                family="Helvetica",
                size=28,
                color="RebeccaPurple"
            )
        )
        
        fig.update_xaxes(
            showline = True, linewidth = 2, linecolor = "white",
            showgrid = True,
            minor_showgrid = True,
            #font = dict(size = 18),
            mirror = True,
        )
        
        fig.update_yaxes(
            showline = True, linewidth = 2, linecolor = "white",
            showgrid = True,
            minor_showgrid = True,
            mirror = True,
        )
        
        st.plotly_chart(fig)
    

# Function to load and process the file
def load_file(file):
    # Load the file data skipping the header and footer lines
    data = np.genfromtxt(file, skip_header=17, skip_footer=1)
    return data