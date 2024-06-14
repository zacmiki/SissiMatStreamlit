import streamlit as st
import plotly.graph_objects as go
import numpy as np
from lmfit.models import ConstantModel, GaussianModel

# -----------------File Loading ------------------------------
def load_rubyfile():
    uploaded_file = st.file_uploader("Choose The Ruby (.txt) file", type=['txt'])
    
    if uploaded_file is not None:
        # Load the file data skipping the header and footer lines
        data = np.genfromtxt(uploaded_file, skip_header=17, skip_footer=1)
        st.session_state.data = data
        
        x = data[:, 0]
        y = data[:, 1]
        st.markdown(f"\n##### 💎 File Loaded: :red[{uploaded_file.name}]")
        st.session_state.file_loaded = True
        return data

# -------------------- Gaussian parameters definition 
def try_gaussian(data, g1_center, g1_amplitude, g1_sigma, g2_center, g2_amplitude, g2_sigma):
    background = ConstantModel(prefix='bkg_')  # preparing the background parameter
    pars = background.guess(data[:, 1], x=data[:, 0])  # guessing the background for my data
    
    gauss1 = GaussianModel(prefix='g1_')
    pars.update(gauss1.make_params())
    
    pars['g1_center'].set(value=g1_center, min=688, max=699)
    pars['g1_sigma'].set(value=g1_sigma, min=.01, max=2)
    pars['g1_amplitude'].set(value=g1_amplitude, min=1000, max=60000)
    
    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    
    pars['g2_center'].set(value=g2_center, min=688, max=699)
    pars['g2_sigma'].set(value=g2_sigma, min=.02, max=2)
    pars['g2_amplitude'].set(value=g2_amplitude, min=1000, max=50000)
    
    model = background + gauss1 + gauss2
    init = model.eval(pars, x=data[:, 0])
    return model, init

# ----------- PAGE VISUALIZATION -----------
def rubyfit():
    st.title("Load and Fit the Ruby File")
    # Initialize the file_loaded state if it doesn't exist
    
    if 'file_loaded' not in st.session_state:
        st.session_state.file_loaded = False
        
    # Show file uploader only if the file is not loaded yet
    if not st.session_state.file_loaded:
        data = load_rubyfile()
    else:
        data = st.session_state.data

    if st.session_state.file_loaded and data is not None:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
    
        with col1:
            g1_center = st.slider("G-1 Center", 688.0, 710.0, 692.5, step=0.05)
    
        with col2:
            g1_amplitude = st.slider("G-1 Amplitude", 500, 50000, 4400, step=50)
    
        with col3:
            g1_sigma = st.slider("G-1 Sigma", 0.01, 2.0, 0.33, step=0.01)
        
        with col4:
            g2_center = st.slider("G-2 Center", 688.0, 710.0, 694.5, step=0.05)
        
        with col5:
            g2_amplitude = st.slider("G-2 Amplitude", 500, 50000, 7900, step=50)
        
        with col6:
            g2_sigma = st.slider("G-2 Sigma", 0.02, 2.0, 0.38, step=0.01)
    
        model, init = try_gaussian(data, g1_center, g1_amplitude, g1_sigma, g2_center, g2_amplitude, g2_sigma)

        fig = go.Figure()

        # Scatter plot of the data
        fig.add_trace(go.Scatter(
            x=data[:, 0], y=data[:, 1],
            mode='markers',
            name='Data',
            showlegend=False
        ))

        # Adding the line graph of (data[:, 0], init) in black
        fig.add_trace(go.Scatter(
            x=data[:, 0], y=init,
            mode='lines',
            line=dict(color='red'),
            name='Gaussian Fit',
            showlegend=False
        ))

        # Set x-axis range manually
        x_range = st.slider("X-axis range", float(data[:, 0].min()), float(data[:, 0].max()), (float(data[:, 0].min()), float(data[:, 0].max())))
        fig.update_layout(xaxis=dict(range=x_range))

        st.plotly_chart(fig, use_container_width=True)