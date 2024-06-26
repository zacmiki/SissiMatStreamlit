import streamlit as st
import plotly.graph_objects as go
import numpy as np
from lmfit.models import ConstantModel, GaussianModel
import matplotlib.pyplot as plt

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
    return model, init, pars

# ----------- PAGE VISUALIZATION -----------
def rubyfit():
    st.title(":rainbow[Load and Fit the Ruby File]")
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
            g1_center = st.slider("G-1 Center", 688.0, 710.0, 
            float(st.session_state.get('g1_center', 692.5)), step=0.05)
    
        with col2:
            g1_amplitude = st.slider("G-1 Amplitude", 500, 50000, 
            int(st.session_state.get('g1_amplitude', 4400)), step=50)
    
        with col3:
            g1_sigma = st.slider("G-1 Sigma", 0.01, 2.0,
            float(st.session_state.get('g1_sigma', 0.33)), step=0.01)
        
        with col4:
            g2_center = st.slider("G-2 Center", 688.0, 710.0,
            float(st.session_state.get('g2_center', 694.5)), step=0.05)
        
        with col5:
            g2_amplitude = st.slider("G-2 Amplitude", 500, 50000,
            int(st.session_state.get('g2_amplitude', 7900)), step=50)
        
        with col6:
            g2_sigma = st.slider("G-2 Sigma", 0.02, 2.0,
            float(st.session_state.get('g2_sigma', 0.38)), step=0.01)
    
        model, init, pars = try_gaussian(data, g1_center, g1_amplitude, g1_sigma, g2_center, g2_amplitude, g2_sigma)

        fig = go.Figure()
        
        fig.update_layout(
        height = 600,
        )

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
        
        if st.button("Fit by using current parameters"):
            result = model.fit(data[:,1], pars, x = data[:,0])
            comps = result.eval_components(x = data[:,0])
            init = result
            fig = go.Figure()
            #st.markdown(result.fit_report(min_correl=0.5))
            
            # Update the sliders with the fitted values
            st.session_state.g1_center = float(result.params['g1_center'].value)
            st.session_state.g1_amplitude = float(result.params['g1_amplitude'].value)
            st.session_state.g1_sigma = float(result.params['g1_sigma'].value)
            st.session_state.g2_center = float(result.params['g2_center'].value)
            st.session_state.g2_amplitude = float(result.params['g2_amplitude'].value)
            st.session_state.g2_sigma = float(result.params['g2_sigma'].value)
            
            g1_centroid = round(result.params['g1_center'].value,4), 
            g2_centroid = round(result.params['g2_center'].value,4)
            rsquared = round(1 - (result.residual.var() / np.var(data[:, 1])),4)
            
            st.markdown(f"\nGauss1_Center = {g1_centroid}")
            st.markdown(f"\nGauss2_Center = {g2_centroid}")
            st.markdown(f"\nThe R-Squared is {rsquared}")
            
            x = data[:,0]
            y = data[:,1]
            
            fig, axes = plt.subplots(1, 2, figsize=(18, 9))
            axes[0].plot(x, y, ".", label='Data')
            axes[0].plot(x, result.best_fit, '-', label='best fit')
            axes[0].plot(x, y - result.best_fit, '--', label='Residuals')
            #axes[0].fill_between(x, result.best_fit,'-', label='best fit')


            axes[0].set_xlim(690,705)
            axes[0].grid(which = 'major', axis = 'y', linewidth = .2) 
            axes[0].grid(which = 'both', axis = 'x', lw = .2) 
            axes[0].legend()

            axes[1].plot(x, y - comps['bkg_'], ".")
            #axes[1].plot(x, comps['dm1_'], '--', label='Doniach 1')
            #axes[1].plot(x, comps['dm2_'], '--', label='Doniach 2')
            axes[1].fill_between(x, comps['g1_'], '--', label='Gauss 1')
            axes[1].fill_between(x, comps['g2_'], '--', label='Gauss 2', alpha = 0.5)

            axes[1].grid(which = 'major', axis = 'y', linewidth = .2) 
            axes[1].grid(which = 'both', axis = 'x', lw = .2) 
            axes[1].set_xlim(690,705)
            axes[1].legend()
            st.pyplot(fig)
                                
