import streamlit as st
import plotly.graph_objects as go
import numpy as np
from lmfit.models import ConstantModel, VoigtModel
import matplotlib.pyplot as plt
from fitruby import load_rubyfile

# -------------------- Gaussian parameters definition 
def try_voigt(data, v1_center, v1_amplitude, v1_sigma, v2_center, v2_amplitude, v2_sigma):

    background = ConstantModel(prefix='bkg_')  # preparing the background parameter
    pars = background.guess(data[:, 1], x=data[:, 0])  # guessing the background for my data
    
    voigt1 = VoigtModel(prefix='v1_')
    pars.update(voigt1.make_params())
    
    pars['v1_center'].set(value=v1_center, min=688, max=699)
    pars['v1_sigma'].set(value=v1_sigma, min=.01, max=2)
    pars['v1_amplitude'].set(value=v1_amplitude, min=1000, max=60000)
    
    voigt2 = VoigtModel(prefix='v2_')
    pars.update(voigt2.make_params())
    
    pars['v2_center'].set(value=v2_center, min=688, max=699)
    pars['v2_sigma'].set(value=v2_sigma, min=.02, max=2)
    pars['v2_amplitude'].set(value=v2_amplitude, min=1000, max=50000)
    
    model = background + voigt1 + voigt2
    init = model.eval(pars, x=data[:, 0])
    return model, init, pars

# ----------- PAGE VISUALIZATION -----------
def rubyfit_v():
    st.title(":rainbow[Load and Fit the Ruby File (Voigt)]")
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
            v1_center = st.slider("Voigt1 Center", 688.0, 710.0, 
            float(st.session_state.get('v1_center', 692.5)), step=0.05)
    
        with col2:
            v1_amplitude = st.slider("Voigt1 Amplitude", 500, 50000, 
            int(st.session_state.get('v1_amplitude', 4400)), step=50)
    
        with col3:
            v1_sigma = st.slider("Voigt Sigma", 0.01, 2.0,
            float(st.session_state.get('v1_sigma', 0.33)), step=0.01)
        
        with col4:
            v2_center = st.slider("Voigt2 Center", 688.0, 710.0,
            float(st.session_state.get('v2_center', 694.5)), step=0.05)
        
        with col5:
            v2_amplitude = st.slider("Voigt Amplitude", 500, 50000,
            int(st.session_state.get('v2_amplitude', 7900)), step=50)
        
        with col6:
            v2_sigma = st.slider("Voigt Sigma", 0.02, 2.0,
            float(st.session_state.get('v2_sigma', 0.38)), step=0.01)
    
        model, init, pars = try_voigt(data, v1_center, v1_amplitude, v1_sigma, v2_center, v2_amplitude, v2_sigma)

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
            st.session_state.v1_center = float(result.params['v1_center'].value)
            st.session_state.v1_amplitude = float(result.params['v1_amplitude'].value)
            st.session_state.v1_sigma = float(result.params['v1_sigma'].value)
            st.session_state.v2_center = float(result.params['v2_center'].value)
            st.session_state.v2_amplitude = float(result.params['v2_amplitude'].value)
            st.session_state.v2_sigma = float(result.params['v2_sigma'].value)
            
            v1_centroid = round(result.params['v1_center'].value,4), 
            v2_centroid = round(result.params['v2_center'].value,4)
            rsquared = round(1 - (result.residual.var() / np.var(data[:, 1])),4)
            
            st.markdown(f"\nVoigt1_Center = {v1_centroid}")
            st.markdown(f"\nVoigt2_Center = {v2_centroid}")
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
            axes[1].fill_between(x, comps['v1_'], '--', label='Gauss 1')
            axes[1].fill_between(x, comps['v2_'], '--', label='Gauss 2', alpha = 0.5)

            axes[1].grid(which = 'major', axis = 'y', linewidth = .2) 
            axes[1].grid(which = 'both', axis = 'x', lw = .2) 
            axes[1].set_xlim(690,705)
            axes[1].legend()
            st.pyplot(fig)
                                