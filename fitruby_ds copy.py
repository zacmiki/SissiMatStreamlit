import streamlit as st
import plotly.graph_objects as go
import numpy as np
from lmfit.models import ConstantModel, DoniachModel
import matplotlib.pyplot as plt
from fitruby import load_rubyfile

# -------------------- Gaussian parameters definition 
def try_doniach(data, dm1_center, dm1_amplitude, dm1_sigma, dm2_center, dm2_amplitude, dm2_sigma):

    background = ConstantModel(prefix='bkg_')  # preparing the background parameter
    pars = background.guess(data[:, 1], x=data[:, 0])  # guessing the background for my data
    
    dm1 = DoniachModel(prefix='dm1_')
    pars.update(dm1.make_params())
    
    pars['dm1_center'].set(value=dm1_center, min=688, max=699)
    pars['dm1_sigma'].set(value=dm1_sigma, min=.01, max=2)
    pars['dm1_amplitude'].set(value=dm1_amplitude, min=200, max=60000)
    pars['dm1_gamma'].set(value=0, min=0, max=2)
    
    dm2 = DoniachModel(prefix='dm2_')
    pars.update(dm2.make_params())
    
    pars['dm2_center'].set(value=dm2_center, min=688, max=699)
    pars['dm2_sigma'].set(value=dm2_sigma, min=.02, max=2)
    pars['dm2_amplitude'].set(value=dm2_amplitude, min=200, max=50000)
    pars['dm2_gamma'].set(value=0, min=0, max=2)
    
    model = background + dm1 + dm2
    init = model.eval(pars, x= data[:, 0])
    return model, init, pars

# ----------- PAGE VISUALIZATION -----------
def rubyfitds():
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
            dm1_center = st.slider("DS-1 Center", 688.0, 710.0, 692.5, step=0.05)
    
        with col2:
            dm1_amplitude = st.slider("DS-1 Amplitude", 200, 50000, 4400, step=50)
    
        with col3:
            dm1_sigma = st.slider("DS-1 Sigma", 0.01, 2.0, 0.33, step=0.01)
        
        with col4:
            dm2_center = st.slider("DS-2 Center", 688.0, 710.0, 694.5, step=0.05)
        
        with col5:
            dm2_amplitude = st.slider("DS-2 Amplitude", 200, 50000, 7900, step=50)
        
        with col6:
            dm2_sigma = st.slider("DS-2 Sigma", 0.02, 2.0, 0.38, step=0.01)
    
        model, init, pars = try_doniach(data, dm1_center, dm1_amplitude, dm1_sigma, dm2_center, dm2_amplitude, dm2_sigma)

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
        
        if st.button("Fit by using current parameters"):
            result = model.fit(data[:,1], pars, x = data[:,0])
            comps = result.eval_components(x = data[:,0])
            init = result
            fig = go.Figure()
            #st.markdown(result.fit_report(min_correl=0.5))
            st.session_state.dm1_center = round(result.params['dm1_center'].value,4), 
            st.session_state.dm2_center = round(result.params['dm2_center'].value,4)
            
            st.markdown(f"Doniach_C1 = {st.session_state.dm1_center}")
            st.markdown(f"Doniach_C2 = {st.session_state.dm2_center}")
            
            x = data[:,0]
            y = data[:,1]
            
            fig, axes = plt.subplots(1, 2, figsize=(18, 9))
            axes[0].plot(x, y, ".", label='Data')
            axes[0].plot(x, result.best_fit, '-', label='best fit')
            axes[0].plot(x, y - result.best_fit, '--', label='Residuals')
            #axes[0].fill_between(x, result.best_fit,'-', label='best fit')


            axes[0].set_xlim(690,700)
            axes[0].grid(which = 'major', axis = 'y', linewidth = .2) 
            axes[0].grid(which = 'both', axis = 'x', lw = .2) 
            axes[0].legend()

            axes[1].plot(x, y - comps['bkg_'], ".")
            axes[1].plot(x, comps['dm1_'], '--', label='Doniach 1')
            axes[1].plot(x, comps['dm2_'], '--', label='Doniach 2')
            #axes[1].fill_between(x, comps['g1_'], '--', label='Gauss 1')
            #axes[1].fill_between(x, comps['g2_'], '--', label='Gauss 2', alpha = 0.5)

            axes[1].grid(which = 'major', axis = 'y', linewidth = .2) 
            axes[1].grid(which = 'both', axis = 'x', lw = .2) 
            axes[1].set_xlim(690,700)
            axes[1].legend()
            st.pyplot(fig)
            
            if st.button("Restart"):
                st.session_state.file_loaded = False
                data = load_rubyfile()
                st.rerun()
                                
            