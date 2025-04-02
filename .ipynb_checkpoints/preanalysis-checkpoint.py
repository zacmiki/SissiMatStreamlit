import streamlit as st
import xarray as xr
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg
from numpy.linalg import norm
from sissi_xarrays import graphSSC_xArray, LoadSSC_xArray, cut_spectrum

st.set_page_config(layout="wide")

# ---------------- BASELINE SUBTRACTION FUNCTION -----------------
def baseline_arPLS(y, ratio=1e-6, lam=1e03, niter=30, full_output=False):
    """
    Adaptive Robust Penalized Least Squares (arPLS) baseline estimation.
    
    Args:
        y (np.ndarray): Input signal
        ratio (float): Convergence criterion
        lam (float): Smoothness parameter
        niter (int): Maximum number of iterations
        full_output (bool): Return additional information if True
    
    Returns:
        np.ndarray or tuple: Estimated baseline or (baseline, residual, info)
    """
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)  # Smoothness matrix

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))

        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)  # Update diagonal values

        count += 1

        if count > niter:
            st.warning('Maximum number of iterations exceeded')
            break

    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return z

# ---------------- HELPER FUNCTIONS -----------------
def process_uploaded_file(uploaded_file, key):
    """Processes an uploaded OPUS file and stores the dataset in session state."""
    if uploaded_file is not None:
        file_name = uploaded_file.name
        st.session_state[f"fileloaded_{key}"] = file_name
        file_extension = file_name.split(".")[-1]
        
        if file_extension.isdigit():
            with open("temp.opus", "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            dataset = LoadSSC_xArray("temp.opus")
            st.session_state[f"dataset_{key}"] = dataset  # Store dataset
        else:
            st.write(":red[Please upload a file with an integer extension.]")

def normalize_Opus_Datasets(sample, reference):
    """
    Performs normalization: sample spectrum divided by reference spectrum.
    Creates a new xarray dataset with normalized data.
    
    Args:
        sample (xr.Dataset): Sample spectrum dataset
        reference (xr.Dataset): Reference spectrum dataset
    """
    if sample is not None and reference is not None:
        # Check if x coordinates match
        if not np.allclose(sample.x.values, reference.x.values):
            st.error("⚠️ Sample and Reference spectra have different x coordinates.")
            return
        
        # Create normalized data
        normalized_values = sample.ssc.values / reference.ssc.values
        
        # Create a new dataset with normalized data
        normalized_dataset = xr.Dataset(
            {
                'ssc': (('x',), sample.ssc.values),  # Original sample SSC
                'intensity': (('x',), normalized_values)  # Normalized data
            },
            coords={'x': sample.x.values}
        )
        
        # Store the normalized dataset
        st.session_state["dataset_sample"] = normalized_dataset
        st.session_state["normalized_spectrum"] = normalized_dataset
        st.session_state["cut_range"] = (normalized_dataset.x.values.min(), normalized_dataset.x.values.max())
        st.session_state["display_state"] = "normalized"
        st.success("✅ Normalization successful!")
    else:
        st.error("⚠️ Please upload both Sample and Reference Spectra before normalizing.")

def save_spectrum(dataset, filename):
    """
    Saves the xarray dataset to a CSV file.
    Includes x, ssc, and intensity if available.
    
    Args:
        dataset (xr.Dataset): Input dataset
        filename (str): Output filename
    """
    if dataset is not None:
        # Prepare data for CSV
        data = {'x': dataset.x.values, 'ssc': dataset.ssc.values}
        
        # Add intensity if it exists
        if 'intensity' in dataset:
            data['intensity'] = dataset.intensity.values
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        st.success(f"✅ File saved as {filename}")

def plot_baseline_subtraction(x, intensity, baseline, baseline_subtracted):
    """
    Creates a plotly figure for baseline subtraction visualization.
    
    Args:
        x (np.ndarray): X-axis values
        intensity (np.ndarray): Original intensity values
        baseline (np.ndarray): Estimated baseline
        baseline_subtracted (np.ndarray): Baseline subtracted signal
    
    Returns:
        go.Figure: Plotly figure with multiple traces
    """
    fig = go.Figure()
    
    # Original signal
    fig.add_trace(go.Scatter(
        x=x, y=intensity, 
        mode='lines', 
        name='Original Signal'
    ))
    
    # Baseline
    fig.add_trace(go.Scatter(
        x=x, y=baseline, 
        mode='lines', 
        name='Estimated Baseline', 
        line=dict(color='red', dash='dot')
    ))
    
    # Baseline subtracted signal
    fig.add_trace(go.Scatter(
        x=x, y=baseline_subtracted, 
        mode='lines', 
        name='Baseline Subtracted', 
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title='Baseline Subtraction',
        xaxis_title='Wavenumber (cm⁻¹)',
        margin=dict(l=40, r=40, t=60, b=40),  # Adjusted margins
        autosize=True,  # Allow auto-sizing
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=1.02,  # Position above the chart
            xanchor="right",
            x=1
        ),
        height=450  # Slightly reduced height
    )
    
    fig.update_xaxes(showline=True, linewidth=1, linecolor="white", showgrid=True, mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="white", zeroline=True, zerolinecolor="brown", showgrid=True, mirror=True)
    
    return fig

# ---------------- MAIN FUNCTION -----------------
def online_analysis():
    
    if st.button("Reset All"):
        st.session_state.clear()
        st.rerun()
    
    col1, col2 = st.columns(2)  # Two side-by-side columns

    with col1:
        st.subheader("Sample Spectrum")
        sample_spectrum = st.file_uploader("Choose an OPUS file", key="dataspectrum")
        process_uploaded_file(sample_spectrum, "sample")

    with col2:
        st.subheader("Background Spectrum")
        ref_spectrum = st.file_uploader("Choose an OPUS file", key="refspectrum")
        process_uploaded_file(ref_spectrum, "ref")

    # ---------------- SHOW RAW SPECTRA -----------------
    col1, col2 = st.columns(2)
    if "dataset_sample" in st.session_state:
        with col1:
            graphSSC_xArray(st.session_state["dataset_sample"], "Sample Spectrum")
    
    if "dataset_ref" in st.session_state:
        with col2:
            graphSSC_xArray(st.session_state["dataset_ref"], "Reference Spectrum")

    # ---------------- ACTION BUTTON -----------------
    if "dataset_sample" in st.session_state and "dataset_ref" in st.session_state:
        st.divider()  # Visual separator
        if st.button("🔄 Normalize"):
            normalize_Opus_Datasets(st.session_state["dataset_sample"], st.session_state["dataset_ref"])

    # ---------------- DISPLAY NORMALIZED SPECTRUM & CUTTING -----------------
    if st.session_state.get("display_state") == "normalized":
        st.subheader("Normalized Spectrum")
        
        # Initialize range for cutting
        min_x, max_x = st.session_state["cut_range"]
        
        # Allow user to select cutting method
        cut_method = st.radio("Choose Cut Range Method:", ["Slider", "Manual Entry"], horizontal=True)
        
        if cut_method == "Slider":
            # Slider for cutting range selection
            cut_range = st.slider(
                "Select Cut Range (cm⁻¹)", 
                min_value=int(min_x), max_value=int(max_x), 
                value=(int(min_x), int(max_x)), step=10
            )
        else:
            # Manual entry for cutting range
            col1, col2 = st.columns(2)
            with col1:
                cut_min = st.number_input("Min Cut Value (cm⁻¹)", 
                                         min_value=int(min_x), 
                                         max_value=int(max_x),
                                         value=int(min_x))
            with col2:
                cut_max = st.number_input("Max Cut Value (cm⁻¹)", 
                                         min_value=int(min_x), 
                                         max_value=int(max_x),
                                         value=int(max_x))
            cut_range = (cut_min, cut_max)

        # Apply cutting dynamically
        cut_dataset = cut_spectrum(st.session_state["normalized_spectrum"], *cut_range)
        
        # Baseline Subtraction Section
        st.subheader("Baseline Subtraction")
        
        # Parameter sliders for baseline subtraction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lam = st.slider(
                "Smoothness (λ)", 
                min_value=1e03, max_value=1e10, 
                value=1e05, format='%e', step=1e03
            )
        
        with col2:
            ratio = st.slider(
                "Convergence Ratio", 
                min_value=0.0000000001, max_value=0.1, 
                value=1e-6, format='%e'
            )
        
        with col3:
            niter = st.slider(
                "Max Iterations", 
                min_value=5, max_value=50, 
                value=10, step=1
            )
        
        # Perform baseline subtraction
        x_values = cut_dataset.x.values
        intensity_values = cut_dataset.intensity.values
        
        baseline = baseline_arPLS(
            intensity_values, 
            ratio=ratio, 
            lam=lam, 
            niter=niter
        )
        
        baseline_subtracted = intensity_values - baseline
        
        # Visualize baseline subtraction
        baseline_fig = plot_baseline_subtraction(
            x_values, 
            intensity_values, 
            baseline, 
            baseline_subtracted
        )
        
        st.plotly_chart(baseline_fig, use_container_width=True)
        
        # Save button for baseline subtracted spectrum
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Baseline Subtracted Spectrum"):
                filename = f"baseline_subtracted_{st.session_state['fileloaded_sample'].split('.')[0]}.csv"
                baseline_df = pd.DataFrame({
                    'x': x_values, 
                    'original_intensity': intensity_values,
                    'baseline': baseline,
                    'baseline_subtracted': baseline_subtracted
                })
                baseline_df.to_csv(filename, index=False)
                st.success(f"✅ File saved as {filename}")
