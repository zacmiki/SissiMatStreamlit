import streamlit as st
import xarray as xr
import numpy as np
import plotly.graph_objects as go
import pandas as pd
#from scipy import sparse
#from scipy.sparse import linalg
#from numpy.linalg import norm
from sissi_xarrays import graphSSC_xArray, LoadSSC_xArray, cut_spectrum
from baselinesliders import baseline_arPLS
import os

# -----------------------------
st.set_page_config(layout="wide")

# ---------------- HELPER FUNCTIONS -----------------
def process_uploaded_files(uploaded_files, key):
    """
    Processes multiple uploaded OPUS files, computes average SSC values,
    and stores the result in session state.
    
    Args:
        uploaded_files (list): List of uploaded files
        key (str): Key identifier ('sample' or 'ref')
    """
    if uploaded_files and len(uploaded_files) > 0:
        # Store file names
        file_names = [file.name for file in uploaded_files]
        st.session_state[f"fileloaded_{key}"] = file_names
        
        datasets = []
        valid_files_count = 0
        
        for i, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            file_extension = file_name.split(".")[-1]
            
            if file_extension.isdigit():
                # Save temporary file
                temp_file = f"temp_{i}.opus"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load dataset
                try:
                    dataset = LoadSSC_xArray(temp_file)
                    datasets.append(dataset)
                    valid_files_count += 1
                except Exception as e:
                    st.error(f"Error loading {file_name}: {str(e)}")
                
                # Clean up temp file
                try:
                    os.remove(temp_file)
                except:
                    pass
            else:
                st.write(f":red[Skipping {file_name}: not a valid OPUS file (needs integer extension).]")
        
        if valid_files_count > 0:
            # Check if all datasets have the same x coordinates
            x_values = datasets[0].x.values
            all_same_x = all(np.allclose(ds.x.values, x_values) for ds in datasets)
            
            if not all_same_x:
                st.error("⚠️ Not all spectra have the same x coordinates. Cannot average.")
                return
            
            # Stack SSC values and compute average
            ssc_values = np.vstack([ds.ssc.values for ds in datasets])
            avg_ssc = np.mean(ssc_values, axis=0)
            
            # Create a new dataset with averaged SSC values
            avg_dataset = xr.Dataset(
                {'ssc': (('x',), avg_ssc)},
                coords={'x': x_values}
            )
            
            # Store the averaged dataset
            st.session_state[f"dataset_{key}"] = avg_dataset
            
            st.success(f"✅ Successfully averaged {valid_files_count} files for {key}!")
            
            # Return metadata for display
            return {
                'count': valid_files_count,
                'filenames': file_names,
                'ssc_shape': avg_ssc.shape
            }
        else:
            st.error("⚠️ No valid files were loaded.")
            
    return None

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
        st.session_state["normalized_spectrum"] = normalized_dataset
        st.session_state["cut_range"] = (normalized_dataset.x.values.min(), normalized_dataset.x.values.max())
        st.session_state["display_state"] = "normalized"
        st.success("✅ Normalization successful!")
    else:
        st.error("⚠️ Please upload both Sample and Reference Spectra before normalizing.")

def plot_normalized_spectrum(dataset):
    """
    Plot the normalized spectrum.
    
    Args:
        dataset (xr.Dataset): Normalized dataset with intensity values
        
    Returns:
        go.Figure: Plotly figure with the normalized spectrum
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dataset.x.values, y=dataset.intensity.values,
        mode='lines',
        name='Normalized Spectrum'
    ))
    
    fig.update_layout(
        title='Normalized Spectrum',
        xaxis_title='Wavenumber (cm⁻¹)',
        yaxis_title='Normalized Intensity',
        autosize=True,
        height=450
    )
    
    fig.update_xaxes(showline=True, linewidth=1, linecolor="white", showgrid=True, mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="white", showgrid=True, mirror=True)
    
    return fig

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
    
    st.markdown(
        '''
        <div style="
            background-color:#183927; 
            padding:20px; 
            border-radius:5px; 
            display: inline-block; 
            width: fit-content;
            max-width: 80%; /* Adjust width limit if needed */
        ">
            <b>Before Starting each new Preanalysis,</b>
            <b><font color="yellow">it is advisable for you to RESET THE CACHE</b><br></font>
            and remove the previously loaded files.
        </div>
        ''', 
        unsafe_allow_html=True
    )
    
    st.divider()
    
    if st.button("Reset All"):
        st.session_state.clear()
        st.rerun()
    
    col1, col2 = st.columns(2)  # Two side-by-side columns

    with col1:
        st.subheader("Sample Spectra")
        sample_spectra = st.file_uploader("Choose OPUS files", 
                                          key="dataspectra", 
                                          accept_multiple_files=True,
                                          type=None)  # No type restriction, we'll check extensions
        
        if sample_spectra:
            sample_info = process_uploaded_files(sample_spectra, "sample")
            if sample_info:
                st.info(f"Loaded {sample_info['count']} sample files. Computed average spectrum.")

    with col2:
        st.subheader("Background Spectra")
        ref_spectra = st.file_uploader("Choose OPUS files", 
                                       key="refspectra", 
                                       accept_multiple_files=True,
                                       type=None)
        
        if ref_spectra:
            ref_info = process_uploaded_files(ref_spectra, "ref")
            if ref_info:
                st.info(f"Loaded {ref_info['count']} reference files. Computed average spectrum.")

    # ---------------- SHOW SPECTRA -----------------
    col1, col2 = st.columns(2)
    if "dataset_sample" in st.session_state:
        with col1:
            graphSSC_xArray(st.session_state["dataset_sample"], "Averaged Sample Spectrum")
    
    if "dataset_ref" in st.session_state:
        with col2:
            graphSSC_xArray(st.session_state["dataset_ref"], "Averaged Reference Spectrum")

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

        # Apply cutting dynamically to create cut_dataset
        cut_dataset = cut_spectrum(st.session_state["normalized_spectrum"], *cut_range)
        
        # Display the cut normalized spectrum
        normalized_fig = plot_normalized_spectrum(cut_dataset)
        st.plotly_chart(normalized_fig, use_container_width=True)
        
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
                # Use first filename from the list for naming
                if isinstance(st.session_state['fileloaded_sample'], list) and len(st.session_state['fileloaded_sample']) > 0:
                    base_filename = st.session_state['fileloaded_sample'][0].split('.')[0]
                    if len(st.session_state['fileloaded_sample']) > 1:
                        base_filename += f"_avgd_bkgsubtracted"
                else:
                    base_filename = "averaged_spectrum"
                
                filename = f"{base_filename}_bkgsubtracted.csv"
                
                # Include averaged SSC data in the saved file
                sample_ssc = st.session_state["dataset_sample"].ssc.values
                ref_ssc = st.session_state["dataset_ref"].ssc.values
                
                # Create DataFrame with all data
                baseline_df = pd.DataFrame({
                    'x': x_values, 
                    'sample_ssc': sample_ssc if len(sample_ssc) == len(x_values) else np.interp(x_values, st.session_state["dataset_sample"].x.values, sample_ssc),
                    'reference_ssc': ref_ssc if len(ref_ssc) == len(x_values) else np.interp(x_values, st.session_state["dataset_ref"].x.values, ref_ssc),
                    'intensity': intensity_values,
                    'baseline': baseline,
                    'intensity_bcksubtr': baseline_subtracted
                })

                # Create a download link instead of saving to disk
                csv = baseline_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV File",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )

# Call the main function (can be replaced with `if __name__ == "__main__":`
# if this is imported as a module)
online_analysis()
