import streamlit as st
import numpy as np
import xarray as xr
import plotly.graph_objects as go


def compute_absorbance(spectrum: xr.Dataset, data: str):
    """
    Compute the absorbance from a given spectral dataset.

    Parameters:
    -----------
    spectrum : xarray.Dataset
        The dataset containing spectral data.
    data : str
        The name of the data variable in `spectrum` to compute absorbance from.
        
        # Perform the logarithmic transformation on the intensity variable (handling zero values)
        #absorbance = - np.log(spectrum["bck_subtracted"+5].where(spectrum["bck_subtracted"] > 0))

   """

    # Verify that `data` exists in `spectrum`
    if data not in spectrum.data_vars:
        print(f"Warning: '{data}' is not a valid data variable in the provided spectrum dataset.")
        print(f"Available variables: {list(spectrum.data_vars.keys())}")
        return  # Exit without raising an error

    # Retrieve the data array
    data_array = spectrum[data]

    # Compute absorbance
    if data == "intensity":
        absorbance = -np.log(data_array)
    else:
        absorbance = -np.log(data_array + np.abs(np.min(data_array)))
    
    # Store absorbance in the dataset
    spectrum["absorbance"] = absorbance

    return

def graphSSC_xArray(dataset, title):
    """Displays a graph of the dataset."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataset.x.values, y=dataset.ssc.values, mode="lines"))

    fig.update_layout(
        title_text=title,
        margin=dict(l=40, r=40, t=40, b=40),
        autosize=True,
        height=400,
    )
    fig.update_xaxes(showline=True, linewidth=2, linecolor="white", showgrid=True, mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor="white", zeroline=True, zerolinecolor="brown", showgrid=True, mirror=True)

    st.plotly_chart(fig, clear_figure=True)  # Clears previous graph before displaying a new one
    
def LoadSSC_xArray(filepath):
    loaded = loadSSC(filepath)
    
    dataset = xr.Dataset(
    {
        "ssc": (["x"], loaded.y),  # I put as Y only the SSC Spectrum
    },
    coords ={"x": loaded.x} # I set as Coordinate only the X containing the wavenumers
    )
    return dataset

# --------------------------------------

def loadSSC(fileName):
    import opusFC

    dbs = opusFC.listContents(fileName)
#    print(f"File {os.path.basename(fileName)} Loaded {dbs}")

    for item in range(len(dbs)):
        if (dbs[item][0]) == 'SSC':
            data = opusFC.getOpusData(fileName, dbs[item])

    return data

# -------------------------------------

def cut_spectrum(dataset, begin, end):
    """
    Dynamically updates the spectrum cut range.
    Ensures the 'intensity' variable is also cut.
    
    Args:
        dataset (xr.Dataset): Input dataset
        begin (float): Start of cut range
        end (float): End of cut range
    
    Returns:
        xr.Dataset: Cut dataset
    """
    if dataset is not None:
        # Cut the dataset
        cut_dataset = dataset.sel(x=slice(begin, end))
        
        return cut_dataset
    return None

