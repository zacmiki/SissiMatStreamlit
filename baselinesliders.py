import ipywidgets as widgets
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

# -------------------------------------------------------------
def cut_spectrum(dataset, begin, end):
    import xarray as xr
    datasetcut = dataset.sel(x = slice(begin,end))

    return datasetcut
# -------------------------------------------------------------

def baseline_arPLS(y, ratio=1e-6, lam=100, niter=10, full_output=False):
    from scipy import sparse
    from scipy.sparse import linalg
    import numpy as np
    from numpy.linalg import norm
    
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252 -- Analyst, 2015, 140, 250 ---

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
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values

        count += 1

        if count > niter:
            print('Maximum number of iterations exceeded')
            break

    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return z
    
''' ------------------------------------------------------ '''
# -------------------------------------------------------------

def LoadSSC_xArray(filepath):
    import xarray as xr
    loaded = loadSSC(filepath)
    
    dataset = xr.Dataset(
    {
        "ssc": (["x"], loaded.y),  # I put as Y only the SSC Spectrum
    },
    coords ={"x": loaded.x} # I set as Coordinate only the X containing the wavenumers
    )
    return dataset

def normalize_Opus_Datasets(data, background):
    data["intensity"] = (["x"], data.ssc.values / background.ssc.values) # performing the normalization
    return

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

def AverageSSC_xArray(*filepaths):
    datasets = [LoadSSC_xArray(fp) for fp in filepaths]  # Load all datasets
    
    # Ensure all datasets have the same 'x' values before averaging
    x_values = datasets[0]['x']
    for ds in datasets:
        if not (ds['x'].equals(x_values)):
            raise ValueError("Mismatch in 'x' coordinate values across datasets.")
    
    # Stack along a new dimension and compute the mean along that dimension
    combined = xr.concat(datasets, dim='dataset')
    avg_ssc = combined['ssc'].mean(dim='dataset')
    
    # Create a new dataset with the same 'x' and the averaged 'ssc'
    avg_dataset = xr.Dataset({
        'ssc': avg_ssc
    }, coords={'x': x_values})
    
    return avg_dataset


def savespectrum(dataset: xr.Dataset, filename: str = "spectrum.csv") -> str:
    """
    Saves an xarray Dataset to a tab-delimited CSV file.
    
    Parameters:
    dataset (xr.Dataset): The input xarray dataset.
    filename (str): The name of the output CSV file. Default is 'spectrum.csv'.
    
    Returns:
    str: The filename of the saved CSV file.
    """
    # Convert dataset to DataFrame
    df = dataset.to_dataframe()
    
    # Save as a tab-delimited CSV file
    df.to_csv(filename, sep='\t', index=True)  # Keeping index (x values)
    
    return filename


def arpls_subtraction(spectrum, data = "intensity"):
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    """
    Create an interactive plot with sliders to adjust arPLS baseline subtraction parameters.
    The background-subtracted spectrum is added to the dataset.
    
    Parameters:
    -----------
    spectrum : xarray.Dataset
        The dataset containing spectral data.
    data : str or xarray.DataArray
        The name of the data variable (as a string) or the DataArray itself.
    baseline_arPLS_function : function
        The function to calculate the baseline.
    """
    
    # If data is given as a DataArray, extract its name; otherwise, fetch it from the dataset
    if isinstance(data, str):
        data_variable_name = data
        data_array = spectrum[data]
    else:
        data_array = data
        data_variable_name = data_array.name  # Get the variable name dynamically
    
    # Create logarithmic sliders for ratio and lambda
    ratio_slider = widgets.FloatLogSlider(
        value=1e-3,
        base=10,
        min=-6,  # 10^-6
        max=-1,  # 10^-1
        step=0.1,
        description='Ratio:',
        readout_format='.2e'
    )
    
    lambda_slider = widgets.FloatLogSlider(
        value=1e9,
        base=10,
        min=3,  # 10^1
        max=13,  # 10^11
        step=0.1,
        description='Lambda:',
        readout_format='.2e'
    )
    
    # Create integer slider for niter
    niter_slider = widgets.IntSlider(
        value=30,
        min=10,
        max=100,
        step=1,
        description='N. Iterations:',
        continuous_update=False
    )
    
    # Button for saving the current background subtraction 
    save_button = widgets.Button(
        description='Save Subtracted',
        button_style='success',
        tooltip='Save the current background-subtracted spectrum to spectrum["bck_subtracted"]'
    )
    
    # Output widget for the plot
    output = widgets.Output()
    
    # Storage for current background values
    current_bkg = [None]
    current_subtracted = [None]
    
    # Function to update the plot
    def update_plot(ratio, lam, niter):
        with output:
            output.clear_output(wait=True)
            
            # Calculate background
            bkg = baseline_arPLS_function(data_array, lam=lam, ratio=ratio, niter=niter)
            subtracted = data_array - bkg
            
            # Store current values for save button
            current_bkg[0] = bkg
            current_subtracted[0] = subtracted
            
            # Create plot
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(spectrum.x.values, data_array, label="Acquired")
            ax.plot(spectrum.x.values, bkg, label=f"Background - arPLS (λ={lam:.2e}, ratio={ratio:.2e}, iter={niter})")
            #ax.plot(spectrum.x.values, subtracted, label="Background Subtracted")
            ax.plot(spectrum.x.values, current_subtracted[0], label="Background Subtracted")
            
            # Add horizontal line at y=0
            ax.axhline(y=0, linestyle="--", linewidth=0.7, color='gray')
            
            # Add grid only for x-axis
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            ax.legend()
            ax.set_title("Baseline Subtraction with arPLS (adaptive regularized Penalized Least Squares)")
            plt.tight_layout()
            plt.show()
    
    # Function to save the background-subtracted spectrum
    def save_subtracted(b):
        if current_subtracted[0] is not None:
            # Ensure background-subtracted data is saved in the dataset
            spectrum["bck_subtracted"] = (["x"], current_subtracted[0].values)
                
            # Provide feedback
            with output:
                output.clear_output(wait=True)
                print('✅ Background-subtracted spectrum saved in `spectrum["bck_subtracted"]`.')

            #update_plot(ratio_slider.value, lambda_slider.value, niter_slider.value)


    # Function to handle slider changes
    def on_slider_change(change):
        update_plot(ratio_slider.value, lambda_slider.value, niter_slider.value)
    
    # Connect the sliders to the update function
    ratio_slider.observe(on_slider_change, names='value')
    lambda_slider.observe(on_slider_change, names='value')
    niter_slider.observe(on_slider_change, names='value')
    
    # Connect save button
    save_button.on_click(save_subtracted)
    
    # Layout widgets
    sliders = widgets.VBox([ratio_slider, lambda_slider, niter_slider, save_button])
    app = widgets.VBox([sliders, output])
    
    # Initialize the plot
    update_plot(ratio_slider.value, lambda_slider.value, niter_slider.value)
    
    # Display the app
    display(app)

    