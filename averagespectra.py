import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from ringparameters import *
from sissi_util import loadSSC
import numpy as np
import io


def load_opus_data(file_path):
    # Implement the function to load data from OPUS file
    # Assuming it returns a dictionary with 'x' and 'y' keys
    try:
        data = loadSSC(
            file_path
        )  # Update this according to your function's actual return type
        if data is not None and hasattr(data, "x") and hasattr(data, "y"):
            return data
        else:
            return None
    except Exception as e:
        st.error(f"Error loading OPUS data from {file_path}: {e}")
        return None


def get_elettra_status():
    if st.button(f"""\n\n## GET ELETTRA STATUS"""):
        st.markdown(
            f"<h1 style = 'text-align: center; color: grey;'>Machine Status<br> {get_machine_status()}</h1>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"\n##### Ring Energy = {get_energy_value()}")
        with col2:
            st.info(f"\n##### Ring Current = {get_current_value()}")


def averaging():
    st.session_state.fileloaded = ""
    fileloaded = ""
    first_file_name = "averaged_spectrum"  # Default

    with st.form("my-form", clear_on_submit=True):
        path = st.file_uploader(
            "Choose the OPUS Files to Average\n  :red[The files must have the same number of datapoints]",
            accept_multiple_files=True,
            label_visibility="visible",
        )

        submitted = st.form_submit_button("submit")

    if path:
        y_values_sum = None
        x_values = None
        num_files = 0
        spectra_data = []  # Store individual spectra for plotting

        for uploaded_file in path:
            if uploaded_file is not None:
                file_name = uploaded_file.name

                st.session_state.fileloaded = file_name
                file_extension = file_name.split(".")[-1]
                if file_extension.isdigit():
                    with open("temp.opus", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        data = load_opus_data("temp.opus")
                        if data is not None:
                            if y_values_sum is None:
                                # Initialize the sum with the first file's y-values
                                y_values_sum = np.zeros_like(data.y)
                                x_values = data.x
                                first_file_name = file_name.split(".")[0]
                            else:
                                # Check if x values match
                                if not np.allclose(x_values, data.x):
                                    st.warning(
                                        f"⚠️ File {file_name} has different x values - skipping"
                                    )
                                    continue
                            y_values_sum += data.y
                            num_files += 1
                            spectra_data.append(
                                {"x": data.x, "y": data.y, "name": file_name}
                            )
                        else:
                            st.error(f"Failed to load data from {file_name}.")
                else:
                    st.write("Please choose files with integer extensions.")
            else:
                st.write("Please choose only OPUS files.")

        if num_files > 0 and y_values_sum is not None:
            # Compute the average
            averaged_spectrum = y_values_sum / num_files

            st.success(f"✓ Averaged {num_files} spectra successfully!")

            # Plot with Plotly
            fig = go.Figure()

            # Add individual spectra (faded)
            for i, spec in enumerate(spectra_data):
                fig.add_trace(
                    go.Scatter(
                        x=spec["x"],
                        y=spec["y"],
                        mode="lines",
                        name=f"Spectrum {i + 1}",
                        line=dict(color="lightgray", width=1),
                        opacity=0.5,
                    )
                )

            # Add averaged spectrum (prominent)
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=averaged_spectrum,
                    mode="lines",
                    name=f"Average (n={num_files})",
                    line=dict(color="red", width=3),
                )
            )

            fig.update_layout(
                title="OPUS Spectra Averaging",
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="Intensity",
                hovermode="x unified",
                height=500,
                showlegend=True,
            )

            fig.update_xaxes(
                showline=True,
                linewidth=2,
                linecolor="white",
                showgrid=True,
                mirror=True,
            )
            fig.update_yaxes(
                showline=True,
                linewidth=2,
                linecolor="white",
                showgrid=True,
                mirror=True,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Download as CSV
            spectrum = np.column_stack((x_values, averaged_spectrum))
            output = io.StringIO()
            np.savetxt(output, spectrum, delimiter=",", fmt="%s")
            csv_string = output.getvalue()

            st.download_button(
                label=f"Download Averaged Spectrum as CSV",
                data=csv_string,
                file_name=f"{first_file_name}_averaged.csv",
                mime="text/csv",
                key=f"download_averaged",
            )

            # Also offer individual spectrum download option
            with st.expander("Download Options"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**CSV Format:**")
                    st.code("wavenumber,intensity", language=None)
                with col2:
                    st.markdown("**Files Averaged:**")
                    st.write([s["name"] for s in spectra_data])

        else:
            st.write("No valid OPUS files uploaded.")


def averagespectrapage():
    st.title(":rainbow[Averaging Utility]")
    st.subheader(
        "Pick a series of equal OPUS Spectra to load them then click the -Submit- Button"
    )

    st.markdown(
        f"<h3 style = 'text-align: center; color: yellow;'> Choose a series of n equal spectra to sum and average them </h3>",
        unsafe_allow_html=True,
    )
    # get_elettra_status()

    averaging()
