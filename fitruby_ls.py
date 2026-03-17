import streamlit as st
import plotly.graph_objects as go
import numpy as np
from lmfit.models import ConstantModel, LorentzianModel
import matplotlib.pyplot as plt
from fitruby import load_rubyfile


# -------------------- Lorentzian parameters definition
def try_lorentzian(
    data, l1_center, l1_amplitude, l1_sigma, l2_center, l2_amplitude, l2_sigma
):
    background = ConstantModel(prefix="bkg_")  # preparing the background parameter
    pars = background.guess(
        data[:, 1], x=data[:, 0]
    )  # guessing the background for my data

    l1 = LorentzianModel(prefix="l1_")
    pars.update(l1.make_params())

    pars["l1_center"].set(value=l1_center, min=688, max=699)
    pars["l1_sigma"].set(value=l1_sigma, min=0.01, max=2)
    pars["l1_amplitude"].set(value=l1_amplitude, min=200, max=60000)

    l2 = LorentzianModel(prefix="l2_")
    pars.update(l2.make_params())

    pars["l2_center"].set(value=l2_center, min=688, max=699)
    pars["l2_sigma"].set(value=l2_sigma, min=0.02, max=2)
    pars["l2_amplitude"].set(value=l2_amplitude, min=200, max=50000)

    model = background + l1 + l2
    init = model.eval(pars, x=data[:, 0])
    return model, init, pars


def rubyfitls():
    st.title(":rainbow[Load and Fit the Ruby File]")

    # Initialize the file_loaded state if it doesn't exist
    if "file_loaded" not in st.session_state:
        st.session_state.file_loaded = False

    # File loading section
    st.markdown("### 📂 Load Ruby File")

    # Check if file already loaded from DAC Utilities
    if st.session_state.get("file_loaded", False) and "data" in st.session_state:
        st.success(
            f"✓ File already loaded: {st.session_state.get('ruby_filename', 'Ruby data')}"
        )
        data = st.session_state.data
        if st.button("Load a different file"):
            st.session_state.file_loaded = False
            if "data" in st.session_state:
                del st.session_state["data"]
            st.rerun()
    else:
        # No file loaded yet - show uploader
        data = load_rubyfile()

    # Fitting section
    if st.session_state.file_loaded and data is not None:
        st.markdown("---")
        st.markdown("### 🔧 Fit Parameters")

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            l1_center = st.slider(
                "Lor1 Center",
                688.0,
                710.0,
                float(st.session_state.get("l1_center", 692.5)),
                step=0.05,
            )

        with col2:
            l1_amplitude = st.slider(
                "Lor1 Amplitude",
                200,
                50000,
                int(st.session_state.get("l1_amplitude", 4400)),
                step=50,
            )

        with col3:
            l1_sigma = st.slider(
                "Lor1 Sigma",
                0.01,
                2.0,
                float(st.session_state.get("l1_sigma", 0.33)),
                step=0.01,
            )

        with col4:
            l2_center = st.slider(
                "Lor2 Center",
                688.0,
                710.0,
                float(st.session_state.get("l2_center", 694.5)),
                step=0.05,
            )

        with col5:
            l2_amplitude = st.slider(
                "Lor2 Amplitude",
                200,
                50000,
                int(st.session_state.get("l2_amplitude", 7900)),
                step=50,
            )

        with col6:
            l2_sigma = st.slider(
                "Lor2 Sigma",
                0.02,
                2.0,
                float(st.session_state.get("l2_sigma", 0.38)),
                step=0.01,
            )

        model, init, pars = try_lorentzian(
            data, l1_center, l1_amplitude, l1_sigma, l2_center, l2_amplitude, l2_sigma
        )

        # Preview plot with Plotly
        fig = go.Figure()

        fig.update_layout(height=600)

        # Scatter plot of the data
        fig.add_trace(
            go.Scatter(
                x=data[:, 0],
                y=data[:, 1],
                mode="markers",
                name="Data",
                showlegend=False,
            )
        )

        # Adding the line graph of initial fit
        fig.add_trace(
            go.Scatter(
                x=data[:, 0],
                y=init,
                mode="lines",
                line=dict(color="red"),
                name="Lorentzian Fit",
                showlegend=False,
            )
        )

        # Set x-axis range manually
        x_range = st.slider(
            "X-axis range",
            float(data[:, 0].min()),
            float(data[:, 0].max()),
            (float(data[:, 0].min()), float(data[:, 0].max())),
        )

        fig.update_layout(
            title="Ruby Fluorescence - Lorentzian Fit Preview",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Intensity (counts)",
            xaxis=dict(range=x_range),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Fit button
        if st.button("Fit with current parameters", type="primary"):
            result = model.fit(data[:, 1], pars, x=data[:, 0])
            comps = result.eval_components(x=data[:, 0])

            # Update session state with fitted values
            st.session_state.l1_center = float(result.params["l1_center"].value)
            st.session_state.l1_amplitude = float(result.params["l1_amplitude"].value)
            st.session_state.l1_sigma = float(result.params["l1_sigma"].value)
            st.session_state.l2_center = float(result.params["l2_center"].value)
            st.session_state.l2_amplitude = float(result.params["l2_amplitude"].value)
            st.session_state.l2_sigma = float(result.params["l2_sigma"].value)

            rsquared = round(1 - (result.residual.var() / np.var(data[:, 1])), 4)

            # Display results
            st.success("Fit completed!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lorentzian 1 Center", f"{st.session_state.l1_center:.4f} nm")
            with col2:
                st.metric("Lorentzian 2 Center", f"{st.session_state.l2_center:.4f} nm")
            with col3:
                st.metric("R²", f"{rsquared}")

            x = data[:, 0]
            y = data[:, 1]

            # Matplotlib plot
            fig, axes = plt.subplots(1, 2, figsize=(18, 9))
            axes[0].plot(x, y, ".", label="Data")
            axes[0].plot(x, result.best_fit, "-", label="best fit")
            axes[0].plot(x, y - result.best_fit, "--", label="Residuals")

            axes[0].set_xlim(690, 700)
            axes[0].set_xlabel("Wavelength (nm)")
            axes[0].set_ylabel("Intensity")
            axes[0].grid(which="major", axis="y", linewidth=0.2)
            axes[0].grid(which="both", axis="x", lw=0.2)
            axes[0].legend()

            axes[1].plot(x, y - comps["bkg_"], ".")
            axes[1].plot(x, comps["l1_"], "--", label="Lorentzian 1")
            axes[1].plot(x, comps["l2_"], "--", label="Lorentzian 2")

            axes[1].set_xlim(690, 700)
            axes[1].set_xlabel("Wavelength (nm)")
            axes[1].grid(which="major", axis="y", linewidth=0.2)
            axes[1].grid(which="both", axis="x", lw=0.2)
            axes[1].legend()
            st.pyplot(fig)
