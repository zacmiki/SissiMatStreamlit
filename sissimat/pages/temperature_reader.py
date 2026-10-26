"""Live Lake Shore 331 temperature reader page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import serial
import streamlit as st

from sissimat.instruments.lakeshore331 import (
    LakeShore331,
    TemperatureReading,
    append_temperature_snapshot,
    available_serial_ports,
)


STATE_PREFIX = "temperature_reader_"
CONTROLLER_KEY = f"{STATE_PREFIX}controller"
RUNNING_KEY = f"{STATE_PREFIX}running"
IDENTITY_KEY = f"{STATE_PREFIX}identity"
PORT_KEY = f"{STATE_PREFIX}port"
HISTORY_KEY = f"{STATE_PREFIX}history"
LAST_READING_KEY = f"{STATE_PREFIX}last_reading"
STORED_KEY = f"{STATE_PREFIX}stored"


def _initialise_state() -> None:
    defaults = {
        RUNNING_KEY: False,
        IDENTITY_KEY: "",
        PORT_KEY: "",
        HISTORY_KEY: [],
        LAST_READING_KEY: None,
        STORED_KEY: [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def close_temperature_reader() -> None:
    """Release the serial port when acquisition stops or the page is left."""

    controller = st.session_state.get(CONTROLLER_KEY)
    if controller is not None:
        try:
            controller.close()
        except (OSError, serial.SerialException):
            pass
    st.session_state.pop(CONTROLLER_KEY, None)
    st.session_state[RUNNING_KEY] = False


def _append_history(reading: TemperatureReading) -> None:
    history = st.session_state[HISTORY_KEY]
    history.append(
        {
            "timestamp": reading.timestamp,
            "A": reading.channel_a_kelvin,
            "B": reading.channel_b_kelvin,
        }
    )
    del history[:-3600]
    st.session_state[LAST_READING_KEY] = reading


def _temperature_figure(
    history: list[dict], display: str, ignore_zero: bool
) -> go.Figure:
    figure = go.Figure()
    channels = {"A + B": ["A", "B"], "A only": ["A"], "B only": ["B"]}[display]
    frame = pd.DataFrame(history)
    colours = {"A": "#00A6D6", "B": "#FF7F0E"}
    for channel in channels:
        values = frame[channel].mask(frame[channel] <= 0) if ignore_zero else frame[channel]
        figure.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=values,
                mode="lines+markers",
                name=f"Channel {channel}",
                line={"color": colours[channel], "width": 2},
                marker={"size": 5},
                connectgaps=False,
            )
        )
    figure.update_layout(
        xaxis_title="Time",
        yaxis_title="Temperature (K)",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        uirevision="temperature-reader",
    )
    return figure


@st.fragment(run_every=1.0)
def _live_acquisition(display: str, ignore_zero: bool) -> None:
    controller = st.session_state.get(CONTROLLER_KEY)
    if not st.session_state.get(RUNNING_KEY) or controller is None:
        return

    try:
        reading = controller.read_temperatures()
    except (OSError, TimeoutError, RuntimeError, serial.SerialException) as error:
        close_temperature_reader()
        st.error(f"Acquisition stopped: {error}")
        return

    _append_history(reading)
    metric_a, metric_b, samples = st.columns(3)
    metric_a.metric("Channel A", f"{reading.channel_a_kelvin:.3f} K")
    metric_b.metric("Channel B", f"{reading.channel_b_kelvin:.3f} K")
    samples.metric("Live samples", f"{len(st.session_state[HISTORY_KEY]):,}")

    history = st.session_state[HISTORY_KEY]
    if history:
        st.plotly_chart(
            _temperature_figure(history, display, ignore_zero),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
            key="temperature_live_plot",
        )


def _store_current_value(output_path: str, note: str) -> None:
    controller = st.session_state.get(CONTROLLER_KEY)
    if not st.session_state.get(RUNNING_KEY) or controller is None:
        st.error("Start an acquisition before storing a value.")
        return
    try:
        reading = controller.read_temperatures()
        _append_history(reading)
        saved_path = append_temperature_snapshot(
            output_path,
            reading,
            controller=st.session_state[IDENTITY_KEY],
            serial_port=st.session_state[PORT_KEY],
            note=note,
        )
    except (OSError, TimeoutError, RuntimeError, serial.SerialException, ValueError) as error:
        st.error(f"Could not store the value: {error}")
        return

    st.session_state[STORED_KEY].append(
        {
            "Time (UTC)": reading.timestamp.isoformat(timespec="seconds"),
            "A (K)": reading.channel_a_kelvin,
            "B (K)": reading.channel_b_kelvin,
            "Note": note.strip(),
        }
    )
    st.success(f"Stored A and B in {saved_path}")


def render_temperature_reader() -> None:
    """Render the T Reading page."""

    _initialise_state()
    st.title("🌡️ T Reading")
    st.caption(
        "Live, read-only Lake Shore 331 acquisition. The controller setpoint, "
        "heater, PID, and other settings are never changed."
    )

    ports = available_serial_ports()
    detected = {port.device: port.label for port in ports}
    choices = list(detected)
    if choices:
        selected_port = st.selectbox(
            "Serial port",
            choices,
            format_func=lambda device: detected[device],
            disabled=st.session_state[RUNNING_KEY],
            help="On Windows this is normally COM3, COM4, or similar.",
        )
    else:
        st.warning("No serial ports were detected. Connect the USB adapter and refresh.")
        selected_port = st.text_input(
            "Serial port",
            value="COM3",
            disabled=st.session_state[RUNNING_KEY],
        ).strip()

    controls = st.columns([1, 1, 3])
    if controls[0].button(
        "Start acquisition",
        type="primary",
        disabled=st.session_state[RUNNING_KEY] or not selected_port,
        use_container_width=True,
    ):
        close_temperature_reader()
        controller = None
        try:
            controller = LakeShore331(selected_port, baudrate=9600)
            identity = controller.identify()
        except (OSError, TimeoutError, serial.SerialException) as error:
            if controller is not None:
                controller.close()
            st.error(f"Could not connect to {selected_port}: {error}")
        else:
            st.session_state[CONTROLLER_KEY] = controller
            st.session_state[RUNNING_KEY] = True
            st.session_state[IDENTITY_KEY] = identity
            st.session_state[PORT_KEY] = selected_port
            st.session_state[HISTORY_KEY] = []
            st.session_state[LAST_READING_KEY] = None
            st.rerun()

    if controls[1].button(
        "Stop",
        disabled=not st.session_state[RUNNING_KEY],
        use_container_width=True,
    ):
        close_temperature_reader()
        st.rerun()

    if st.session_state[RUNNING_KEY]:
        st.success(
            f"Connected to {st.session_state[IDENTITY_KEY]} on "
            f"{st.session_state[PORT_KEY]} at 9600 baud."
        )
    else:
        st.info("Select the COM port and start acquisition.")

    display = st.radio(
        "Channels displayed",
        ["A + B", "A only", "B only"],
        horizontal=True,
        key=f"{STATE_PREFIX}display",
    )
    ignore_zero = st.checkbox(
        "Hide 0 K readings from the graph",
        value=True,
        help="A 0 K response usually indicates an unconfigured or invalid input.",
        key=f"{STATE_PREFIX}ignore_zero",
    )

    if st.session_state[RUNNING_KEY]:
        _live_acquisition(display, ignore_zero)

    st.subheader("Store a temperature snapshot")
    output_path = st.text_input(
        "CSV file",
        value="temperature_snapshots.csv",
        help="A relative path is saved from the directory where Streamlit was started.",
    ).strip()
    note = st.text_input(
        "Experiment note (optional)",
        placeholder="Sample, PID/heater condition, or FTIR acquisition name",
    )
    if st.button(
        "Store Value",
        type="primary",
        disabled=not st.session_state[RUNNING_KEY] or not output_path,
    ):
        _store_current_value(output_path, note)

    stored = st.session_state[STORED_KEY]
    if stored:
        st.markdown("#### Values stored during this session")
        st.dataframe(pd.DataFrame(stored), hide_index=True, width="stretch")
        path = Path(output_path).expanduser()
        if path.is_file():
            st.download_button(
                "Download snapshot CSV",
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/csv",
            )
