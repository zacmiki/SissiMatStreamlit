import streamlit as st
import numpy as np

# Conversion functions
def wn_to_thz(wn):
    """Convert wavenumber (cm^-1) to THz"""
    return wn * 0.0299792458

def thz_to_wn(thz):
    """Convert THz to wavenumber (cm^-1)"""
    return thz / 0.0299792458

def wn_to_wavelength_micron(wn):
    """Convert wavenumber (cm^-1) to wavelength (micron)"""
    if wn == 0:
        return float('inf')
    return 10000 / wn

def wavelength_micron_to_wn(wavelength):
    """Convert wavelength (micron) to wavenumber (cm^-1)"""
    if wavelength == 0:
        return float('inf')
    return 10000 / wavelength

def wn_to_energy(wn):
    """Convert wavenumber (cm^-1) to energy (meV)"""
    return wn * 0.12398

def energy_to_wn(energy):
    """Convert energy (meV) to wavenumber (cm^-1)"""
    return energy / 0.12398

def wn_to_temperature(wn):
    """Convert wavenumber (cm^-1) to temperature (K)"""
    return wn * 1.4387

def temperature_to_wn(temp):
    """Convert temperature (K) to wavenumber (cm^-1)"""
    return temp / 1.4387

def wn_to_period_ps(wn):
    """Convert wavenumber (cm^-1) to period (ps)"""
    thz = wn_to_thz(wn)
    if thz == 0:
        return float('inf')
    return 1 / thz

def period_ps_to_wn(period):
    """Convert period (ps) to wavenumber (cm^-1)"""
    if period == 0:
        return float('inf')
    thz = 1 / period
    return thz_to_wn(thz)

def wn_to_wavelength_nm(wn):
    """Convert wavenumber (cm^-1) to wavelength (nm)"""
    if wn == 0:
        return float('inf')
    return 10000000 / wn

def wavelength_nm_to_wn(wavelength):
    """Convert wavelength (nm) to wavenumber (cm^-1)"""
    if wavelength == 0:
        return float('inf')
    return 10000000 / wavelength

def wn_to_period_fs(wn):
    """Convert wavenumber (cm^-1) to period (fs)"""
    thz = wn_to_thz(wn)
    if thz == 0:
        return float('inf')
    # 1 THz = 1000 fs period
    return 1000 / thz

def period_fs_to_wn(period_fs):
    """Convert period (fs) to wavenumber (cm^-1)"""
    if period_fs == 0:
        return float('inf')
    # Convert fs to THz (1000 fs = 1 ps = 1/THz)
    thz = 1000 / period_fs
    return thz_to_wn(thz)

def fs_to_ps(fs):
    """Convert femtoseconds to picoseconds"""
    return fs / 1000

def wn_to_magnetic_field(wn):
    """Convert wavenumber (cm^-1) to magnetic field (Tesla)"""
    # Using relationship between wavenumber and Tesla
    # This is approximate and depends on g-factor
    return wn * 0.0214  # Approximate conversion factor

def magnetic_field_to_wn(field):
    """Convert magnetic field (Tesla) to wavenumber (cm^-1)"""
    return field / 0.0214

def format_output(value):
    """Format output for display"""
    if value == float('inf') or value == float('-inf'):
        return "Infinity"
    elif abs(value) < 0.0001 and value != 0:
        return f"{value:.6e}"
    else:
        return f"{value:.4f}"

def page1():
    # Page layout
    st.title("🌈️ Infrared Spectroscopy Utils")
    st.subheader("Converters for IR Spectroscopy")
    st.divider()
    
    # Create a dictionary to store the state
    if 'last_changed' not in st.session_state:
        st.session_state['last_changed'] = None
    
    # Initialize session state values if not present
    for key in ['wavenumber', 'thz', 'wavelength_micron', 'temperature', 
                'period_ps', 'wavelength_nm', 'energy', 'period_fs', 'magnetic_field']:
        if key not in st.session_state:
            st.session_state[key] = 0.0
    
    # Function to update all values based on wavenumber
    def update_all_from_wn(wn):
        try:
            wn = float(wn)
            st.session_state['wavenumber'] = wn
            st.session_state['thz'] = wn_to_thz(wn)
            st.session_state['wavelength_micron'] = wn_to_wavelength_micron(wn)
            st.session_state['temperature'] = wn_to_temperature(wn)
            st.session_state['period_ps'] = wn_to_period_ps(wn)
            st.session_state['wavelength_nm'] = wn_to_wavelength_nm(wn)
            st.session_state['energy'] = wn_to_energy(wn)
            st.session_state['period_fs'] = wn_to_period_fs(wn)
            st.session_state['magnetic_field'] = wn_to_magnetic_field(wn)
        except (ValueError, TypeError):
            pass
    
    # Callback functions for each input field
    def on_wn_change():
        st.session_state['last_changed'] = 'wavenumber'
        try:
            wn = float(st.session_state['wavenumber_input'])
            st.session_state['wavenumber'] = wn
            update_all_from_wn(wn)
        except (ValueError, TypeError):
            pass
    
    def on_thz_change():
        st.session_state['last_changed'] = 'thz'
        try:
            thz = float(st.session_state['thz_input'])
            st.session_state['thz'] = thz
            wn = thz_to_wn(thz)
            update_all_from_wn(wn)
        except (ValueError, TypeError):
            pass
    
    def on_wavelength_micron_change():
        st.session_state['last_changed'] = 'wavelength_micron'
        try:
            wavelength = st.session_state['wavelength_micron_input']
            if wavelength == "Infinity":
                wavelength = float('inf')
            else:
                wavelength = float(wavelength)
            st.session_state['wavelength_micron'] = wavelength
            wn = wavelength_micron_to_wn(wavelength)
            update_all_from_wn(wn)
        except (ValueError, TypeError):
            pass
    
    def on_temperature_change():
        st.session_state['last_changed'] = 'temperature'
        try:
            temp = float(st.session_state['temperature_input'])
            st.session_state['temperature'] = temp
            wn = temperature_to_wn(temp)
            update_all_from_wn(wn)
        except (ValueError, TypeError):
            pass
    
    def on_period_ps_change():
        st.session_state['last_changed'] = 'period_ps'
        try:
            period = st.session_state['period_ps_input']
            if period == "Infinity":
                period = float('inf')
            else:
                period = float(period)
            st.session_state['period_ps'] = period
            wn = period_ps_to_wn(period)
            update_all_from_wn(wn)
        except (ValueError, TypeError):
            pass
    
    def on_wavelength_nm_change():
        st.session_state['last_changed'] = 'wavelength_nm'
        try:
            wavelength = st.session_state['wavelength_nm_input']
            if wavelength == "Infinity":
                wavelength = float('inf')
            else:
                wavelength = float(wavelength)
            st.session_state['wavelength_nm'] = wavelength
            wn = wavelength_nm_to_wn(wavelength)
            update_all_from_wn(wn)
        except (ValueError, TypeError):
            pass
    
    def on_energy_change():
        st.session_state['last_changed'] = 'energy'
        try:
            energy = float(st.session_state['energy_input'])
            st.session_state['energy'] = energy
            wn = energy_to_wn(energy)
            update_all_from_wn(wn)
        except (ValueError, TypeError):
            pass
    
    def on_period_fs_change():
        st.session_state['last_changed'] = 'period_fs'
        try:
            period = st.session_state['period_fs_input']
            if period == "Infinity":
                period = float('inf')
            else:
                period = float(period)
            st.session_state['period_fs'] = period
            wn = period_fs_to_wn(period)
            update_all_from_wn(wn)
        except (ValueError, TypeError):
            pass
    
    def on_magnetic_field_change():
        st.session_state['last_changed'] = 'magnetic_field'
        try:
            field = float(st.session_state['magnetic_field_input'])
            st.session_state['magnetic_field'] = field
            wn = magnetic_field_to_wn(field)
            update_all_from_wn(wn)
        except (ValueError, TypeError):
            pass
    
    # If this is the first run, update everything based on wavenumber
    if st.session_state['last_changed'] is None:
        update_all_from_wn(0)
    
    # Layout with three columns as in the screenshot
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"#### Wavenumber [cm<sup>-1</sup>]:", unsafe_allow_html=True)
        st.text_input("", value=format_output(st.session_state['wavenumber']), 
                     key="wavenumber_input", on_change=on_wn_change, label_visibility="collapsed")
        
        st.markdown("#### Temperature [K]:")
        st.text_input("", value=format_output(st.session_state['temperature']), 
                     key="temperature_input", on_change=on_temperature_change, label_visibility="collapsed")
        
        st.markdown("#### Photon Energy [meV]:")
        st.text_input("", value=format_output(st.session_state['energy']), 
                     key="energy_input", on_change=on_energy_change, label_visibility="collapsed")
    
    with col2:
        st.markdown("#### Frequency [THz]:")
        st.text_input("", value=format_output(st.session_state['thz']), 
                     key="thz_input", on_change=on_thz_change, label_visibility="collapsed")
        
        st.markdown("#### Period [ps]:")
        st.text_input("", value=format_output(st.session_state['period_ps']), 
                     key="period_ps_input", on_change=on_period_ps_change, label_visibility="collapsed")
        
        st.markdown("#### Period [fs]:")
        st.text_input("", value=format_output(st.session_state['period_fs']), 
                     key="period_fs_input", on_change=on_period_fs_change, label_visibility="collapsed")
    
    with col3:
        st.markdown("#### Wavelength [µm]:")
        st.text_input("", value=format_output(st.session_state['wavelength_micron']), 
                     key="wavelength_micron_input", on_change=on_wavelength_micron_change, label_visibility="collapsed")
        
        st.markdown("#### Wavelength [nm]:")
        st.text_input("", value=format_output(st.session_state['wavelength_nm']), 
                     key="wavelength_nm_input", on_change=on_wavelength_nm_change, label_visibility="collapsed")
        
        st.markdown("#### Magnetic Field [T]:")
        st.text_input("", value=format_output(st.session_state['magnetic_field']), 
                     key="magnetic_field_input", on_change=on_magnetic_field_change, label_visibility="collapsed")
    
    st.divider()