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
    return 1000 / thz

def period_fs_to_wn(period_fs):
    """Convert period (fs) to wavenumber (cm^-1)"""
    if period_fs == 0:
        return float('inf')
    thz = 1000 / period_fs
    return thz_to_wn(thz)

def fs_to_ps(fs):
    """Convert femtoseconds to picoseconds"""
    return fs / 1000

def wn_to_magnetic_field(wn):
    """Convert wavenumber (cm^-1) to magnetic field (Tesla)"""
    return wn * 0.0214  # Approximate conversion factor

def magnetic_field_to_wn(field):
    """Convert magnetic field (Tesla) to wavenumber (cm^-1)"""
    return field / 0.0214

def format_output(value, precision=4):
    """Format output for display with adjustable precision"""
    if value == float('inf') or value == float('-inf'):
        return "Infinity"
    elif abs(value) < 0.0001 and value != 0:
        return f"{value:.6e}"
    else:
        return f"{value:.{precision}f}"

def page1():
    # Page layout
    st.title("🌈️ Infrared Spectroscopy Utils")
    st.subheader("Converters for IR Spectroscopy")
    st.divider()
    
    # Initialize session state
    if 'last_changed' not in st.session_state:
        st.session_state['last_changed'] = None
    
    for key in ['wavenumber', 'thz', 'wavelength_micron', 'temperature', 
                'period_ps', 'wavelength_nm', 'energy', 'period_fs', 'magnetic_field']:
        if key not in st.session_state:
            st.session_state[key] = 0.0
    
    def update_all_from_wn(wn):
        try:
            wn = float(wn)
            # Update logic values
            st.session_state['wavenumber'] = wn
            st.session_state['thz'] = wn_to_thz(wn)
            st.session_state['wavelength_micron'] = wn_to_wavelength_micron(wn)
            st.session_state['temperature'] = wn_to_temperature(wn)
            st.session_state['period_ps'] = wn_to_period_ps(wn)
            st.session_state['wavelength_nm'] = wn_to_wavelength_nm(wn)
            st.session_state['energy'] = wn_to_energy(wn)
            st.session_state['period_fs'] = wn_to_period_fs(wn)
            st.session_state['magnetic_field'] = wn_to_magnetic_field(wn)

            # Update UI widget keys with specific formatting
            st.session_state['wavenumber_input'] = format_output(st.session_state['wavenumber'], precision=0)
            st.session_state['energy_input'] = format_output(st.session_state['energy'], precision=1)
            st.session_state['magnetic_field_input'] = f"{format_output(st.session_state['magnetic_field'], precision=2)} T"
            st.session_state['thz_input'] = format_output(st.session_state['thz'], precision = 2)
            st.session_state['wavelength_micron_input'] = format_output(st.session_state['wavelength_micron'], precision = 3)
            st.session_state['temperature_input'] = format_output(st.session_state['temperature'], precision = 2)
            st.session_state['period_ps_input'] = format_output(st.session_state['period_ps'], precision = 3)
            st.session_state['wavelength_nm_input'] = format_output(st.session_state['wavelength_nm'], precision = 0)
            st.session_state['period_fs_input'] = format_output(st.session_state['period_fs'], precision = 0)
            
        except (ValueError, TypeError):
            pass
    
    # Callbacks
    def on_wn_change():
        st.session_state['last_changed'] = 'wavenumber'
        try:
            val = st.session_state['wavenumber_input']
            wn = float(val) if val.upper() != "INFINITY" else float('inf')
            update_all_from_wn(wn)
        except: pass
    
    def on_thz_change():
        st.session_state['last_changed'] = 'thz'
        try:
            val = st.session_state['thz_input']
            thz = float(val) if val.upper() != "INFINITY" else float('inf')
            update_all_from_wn(thz_to_wn(thz))
        except: pass
    
    def on_wavelength_micron_change():
        st.session_state['last_changed'] = 'wavelength_micron'
        try:
            val = st.session_state['wavelength_micron_input']
            wl = float(val) if val.upper() != "INFINITY" else float('inf')
            update_all_from_wn(wavelength_micron_to_wn(wl))
        except: pass
    
    def on_temperature_change():
        st.session_state['last_changed'] = 'temperature'
        try:
            val = st.session_state['temperature_input']
            temp = float(val) if val.upper() != "INFINITY" else float('inf')
            update_all_from_wn(temperature_to_wn(temp))
        except: pass
    
    def on_period_ps_change():
        st.session_state['last_changed'] = 'period_ps'
        try:
            val = st.session_state['period_ps_input']
            p = float(val) if val.upper() != "INFINITY" else float('inf')
            update_all_from_wn(period_ps_to_wn(p))
        except: pass
    
    def on_wavelength_nm_change():
        st.session_state['last_changed'] = 'wavelength_nm'
        try:
            val = st.session_state['wavelength_nm_input']
            wl_nm = float(val) if val.upper() != "INFINITY" else float('inf')
            update_all_from_wn(wavelength_nm_to_wn(wl_nm))
        except: pass
    
    def on_energy_change():
        st.session_state['last_changed'] = 'energy'
        try:
            val = st.session_state['energy_input']
            e = float(val) if val.upper() != "INFINITY" else float('inf')
            update_all_from_wn(energy_to_wn(e))
        except: pass
    
    def on_period_fs_change():
        st.session_state['last_changed'] = 'period_fs'
        try:
            val = st.session_state['period_fs_input']
            p_fs = float(val) if val.upper() != "INFINITY" else float('inf')
            update_all_from_wn(period_fs_to_wn(p_fs))
        except: pass
    
    def on_magnetic_field_change():
        st.session_state['last_changed'] = 'magnetic_field'
        try:
            val = st.session_state['magnetic_field_input']
            # Remove " T" if present before converting to float
            field_str = val.replace(" T", "").strip()
            field = float(field_str) if field_str.upper() != "INFINITY" else float('inf')
            update_all_from_wn(magnetic_field_to_wn(field))
        except: pass
    
    if st.session_state['last_changed'] is None:
        update_all_from_wn(0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"#### Wavenumber [cm<sup>-1</sup>]:", unsafe_allow_html=True)
        st.text_input("Wavenumber", key="wavenumber_input", on_change=on_wn_change, label_visibility="collapsed")
        
        st.markdown("#### Temperature [K]:")
        st.text_input("Temperature", key="temperature_input", on_change=on_temperature_change, label_visibility="collapsed")
        
        st.markdown("#### Photon Energy [meV]:")
        st.text_input("Photon Energy", key="energy_input", on_change=on_energy_change, label_visibility="collapsed")
    
    with col2:
        st.markdown("#### Frequency [THz]:")
        st.text_input("Frequency", key="thz_input", on_change=on_thz_change, label_visibility="collapsed")
        
        st.markdown("#### Period [ps]:")
        st.text_input("Period (ps)", key="period_ps_input", on_change=on_period_ps_change, label_visibility="collapsed")
        
        st.markdown("#### Period [fs]:")
        st.text_input("Period (fs)", key="period_fs_input", on_change=on_period_fs_change, label_visibility="collapsed")
    
    with col3:
        st.markdown("#### Wavelength [µm]:")
        st.text_input("Wavelength (µm)", key="wavelength_micron_input", on_change=on_wavelength_micron_change, label_visibility="collapsed")
        
        st.markdown("#### Wavelength [nm]:")
        st.text_input("Wavelength (nm)", key="wavelength_nm_input", on_change=on_wavelength_nm_change, label_visibility="collapsed")
        
        st.markdown("#### Magnetic Field [T]:")
        st.text_input("Magnetic Field", key="magnetic_field_input", on_change=on_magnetic_field_change, label_visibility="collapsed")
    
    st.divider()
