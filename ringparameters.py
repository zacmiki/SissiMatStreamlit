# Functions to get the current values from the webpage
import requests
from bs4 import BeautifulSoup

def get_current_value():
    url = "https://www.elettra.eu/lightsources/elettra/status.html"
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    beam_table = soup.find('table', class_="data BeamInfo")
    current_row = beam_table.find('tr', id='beaminfo_current_row')
    current_value = current_row.find('td').text.strip()
    return current_value

def get_energy_value():
    url = "https://www.elettra.eu/lightsources/elettra/status.html"
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    beam_table = soup.find('table', class_="data BeamInfo")
    energy_row = beam_table.find('tr', id='beaminfo_energy_row')
    energy_value = energy_row.find('td').text.strip()
    return energy_value

def get_machine_status():
    url = "https://www.elettra.eu/lightsources/elettra/status.html"
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    beam_table = soup.find('table', class_="data Info")
    status_row = beam_table.find('tr', id='info_machine_status_row')
    status_value = status_row.find('td').text.strip()
    return status_value

def get_downtime_cause():
    url = "https://www.elettra.eu/lightsources/elettra/status.html"
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    beam_table = soup.find('table', class_="data Info")
    downtime_row = beam_table.find('tr', id='info_downtime_cause_row')
    downtime_value = downtime_row.find('td').text.strip()
    return downtime_value