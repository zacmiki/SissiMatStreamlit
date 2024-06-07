import streamlit as st
"""
Created on Mon Dec 19 10:04:36 2022

SISSI-MAT useful functions

getListOfFiles(dirName) --- Returns a list of all files in the given directory "dirName"
allOpusFiles(dirName) --- returns a list with all the OPUS files in that dir and all subdirs
savitzky_golay(y, window_size, order, deriv=0, rate=1) --- returns the sav-gol smoothed array

DACPress(wl) - Returns the pressure in GPa - given the ruby line wavelength in nm
DACTemp(wl) -- Returns the temperature in K given the position of ruby line wl in nm
DACLinePos(T) - You give the temperature that you want to reach and it tells you the line position (in nm)

wn2En(waveNumber) -- Returns the Energy in meV from the value in cm-1
tHz2wn(terahertz) - terahertz to wavenumber converter
en2wn(waveNumber) -- Returns the WveNumber from the energy in meV
wn2THz(wavenumber) - wavenumber to terahertz converter


parValues(fileName) -- returns the meaningful Parameters of the OPUS file passed to the procedure
loadandgraph(fileName) -- non bello -- graphs the saved files
graphOpusFile(fileName) - graphs all the spectra of the File - returns the name of file and parameters
loadAB(fileName) - loads only the absorption spectrum , returns a list with the two arrayz X and Y
loadSSC(fileName) - loads only the SingleChannel - returns all the OPUSFC object

read_tab_delimited_file(file_path) - Reads an X,Y DAT tab separated into two numpy arrays
read

@author: miczac
"""

# loadandgraph(fileName, graph, params) -- load the filename and returns the data, graph and params are not zero they are displayed

#------------------------------------------

def getListOfFiles(dirName):
    import os
    
    # Create an empty list to store the names of the files
    fileList = []
    
    # Use the os.walk function to iterate over the files and directories in the specified directory
    for root, dirs, files in os.walk(dirName):
        # Add the names of the files in the current directory to the file list
        for file in files:
            fileList.append(os.path.join(root, file))
    
    return fileList

#------------------------------------------

def allOpusFiles(dirName):
    '''Returns a list with ALL!!!!!! the OPUS files in that dir and all subdirs'''
    import os
    import opusFC

    # Get a list of all files in the specified directory and its subdirectories
    all_files = getListOfFiles(dirName)

    # Use a list comprehension to create a list of the names of the non-empty Opus files
    opus_files = [file for file in all_files if os.path.getsize(file) > 0 and opusFC.isOpusFile(file)]

    return opus_files
#-----------------------------------------

def opusFiles(dirName):
    import opusFC, os 
    from sissi_util import allOpusFiles
    
    fullList = allOpusFiles(dirName)
    fileList = []
    
    for item in range(len(fullList)):
        dbs = opusFC.listContents(fullList[item])
        if len(dbs) != 0:
            fileList.append(fullList[item])
    return fileList
#------------------------------------------

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    '''returns the sav-gol smoothed array'''
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int32(window_size))
        order = np.abs(np.int32(order))
    except ValueError:
        print("window_size and order have to be of type int")

    except window_size % 2 != 1 or window_size < 1:
        print("window_size size must be a positive odd number")

    except window_size < order + 2:
        print("window_size is too small for the polynomials order")

    order_range = range(order+1)
    half_window = (window_size -1) // 2

    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    # pad the signal at the extremes with
    # values taken from the signal itself

    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

#------------------------------------------

def DACPress(wlength):
    '''Returns the pressure in GPa - given the ruby line wavelength in nm'''
    A = float(1904)
    B = float(7.715)
    R1 = float(694.19)

    A1 = float(1870)
    B1 = float(5.63)

    pressure1 = round(A/B * ((wlength/R1)**B -1),4)
    pressure2 = round(A1*((wlength-R1)/R1)*(1 + B1*((wlength-R1)/R1)),4)

    #print(f"\nFor a Ruby wavelength of {wlength} nm the pressure is {pressure1} GPa")
    #print(f"By Using the Ruby2020 formula the pressure is {pressure2} GPa\n")
    return pressure1

#------------------------------------------

def DACTemp(wlength):
    '''Returns the temperature in K given the position of ruby line wl in nm'''
    Tambient = float(298.1) #Our Ambient temperature while acquiring the ref ruby line
    R1 = float(694.19)      #Our Reference ruby Line position at Ambient Temperature

    # Tambient = float(input("\n \nEnter the value of the ambient temp in K: "))   ---- old
    #print(f"\nRef wavelength used: {R1} nm\nAmbient Temperature used: {Tambient} K")

    #T1 = float(273.15 + Tambient)
    T2 = (wlength - R1)/0.00726 + Tambient

    #print("For a wavelength of", wlength, "the temperature is %.3f" % T2, "K")
    return T2

#------------------------------------------
def DACLinePos(temp):
    Tambient = float(298.1) #Ref Ambient temperature while acquiring the ref ruby line
    R1 = float(694.19)      #Ref ruby Line position at Ambient Temperature
    #print("\nRef wavelength and temperature for our ruby are ", R1, "nm at %.3f" % Tambient, "K")

    R2 = R1 + (temp - Tambient) * 0.00726

    #print(f"At {temp}K the line will be at %.3f" % R2, "nm\n")
    return R2

#------------------------------------------

def wn2en(wavenumber):
    h = 4.135667516E-15
    c = 299792458
    wavenum = float(wavenumber)
    energy = h * c * wavenum * 100 * 1000
    #print(f'{wavenumber} cm-1 => = ', energy , 'meV')
    return energy

#------------------------------------------


def en2wn(energy):
    h = 4.135667516E-15
    c = 299792458
    energy = float(energy)
    wn = energy / (h * c * 100 * 1000)
    #print(f'{energy} meV => = ', wn , 'cm-1')
    return wn

#------------------------------------------



def thz2wn(freq2convert):
    '''terahertz to wavenumber converter'''
    convFactor = 33.356
    waveNumber = freq2convert * convFactor
    #print(freq2convert, 'THz ==>',  waveNumber, 'cm-1')
    return waveNumber

#------------------------------------------

def wn2thz(wavenumber):
    convFactor = 33.356
    freq = wavenumber / convFactor
    #print(wavenumber, 'cm-1 ==>',  freq, 'THz')
    return freq

#------------------------------------------

def wl2wn(wavelength):
    factor = 10000000 # ten millions
    wavenumber = factor/wavelength
    #print(wavelength, 'nm ==>',  wavenumber, 'cm-1')
    return wavenumber

#------------------------------------------

def parvalues(fileName):
    '''returns the meaningful Parameters of the OPUS file passed to the procedure'''
    import opusFC
    import matplotlib.pyplot as plt
    import os

    dbs = opusFC.listContents(fileName)

    for item in range(len(dbs)):
        if (dbs[item][0]) != 'SIFG':
            data = opusFC.getOpusData(fileName, dbs[item])
            
            st.markdown("#### Acquisition Parameters:")
            
            st.markdown(
            f"###### :red[Spec Type =] {data.parameters.get('PLF')}\n" +
            f"###### :red[Aperture =]  {data.parameters.get('APT')}  \n###### :red[BSplitter =] \t{data.parameters.get('BMS')}\n" +
            f"###### :red[Source =]  \t{data.parameters.get('SRC')}  \n###### :red[Detector =] \t{data.parameters.get('DTC')}\n" +
            f"###### :red[Frequency =]  \t{data.parameters.get('VEL')} kHz  \n###### :red[Channel =] \t{data.parameters.get('CHN')}\n" +
            f"###### :red[Resol =]  \t{data.parameters.get('RES')} cm-1\n"
            f"###### :red[Data From:] {data.parameters.get('LXV')} :red[to --->] {data.parameters.get('FXV')} $cm^{-1}$\n" + 
            f"###### :red[Pressure =]  \t{data.parameters.get('PRS')} hPa\n"
            f"###### :red[Acq Date =]  \t{data.parameters.get('DAT')}\n"
            f"###### :red[Acq Time =]  \t{data.parameters.get('TIM')}\n"
            )

    return

#------------------------------------------



def loadandgraph(fileName, graphornot = None, paramornot = None):
    '''Load the filename and returns the data, graph and params are not zero they are displayed'''
    import opusFC
    import matplotlib.pyplot as plt
    import os

    dbs = opusFC.listContents(fileName)
    print(f"\nFile {os.path.basename(fileName)} Loaded\n")

    for item in range(len(dbs)):
        if (dbs[item][0]) != 'SIFG':
            data = opusFC.getOpusData(fileName, dbs[item])
            labella = os.path.basename(fileName) + "_" + dbs[item][0]
            suffix = dbs[item][0]

            # If you want to print to graph of not the opus file
            if graphornot == None:
                fig, ftir1 = plt.subplots()  # Create a figure containing a single axis.
                ftir1.minorticks_on()
                ftir1.set(xlabel='Wavenumbers (cm-1)', ylabel='Intensity', title= labella)
                ftir1.plot(data.x, data.y, label = suffix, linewidth= 0.5)  # Plot IR Transformed spectrum.
                ftir1.legend()

                ftir1.grid(which = 'both', axis = 'x', lw = .2)
                ftir1.grid(which = 'major', axis = 'y', linewidth = .2)
                plt.show()

            # If you want to print the files parameters
            if paramornot == None:
                print(f"\n\033[1m{labella} Acquisition Parameters \n\033[0m")
                print(f"Spec Type =\t{data.parameters['PLF']}")
                print(f"Aperture =\t{data.parameters['APT']} \nBSplitter = \t{data.parameters['BMS']}")
                print(f"Source = \t{data.parameters['SRC']} \nDetector = \t{data.parameters['DTC']}")
                print(f"Frequency = \t{data.parameters['VEL']} kHz\nChannel = \t{data.parameters['CHN']}")
                print(f"Resol = \t{data.parameters['RES']} cm-1")
                print(f"Data from:\t{data.parameters['LXV']} to {data.parameters['FXV']} cm-1")
                print(f"Pressure =\t{data.parameters['PRS']} hPa")
    return data
    
#------------------------------------------

def graphOpusFile(fileName):
    '''Load the filename and returns the data, graph and params are not zero they are displayed'''
    import opusFC
    import matplotlib.pyplot as plt
    import os

    dbs = opusFC.listContents(fileName)
    print(f"\nFile {os.path.basename(fileName)} Loaded\n")
    
    if len(dbs) !=0:
        for item in range(len(dbs)):
            data = opusFC.getOpusData(fileName, dbs[item])
            labella = os.path.basename(fileName) + "_" + dbs[item][0]
            suffix = dbs[item][0]

            # If you want to print to graph of not the opus file
            fig, ftir1 = plt.subplots()  # Create a figure containing a single axis.
            ftir1.minorticks_on()
            ftir1.set(xlabel='Wavenumbers (cm-1)', ylabel='Intensity', title= labella)
            ftir1.plot(data.x, data.y, label = suffix, linewidth= 0.5)  # Plot IR Transformed spectrum.
            ftir1.legend()

            ftir1.grid(which = 'both', axis = 'x', lw = .2)
            ftir1.grid(which = 'major', axis = 'y', linewidth = .2)
            plt.show()
        return data
    print("No valid data found inside the File")
    return None


# -------------------------------------------

def loadAB(fileName):
    import opusFC
    import os

    dbs = opusFC.listContents(fileName)
    print(f"File {os.path.basename(fileName)} Loaded {dbs}")

    for item in range(len(dbs)):
        if (dbs[item][0]) == 'AB':
            data = opusFC.getOpusData(fileName, dbs[item])

    return [data.x, data.y]

    #------------------------------------------

def loadSSC(fileName):
    import opusFC
    import os

    dbs = opusFC.listContents(fileName)
#    print(f"File {os.path.basename(fileName)} Loaded {dbs}")

    for item in range(len(dbs)):
        if (dbs[item][0]) == 'SSC':
            data = opusFC.getOpusData(fileName, dbs[item])

    return data


    #------------------------------------------
    
def averageFiles(fileDir, fileToAverage, nfiles):
    from numpy import zeros
    from sissi_util import loadSSC
    import matplotlib.pyplot as plt

#    baseDir = "/net/online4sissi/store/sissimat/"         # Root Directory for all Beamline Data
#    dataDir = "20225357_Stopponi/HighPress_QC/QC_MIR/rawdata/QC MIR/09052023/" #Specific Project Directory
#    dataDir = "20225357_Stopponi/HighPress_QC/QC_MIR/rawdata/QC MIR/" #Specific Project Directory
#    dirname = baseDir + dataDir

    fileName = fileDir + fileToAverage + ".0"
#    nfiles = int(input("\nNumber of spectra to average: "))

    dataLoaded = loadSSC(fileName)
    dataToWrite =  [fileToAverage , dataLoaded.x, zeros(len(dataLoaded.y))]

#    print("\n\033[1mAveraged Files\n\033[0m")

    for i in range(nfiles):
        fileToLoad = fileDir + fileToAverage + '.' + f'{(i)}'
        temp = loadSSC(fileToLoad)
        dataToWrite[2] += temp.y

    dataToWrite[2] /= nfiles

    plt.rcParams['figure.figsize'] = [10, 7] #sets the default window size of inline plots 
    plt.rcParams.update({'font.size': 14})

#   print()
#   print(dataToWrite)
    return(dataToWrite)


# ----------------------------------------------------

def averageFilesFromTo(fileDir, fileToAverage, startfile, endfile):
    from numpy import zeros
    from sissi_util import loadSSC
    import matplotlib.pyplot as plt

#    baseDir = "/net/online4sissi/store/sissimat/"         # Root Directory for all Beamline Data
#    dataDir = "20225357_Stopponi/HighPress_QC/QC_MIR/rawdata/QC MIR/09052023/" #Specific Project Directory
#    dataDir = "20225357_Stopponi/HighPress_QC/QC_MIR/rawdata/QC MIR/" #Specific Project Directory
#    dirname = baseDir + dataDir

    fileName = fileDir + fileToAverage + "." + f'{startfile}'
#    nfiles = int(input("\nNumber of spectra to average: "))

    dataLoaded = loadSSC(fileName)
    dataToWrite =  [fileToAverage , dataLoaded.x, zeros(len(dataLoaded.y))]

#    print("\n\033[1mAveraged Files\n\033[0m")

    for i in range(startfile, endfile + 1):
        fileToLoad = fileDir + fileToAverage + '.' + f'{(i)}'
        temp = loadSSC(fileToLoad)
        dataToWrite[2] += temp.y

    # print(len(list(range(startfile, endfile + 1))))
    dataToWrite[2] /= len(list(range(startfile, endfile + 1)))

#   print()
#   print(dataToWrite)
    return(dataToWrite)

'''
-----------------------------------------------
'''

def find_nearest(array, value):
    import numpy as np
    array = np.asarray(array)
    #idx = (np.abs(array - value)).argmin()
    idx = np.searchsorted(array, value)
    return idx

def concat_spectra(spec1, spec2, wavenumber):
    # Get the indexes corresponding to the given x value for the two spectra
    sp1_idx = find_nearest(spec1[1][:], wavenumber)
    sp2_idx = find_nearest(spec2[1][:], wavenumber)
    
    x_new = np.concatenate((spec1[1][:sp1_idx], spec2[1][sp2_idx:]))
    y_new = np.concatenate((spec1[2][:sp1_idx], spec2[2][sp2_idx:]))

    joined = [spec1[0] + spec2[0], x_new, y_new]
    return joined

# ----------------------------------------------

def read_spa(filepath):
    import numpy as np
    '''
    Input
    Read a file (string) *.spa
    ----------
    Output
    Return spectra, wavelenght (nm), titles
    '''
    with open(filepath, 'rb') as f:
        f.seek(564)
        Spectrum_Pts = np.fromfile(f, np.int32,1)[0]
        f.seek(30)
        SpectraTitles = np.fromfile(f, np.uint8,255)
        SpectraTitles = ''.join([chr(x) for x in SpectraTitles if x!=0])

        f.seek(576)
        Min_Wavenum=np.fromfile(f, np.single, 1)[0]
        Max_Wavenum=np.fromfile(f, np.single, 1)[0]
        print(Min_Wavenum, Max_Wavenum, Spectrum_Pts)
        Wavenumbers = np.flip(np.linspace(Max_Wavenum, Min_Wavenum, Spectrum_Pts))
        
        #Wavenumbers = np.linspace(Min_Wavenum, Max_Wavenum, Spectrum_Pts)

        f.seek(288);

        Flag=0
        while Flag != 3:
            Flag = np.fromfile(f, np.uint16, 1)

        DataPosition=np.fromfile(f,np.uint16, 1)
        f.seek(DataPosition[0])

        Spectra = np.fromfile(f, np.single, Spectrum_Pts)
    return Spectra, Wavenumbers, SpectraTitles

# ---------------------
def read_tab_delimited_file(file_path):
    # Initialize empty lists to store X and Y values
    x_values = []
    y_values = []

    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into X and Y values based on tab delimiter
            x, y = map(float, line.strip().split('\t'))
            
            # Append X and Y values to their respective lists
            x_values.append(x)
            y_values.append(y)

    # Convert lists to numpy arrays
    x_array = np.array(x_values)
    y_array = np.array(y_values)

    return x_array, y_array

'''
# Example usage
file_path = '/Users/miczac/Downloads/cellulose Kimmel.dat'
x_cellu, y_cellu = read_tab_delimited_file(file_path)
'''
# --------------------
