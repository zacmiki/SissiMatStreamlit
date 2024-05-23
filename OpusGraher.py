import streamlit as st
import matplotlib.pyplot as plt
import opusFC

def graphopus():
    st.title("Raw OPUS File Grapher")
    st.subheader(":rainbow[Experimental Feature]")
    st.subheader("Graphing all datasets inside the File")

    uploaded_file = st.file_uploader("Choose an OPUS file")

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_extension = file_name.split(".")[-1]
        
        # Check if the file extension is an integer
        if file_extension.isdigit():
            with open("temp.opus", "wb") as f:
                f.write(uploaded_file.getbuffer())
            opusgrapher("temp.opus")
        else:
            st.write("Please upload a file with an integer extension.")
    else:
        st.write("Please upload an OPUS file.")

def opusgrapher(file_path):
    dbs = opusFC.listContents(file_path)
    dataSets = len(dbs)
    
    st.write("We have ", dataSets, "datasets in the file")
    
    data = [opusFC.getOpusData(file_path, item) for item in dbs]
    spectra_names = [num[0] for num in dbs]
    spectra = [{'x': item.x, 'y': item.y} for item in data]
    
    data = {name: spec for name, spec in zip(spectra_names, spectra)}
    
    fig, axs = plt.subplots(len(data), figsize=(10, len(data) * 3))
    
    for i, key in enumerate(data):
        if len(data) == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.plot(data[key]["x"], data[key]["y"])
        ax.set_title(key)
        ax.grid()

    plt.tight_layout()
    st.pyplot(fig)



