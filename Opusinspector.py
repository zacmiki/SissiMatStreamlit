import streamlit as st
import opusFC
from OpusGraher import opusfileexport

def inspectopusfile():
    st.title("OPUS File Inspector")
    st.subheader(":rainbow[Experimental Feature]")
    st.subheader("Showing Meaningful Experimental Info included in the Opus File Header")

    uploaded_file = st.file_uploader("Choose an OPUS file")

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_extension = file_name.split(".")[-1]
        
        # Check if the file extension is an integer
        if file_extension.isdigit():
            with open("temp.opus", "wb") as f:
                f.write(uploaded_file.getbuffer())
            opusreadinfo("temp.opus")
        else:
            st.write("Please upload a file with an integer extension.")
    else:
        st.write("Please upload an OPUS file.")

def opusreadinfo(file_path):
    from sissi_util import parvalues

    dbs = opusFC.listContents(file_path) 
    dataSets = len(dbs)
    data = [opusFC.getOpusData(file_path, item) for item in dbs]
    spectra_names = [num[0] for num in dbs]
    
    if st.button("Press to Download the Spectra as TXT", key = "download"):
        opusfileexport(file_path)
    
    st.markdown(f"### :blue[There are {dataSets} datasets inside the file]")
    st.markdown(f"##### {dbs}")
    st.markdown(f"### :yellow[The saved spectra are: {spectra_names}]")
    
    #spectra = [...... for item in data]
    #data = {name: spec for name, spec in zip(spectra_names, spectra)}
    st.divider()
    parvalues(file_path) 
