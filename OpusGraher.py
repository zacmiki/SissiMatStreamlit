import streamlit as st
import plotly.graph_objects as go
import numpy as np
import opusFC
import io

def graphopus():
    st.title("Raw OPUS File Grapher")
    st.subheader(":rainbow[Plot of the saved OPUS File as it is]")
    st.subheader("Graphing all datasets inside the File")

    uploaded_file = st.file_uploader("Choose an OPUS file")

    if uploaded_file is not None:
        file_name = uploaded_file.name
        st.session_state.fileloaded = file_name
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
    st.write(f"The Loaded File is: :red[{st.session_state.fileloaded}]")
    
    data = {name: spec for name, spec in zip(spectra_names, spectra)}
    
    for key in data:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[key]['x'], y=data[key]['y'], mode='lines', name=key))
        
        fig.update_layout(
            title_text=key,
            margin=dict(l=40, r=40, t=40, b=40),
            autosize=True,
            height=400,
        )
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="white", 
            showgrid=True, mirror=True,
        )
        fig.update_yaxes(
            showline=True, linewidth=2, linecolor="white",
            zeroline=True, zerolinecolor="brown",
            showgrid=True, mirror=True,
        )
        st.plotly_chart(fig)
        
    if st.button("Press to Download the Spectra as TXT", key="download"):
        opusfileexport(file_path)

def opusfileexport(file_path):
    dbs = opusFC.listContents(file_path)
    dataSets = len(dbs)

    for index, sets in enumerate(dbs):
        data = opusFC.getOpusData(file_path, sets)
        suffix = sets[0]
        filename = f"{st.session_state.fileloaded}.{suffix}.txt"
        spectrum = np.column_stack((data.x, data.y))
        
        # Convert the numpy array to a CSV string
        output = io.StringIO()
        np.savetxt(output, spectrum, delimiter=',', fmt='%s')
        csv_string = output.getvalue()
        
        st.download_button(
            label=f"Download {filename}",
            data=csv_string,
            file_name=filename,
            mime="text/csv",
            key=f"download_{index}"
        )

    return

# Run the main function
if __name__ == "__main__":
    graphopus()
