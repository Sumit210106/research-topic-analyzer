import streamlit as st
import pandas as pd

st.title("Intelligent Research Topic Analyzer")
st.write("Milestone 1: ")

topic = st.text_input("Enter a research topic : ")

if topic:
    st.write(f"You entered: {topic}")


uploaded_file = st.file_uploader("Upload a research paper CSV" , type = ['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded file:")
    st.dataframe(df.head())
    st.write(f"Total Documents: {len(df)}")