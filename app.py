import streamlit as st
import pandas as pd

from modules.vectorization import compute_tfidf
from modules.preprocessing import preprocess_text

# imported the clustering module from clustering.py
from modules.clustering import clustering

st.title("Intelligent Research Topic Analyzer")
st.write("Milestone 1: ")

topic = st.text_input("Enter a research topic : ")

if topic:
    st.write(f"You entered: {topic}")


uploaded_file = st.file_uploader("Upload a research paper CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of uploaded file:")
    st.dataframe(df.head())
    st.write(f"Total Documents: {len(df)}")

    df = df.head(1000)

    if st.button("Run Preprocessing"):

        df["combined_text"] = df["title"] + " " + df["abstract"]
        df["processed_text"] = df["combined_text"].apply(preprocess_text)

        st.session_state["processed_df"] = df

        st.success("Preprocessing Done!")

        st.subheader("Sample Processed Output")
        st.dataframe(df[["combined_text", "processed_text"]].head())

    if "processed_df" in st.session_state:
        
        if st.button("Run TF-IDF"):
            
            processed_df = st.session_state["processed_df"]
            X, vectorizer = compute_tfidf(processed_df["processed_text"])
            
            st.success("TF-IDF Computation Completed")
            
            st.subheader("TF-IDF Matrix Shape")
            st.write(X.shape)
            
            feature_names = vectorizer.get_feature_names_out()
            
            st.subheader("Top 20 Keywords")
            st.write(feature_names[:20])
