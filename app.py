import streamlit as st
import pandas as pd

from modules.vectorization import compute_tfidf
from modules.preprocessing import preprocess_text
from modules.clustering import k_means
from modules.clustering import find_optimal_k

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
            
            st.session_state["X"] = X
            st.session_state["vectorizer"] = vectorizer
            st.success("TF-IDF Computation Completed")
            
            st.subheader("TF-IDF Matrix Shape")
            st.write(X.shape)
            
            feature_names = vectorizer.get_feature_names_out()
            
            st.subheader("Top 20 Keywords")
            st.write(feature_names[:20])

        if "X" in st.session_state:

            st.subheader("K-Means Clustering")
            if st.button("Run K-Means"):
                X = st.session_state["X"]
                processed_df = st.session_state["processed_df"]
                
                best_k, scores = find_optimal_k(X, range(2, 11))
                
                if best_k is None:
                    best_k = 3
                    
                st.write(f"Suggested optimal k: {best_k}")
                    
                st.line_chart(scores)
                
                labels, model = k_means(X, best_k)

                clustered_df = processed_df.copy()
                clustered_df["cluster"] = labels
                st.session_state["clustered_df"] = clustered_df

                st.success("K-Means Clustering Completed")

                st.subheader("Cluster Distribution")
                st.write(clustered_df["cluster"].value_counts())
                st.subheader("Sample Clustered Papers")
                st.dataframe(clustered_df[["title", "cluster"]].head())

                
                     