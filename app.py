import streamlit as st
import pandas as pd

from modules.vectorization import compute_tfidf
from modules.preprocessing import preprocess_text
from modules.clustering import k_means
from modules.clustering import find_optimal_k
from modules.clustering import get_cluster_themes
from modules.summarization import generate_extractive_summary

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

                st.subheader("Cluster Themes")
                vectorizer = st.session_state["vectorizer"]
                themes = get_cluster_themes(model, vectorizer, top_n=5)
                for cluster_idx, top_words in themes.items():
                    st.write(f"**Cluster {cluster_idx} Themes**: {', '.join(top_words)}")

                st.subheader("Extractive Summaries by Cluster")
                for cluster_idx in range(best_k):
                    # Combine text of top 15 papers for summary generation
                    sample_texts = clustered_df[clustered_df["cluster"] == cluster_idx]["combined_text"].head(15).tolist()
                    cluster_text = " ".join(str(text) for text in sample_texts)
                    
                    with st.spinner(f"Generating summary for Cluster {cluster_idx}..."):
                        summary = generate_extractive_summary(cluster_text, num_sentences=3)
                        
                    st.write(f"**Cluster {cluster_idx} Summary**:")
                    st.info(summary)