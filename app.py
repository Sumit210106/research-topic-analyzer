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

            st.subheader("Topic Modeling & Clustering")
            
            analysis_method = st.radio("Choose Analysis Method", ["K-Means Clustering", "LDA Topic Modeling"])
            
            if st.button("Run Analysis"):
                X = st.session_state["X"]
                processed_df = st.session_state["processed_df"]
                vectorizer = st.session_state["vectorizer"]
                
                if analysis_method == "K-Means Clustering":
                    st.write("---")
                    st.subheader("K-Means Clustering Results")
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
                    
                    st.subheader("Evaluation Metrics")
                    st.metric(label="Silhouette Score", value=f"{scores.get(best_k, 0):.4f}")
                    st.info("The **Silhouette Score** measures how similar an object is to its own cluster compared to other clusters. A higher score (closer to 1) indicates dense, well-separated clusters.")

                    st.subheader("Cluster Distribution & Interpretation")
                    distribution = clustered_df["cluster"].value_counts().reset_index()
                    distribution.columns = ["Cluster", "Count"]
                    st.bar_chart(distribution.set_index("Cluster"))
                    
                    st.write("**Distribution Interpretation:**")
                    st.write(f"The largest cluster (Cluster {distribution.iloc[0]['Cluster']}) contains {distribution.iloc[0]['Count']} papers, indicating a dominant research theme in the dataset. "
                             f"The smallest (Cluster {distribution.iloc[-1]['Cluster']}) has {distribution.iloc[-1]['Count']} papers, suggesting a more niche sub-topic.")
                    
                    st.subheader("Sample Clustered Papers")
                    st.dataframe(clustered_df[["title", "cluster"]].head())

                    st.subheader("Cluster Themes")
                    themes = get_cluster_themes(model, vectorizer, top_n=5)
                    for cluster_idx, top_words in themes.items():
                        st.write(f"**Cluster {cluster_idx} Themes**: {', '.join(top_words)}")
                        
                    best_n_groups = best_k

                else:
                    st.write("---")
                    st.subheader("LDA Topic Modeling Results")
                    from modules.topic_modeling import apply_lda, get_lda_themes
                    
                    num_topics = 5
                    st.write(f"Using defined number of topics: {num_topics}")
                    
                    with st.spinner("Running LDA..."):
                        lda_model, labels = apply_lda(X, num_topics=num_topics)
                    
                    clustered_df = processed_df.copy()
                    clustered_df["cluster"] = labels
                    st.session_state["clustered_df"] = clustered_df

                    st.success("LDA Topic Modeling Completed")

                    st.subheader("Topic Distribution")
                    distribution = clustered_df["cluster"].value_counts().reset_index()
                    distribution.columns = ["Topic", "Count"]
                    st.bar_chart(distribution.set_index("Topic"))
                    
                    st.write("**Topic Coherence & Distribution Interpretation:**")
                    st.write("LDA is a probabilistic model. Each paper is assigned to the topic it has the highest probability of belonging to. "
                             f"Topic {distribution.iloc[0]['Topic']} is the most prevalent in this dataset.")
                    st.info("**Topic Coherence Explanation:** A coherent topic will have top keywords that frequently co-occur in the same documents. If the themes look like a random assortment of words, the model may need more topics or better text preprocessing.")
                    
                    st.subheader("Sample Papers by Topic")
                    st.dataframe(clustered_df[["title", "cluster"]].head())

                    st.subheader("Topic Themes")
                    themes = get_lda_themes(lda_model, vectorizer, top_n=5)
                    for topic_idx, top_words in themes.items():
                        st.write(f"**Topic {topic_idx} Themes**: {', '.join(top_words)}")
                        
                    best_n_groups = num_topics

                st.subheader("Extractive Summaries by Group")
                for group_idx in range(best_n_groups):
                    # Combine text of top 15 papers for summary generation
                    sample_texts = clustered_df[clustered_df["cluster"] == group_idx]["combined_text"].head(15).tolist()
                    group_text = " ".join(str(text) for text in sample_texts)
                    
                    with st.spinner(f"Generating summary for Group {group_idx}..."):
                        summary = generate_extractive_summary(group_text, num_sentences=3)
                        
                    st.write(f"**Group {group_idx} Summary**:")
                    st.info(summary)