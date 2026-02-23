import streamlit as st
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from modules.vectorization import compute_tfidf
from modules.preprocessing import preprocess_text
from modules.clustering import k_means, find_optimal_k
from modules.dbscan import auto_dbscan

st.title("Intelligent Research Topic Analyzer")
st.write("Milestone 1")


topic = st.text_input("Enter a research topic")
if topic:
    st.write(f"You entered: {topic}")


uploaded_file = st.file_uploader("Upload research paper CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write(f"Total Documents: {len(df)}")

    df = df.head(1000)

    if st.button("Run Preprocessing"):
        df["combined_text"] = df["title"] + " " + df["abstract"]
        df["processed_text"] = df["combined_text"].apply(preprocess_text)

        st.session_state["processed_df"] = df

        st.success("Preprocessing Done")
        st.dataframe(df[["combined_text", "processed_text"]].head())


if "processed_df" in st.session_state:

    if st.button("Run TF-IDF"):

        processed_df = st.session_state["processed_df"]
        X, vectorizer = compute_tfidf(processed_df["processed_text"])

        st.session_state["X"] = X
        st.session_state["vectorizer"] = vectorizer

        st.success("TF-IDF Computed")
        st.write("Matrix Shape:", X.shape)

        feature_names = vectorizer.get_feature_names_out()
        st.write("Top Keywords:", feature_names[:20])


if "X" in st.session_state:

    st.subheader("K-Means (Automatic)")

    if st.button("Run Auto K-Means"):

        X = st.session_state["X"]
        processed_df = st.session_state["processed_df"]

        best_k, scores = find_optimal_k(X, range(2, 11))

        if best_k is None:
            best_k = 3

        st.write(f"Suggested optimal k: {best_k}")

        if scores:
            st.line_chart(pd.DataFrame({"silhouette": scores}))

        labels, _ = k_means(X, best_k)

        clustered_df = processed_df.copy()
        clustered_df["kmeans_cluster"] = labels
        st.session_state["kmeans_df"] = clustered_df

        st.success("K-Means Completed")


if "kmeans_df" in st.session_state:

    kmeans_df = st.session_state["kmeans_df"]

    st.subheader("K-Means Distribution")
    st.write(kmeans_df["kmeans_cluster"].value_counts())
    st.dataframe(kmeans_df[["title", "kmeans_cluster"]].head())
    st.bar_chart(kmeans_df["kmeans_cluster"].value_counts())


if "X" in st.session_state:

    st.subheader("DBSCAN (Automatic)")

    if st.button("Run Auto DBSCAN"):

        X = st.session_state["X"]
        processed_df = st.session_state["processed_df"]

        svd = TruncatedSVD(n_components=100, random_state=42)
        X_reduced = svd.fit_transform(X)

        st.write("Reduced dimension:", X_reduced.shape[1])

        labels, _, eps, distances = auto_dbscan(X_reduced)

        st.write(f"Estimated eps: {eps}")

        dbscan_df = processed_df.copy()
        dbscan_df["dbscan_cluster"] = labels
        st.session_state["dbscan_df"] = dbscan_df

        st.success("DBSCAN Completed")

        noise_count = (labels == -1).sum()
        st.write(f"Noise documents: {noise_count}")

        st.line_chart(pd.DataFrame(distances, columns=["k-distance"]))


if "dbscan_df" in st.session_state:

    dbscan_df = st.session_state["dbscan_df"]

    st.subheader("DBSCAN Distribution")
    st.write(dbscan_df["dbscan_cluster"].value_counts())
    st.dataframe(dbscan_df[["title", "dbscan_cluster"]].head())
    st.bar_chart(dbscan_df["dbscan_cluster"].value_counts())