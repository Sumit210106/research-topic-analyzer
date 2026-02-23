from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def run_lda(documents, n_topics=5, max_features=2000):
    """
    Apply LDA topic modeling.

    Parameters
    ----------
    documents : list-like
        Preprocessed documents

    n_topics : int
        Number of topics

    max_features : int
        Vocabulary limit

    Returns
    -------
    lda_model : LDA
        Trained model

    dtm : matrix
        Document-term matrix

    vectorizer : CountVectorizer
        Vocabulary
    """

    vectorizer = CountVectorizer(max_features=max_features)
    dtm = vectorizer.fit_transform(documents)

    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )

    lda_model.fit(dtm)

    return lda_model, dtm, vectorizer


def get_topics(lda_model, vectorizer, n_top_words=10):
    """
    Extract top words per topic.
    """

    feature_names = vectorizer.get_feature_names_out()
    topics = {}

    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [
            feature_names[i]
            for i in topic.argsort()[:-n_top_words - 1:-1]
        ]
        topics[f"Topic {topic_idx+1}"] = top_words

    return topics