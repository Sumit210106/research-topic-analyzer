
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf(documents, max_features=2000):
    """
    Compute TF-IDF vectors for a list of documents.
    """
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(documents)
    return X, vectorizer
