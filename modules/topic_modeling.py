from sklearn.decomposition import LatentDirichletAllocation

def apply_lda(X, num_topics=5):
    """
    Applies Latent Dirichlet Allocation (LDA) for topic modeling.
    
    Parameters
    ----------
    X : sparse matrix
        TF-IDF or Bag of Words feature matrix
    num_topics : int
        Number of topics to discover
        
    Returns
    -------
    lda_model : LatentDirichletAllocation
        Trained LDA model
    topic_assignments : ndarray
        The dominant topic for each document
    """
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    # Fit and transform to get the document-topic distributions
    doc_topic_dist = lda_model.fit_transform(X)
    
    # Assign each document to its most dominant topic
    topic_assignments = doc_topic_dist.argmax(axis=1)
    
    return lda_model, topic_assignments

def get_lda_themes(lda_model, vectorizer, top_n=5):
    """
    Extracts the top keywords for each LDA topic.
    """
    themes = {}
    feature_names = vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[:-top_n - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        themes[topic_idx] = top_words
        
    return themes
