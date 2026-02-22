from sklearn.decomposition import LatentDirichletAllocation

def apply_lda(X, num_topics=5):
    model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    topic_doc = model.fit_transform(X)
    topic_assignments = topic_doc.argmax(axis=1)
    
    return model, topic_assignments

def get_lda_themes(lda_model, vectorizer, top_n=5):
    """
    Extracts human-readable themes for each LDA topic.
    Combines the top keywords into a natural-sounding phrase.
    """
    themes = {}
    feature_names = vectorizer.get_feature_names_out()
    
    for idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[:-top_n - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        
        # Humanize the theme representation
        friendly_theme = " and ".join([", ".join(top_words[:-1]), top_words[-1]]) if len(top_words) > 1 else top_words[0]
        themes[idx] = [friendly_theme]
        
    return themes
