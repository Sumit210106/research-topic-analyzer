from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def k_means(A, total_clusters=5):
    """
    Apply K-Means clustering.
    
    Parameters
    ----------
    X : sparse matrix
        TF-IDF feature matrix

    total_clusters : int
        Number of clusters

    random_state : int
        Ensures reproducibility

    Returns
    -------
    labels : ndarray
        Cluster labels for each document

    model : KMeans
        Trained KMeans model
        
    """
    
    kmeans = KMeans(n_clusters=total_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(A)
    return labels,kmeans


def find_optimal_k(X , k_range):
    """
    optimal number of clusters using silhouette score

    Parameters
    ----------
    X : sparse matrix
        TF-IDF feature matrix

    k_range : iterable
        Range of k values to evaluate

    Returns
    -------
    best_k : int
        Optimal cluster count

    scores : dict
        Silhouette score for each k
    """

    scores = {}
    best_k = None
    best_score = -1
    n_samples = X.shape[0]
    
    for k in k_range:
        if k >= n_samples:
            continue
        
        kmeans = KMeans(
            n_clusters=k ,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(X)
        
        #valid if more than 1 cluster exists
        if len(set(labels)) > 1 :
            score = silhouette_score(X, labels)
            scores[k] = score
            
            if score > best_score : 
                best_score = score 
                best_k = k
            
    return best_k , scores
    
    