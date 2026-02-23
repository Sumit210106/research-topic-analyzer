from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np



def estimate_eps(X, min_samples=5):
    """
    Estimate eps using k-distance method.

    Parameters
    ----------
    X : sparse matrix
        TF-IDF feature matrix

    min_samples : int
        Neighbor count

    Returns
    -------
    eps : float
    Estimated eps value
    distances_sorted : ndarray
    Sorted k-distance curve
    """

    neighbors = NearestNeighbors(
        n_neighbors=min_samples,
        metric="cosine"
        )

    neighbors_fit = neighbors.fit(X)

    distances, _ = neighbors_fit.kneighbors(X)


    distances = distances[:, -1]
    distances_sorted = np.sort(distances)

    eps = np.percentile(distances_sorted, 90)

    return eps, distances_sorted



def auto_dbscan(X, min_samples=5):
    """
    Fully automatic DBSCAN clustering.

    Parameters
    ----------
    X : sparse matrix
    TF-IDF feature matrix

    min_samples : int
    Minimum samples for DBSCAN

    Returns
    -------
    labels : ndarray
    Cluster labels (-1 = noise)

    model : DBSCAN
    Trained DBSCAN model

    eps : float
    Estimated eps

    distances_sorted : ndarray
    k-distance curve
    """

    eps, distances_sorted = estimate_eps(X, min_samples)

    dbscan = DBSCAN(
    eps=eps,
    min_samples=min_samples,
    metric="cosine"
    )

    labels = dbscan.fit_predict(X)

    return labels, dbscan, eps, distances_sorted