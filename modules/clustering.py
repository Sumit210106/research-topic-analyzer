from sklearn.cluster import KMeans

def clustering(A, total_clusters=5):
    kmeans = KMeans(n_clusters=total_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(A)
    return kmeans, labels