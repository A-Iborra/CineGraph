# clustering.py

from sklearn.cluster import KMeans
import numpy as np

def perform_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    return kmeans.labels_
