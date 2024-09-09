import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
def engagment_score(data,k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    less_engaged_cluster = 1
    less_engaged_centroid = centroids[less_engaged_cluster]
    engagement_scores = []
    
    for _, user_data in data.iterrows():
        user_data_point = np.array([user_data['sessions_frequency'], user_data['total_session_duration'], user_data['total_traffic']])
        if user_data_point.shape != less_engaged_centroid.shape:
            raise ValueError(f"Dimension mismatch: user_data_point shape {user_data_point.shape} vs centroid shape {less_engaged_centroid.shape}")
        engagement_score = euclidean(user_data_point,less_engaged_centroid)
        engagement_scores.append(engagement_score)
    return engagement_scores
def experience_score(data,k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    worst_engaged_cluster = 0
    worst_engaged_centroid = centroids[worst_engaged_cluster]
    experience_scores = []
    
    for _, user_data in data.iterrows():
        user_data_point = np.array([user_data['sessions_frequency'], user_data['total_session_duration'], user_data['total_traffic']])
        if user_data_point.shape != worst_engaged_centroid.shape:
            raise ValueError(f"Dimension mismatch: user_data_point shape {user_data_point.shape} vs centroid shape {worst_engaged_centroid.shape}")
        experience_score = euclidean(user_data_point,worst_engaged_centroid)
        experience_scores.append(experience_score)
    return experience_scores

