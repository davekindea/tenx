import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

def app():
    path_to_src = os.path.abspath(os.path.join("c:\\Users\\user\\OneDrive\\Desktop\\see\\tenx", 'week2', 'src'))
    sys.path.append(path_to_src)
    try:
        from load_data import load_data_from_postgres
        print("Modules imported successfully")
    except ModuleNotFoundError as e:
        print(f"ModuleNotFoundError: {e}")
    query = "SELECT * FROM xdr_data;"  
    data = load_data_from_postgres(query)
    if data is not None:
        print("Successfully loaded the data")
    else:
        print("Failed to load data.")
    try:
        from User_engagement_experience_analysis import engagment_score,experience_score
        print("Modules imported successfully")
    except ModuleNotFoundError as e:
        print(f"ModuleNotFoundError: {e}")
    sessions_frequency = data.groupby('MSISDN/Number').size()
    total_session_duration = data.groupby('MSISDN/Number')['Dur. (ms)'].sum()
    data['total_traffic'] = data['Total UL (Bytes)'] + data['Total DL (Bytes)']
    total_traffic = data.groupby('MSISDN/Number')['total_traffic'].sum()
    data1=pd.DataFrame({
         "sessions_frequency":sessions_frequency,
        "total_session_duration":total_session_duration,
        "total_traffic": total_traffic

    })
    
    aggregated_data = pd.DataFrame({
        "sessions_frequency":sessions_frequency,
        "session_duration":total_session_duration,
        "session_total_traffic": total_traffic

    }).reset_index()

    engagement_scores=engagment_score(data1,3)
    aggregated_data["engagement_scores"]=engagement_scores
    experience_scores=experience_score(data1,3)
    aggregated_data["experience_scores"]=experience_scores
    st.title("Customer Satisfaction Analysis Dashboard")
    st.subheader("Engagement & Experience Score Calculation")
    st.write(aggregated_data[['MSISDN/Number', 'engagement_scores', 'experience_scores']])
    st.subheader("Top 10 Satisfied Customers")
    aggregated_data['satisfaction_score'] = (aggregated_data['engagement_scores'] + aggregated_data['experience_scores']) / 2
    top_10_satisfied = aggregated_data.nlargest(10, 'satisfaction_score')
    st.dataframe(top_10_satisfied[['MSISDN/Number', 'satisfaction_score']])
    st.subheader("Satisfaction Prediction Model")
    X = aggregated_data[['engagement_scores', 'experience_scores']]
    y = aggregated_data['satisfaction_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    st.write(f"R² Score: {regressor.score(X_test, y_test):.2f}")


    st.subheader("K-Means Clustering on Engagement & Experience Scores")

    kmeans = KMeans(n_clusters=2, random_state=42)
    aggregated_data['cluster'] = kmeans.fit_predict(aggregated_data[['engagement_scores', 'experience_scores']])


    fig, ax = plt.subplots()
    sns.scatterplot(x='engagement_scores', y='experience_scores', hue='cluster', data=aggregated_data, ax=ax)
    plt.title("K-Means Clustering (k=2) on Engagement & Experience Scores")
    st.pyplot(fig)


    st.subheader("Cluster Aggregation: Average Satisfaction & Experience Score")

    cluster_aggregation = aggregated_data.groupby('cluster').agg({
    'satisfaction_score': 'mean',
    'experience_scores': 'mean'
}).reset_index()
    st.dataframe(cluster_aggregation)

