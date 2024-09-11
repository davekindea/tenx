import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import sys
from sklearn.cluster import KMeans
import os
from sklearn.decomposition import PCA
def app():
    st.title("User Overview Analysis")
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

# Aggregate metrics per customer
    sessions_frequency = data.groupby('MSISDN/Number').size()
    total_session_duration = data.groupby('MSISDN/Number')['Dur. (ms)'].sum()
    data['total_traffic'] = data['Total UL (Bytes)'] + data['Total DL (Bytes)']
    total_traffic = data.groupby('MSISDN/Number')['total_traffic'].sum()
    
    aggregated_data = pd.DataFrame({
        "sessions_frequency":sessions_frequency,
        "session_duration":total_session_duration,
        "session_total_traffic": total_traffic

    })

# Top 10 customers per engagement metric
    top_10_sessions = aggregated_data.nlargest(10, 'sessions_frequency')
    top_10_duration = aggregated_data.nlargest(10, 'session_duration')
    top_10_traffic = aggregated_data.nlargest(10, 'session_total_traffic')

    st.write("Top 10 Customers by Sessions Frequency:")
    st.write(top_10_sessions)

    st.write("Top 10 Customers by Session Duration:")
    st.write(top_10_duration)

    st.write("Top 10 Customers by Session Total Traffic:")
    st.write(top_10_traffic)
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(aggregated_data[['sessions_frequency', 'session_duration', 'session_total_traffic']])

# Perform K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(normalized_data)
    aggregated_data['cluster'] = clusters

# Compute statistics per cluster
    cluster_stats = aggregated_data.groupby('cluster').agg({
    'sessions_frequency': ['min', 'max', 'mean', 'sum'],
    'session_duration': ['min', 'max', 'mean', 'sum'],
    'session_total_traffic': ['min', 'max', 'mean', 'sum']
})

    st.write("Cluster Statistics:")
    st.write(cluster_stats)

# Plot cluster distribution
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='sessions_frequency', y='session_duration', hue='cluster', data=aggregated_data, palette='viridis')
    plt.title('Customer Engagement Clusters')
    st.pyplot(plt)


    plt.figure(figsize=(10, 6))
    sns.barplot(x='cluster', y='sessions_frequency', data=aggregated_data)
    plt.title('Total Session Frequency per Cluster')
    plt.ylabel('Total Session Frequency')
    plt.xlabel('Cluster')
    st.pyplot(plt)


    st.title("Total traffic per Cluster")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='cluster', y=('session_total_traffic', 'mean'), data=cluster_stats)
    plt.title('total_traffic per Cluster')
    plt.ylabel('total_traffic')
    plt.xlabel('Cluster')
    st.pyplot(plt)

    st.title("Total Session Duration per Cluster")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='cluster', y='session_duration', data=aggregated_data)
    plt.title('Total Session Duration per Cluster')
    plt.ylabel('Total Session Duration')
    plt.xlabel('Cluster')
    st.pyplot(plt)


    social_media_usage_per_user = data.groupby('MSISDN/Number')[['Social Media DL (Bytes)', 'Social Media UL (Bytes)']].sum().sum(axis=1)
    YouTube_usage_per_user = data.groupby('MSISDN/Number')[['Youtube DL (Bytes)', 'Youtube UL (Bytes)']].sum().sum(axis=1)
    Netflix_usage_per_user = data.groupby('MSISDN/Number')[['Netflix DL (Bytes)', 'Netflix UL (Bytes)']].sum().sum(axis=1)
    Google_usage_per_user = data.groupby('MSISDN/Number')[['Google DL (Bytes)', 'Google UL (Bytes)']].sum().sum(axis=1)
    Email_usage_per_user = data.groupby('MSISDN/Number')[['Email DL (Bytes)', 'Email UL (Bytes)']].sum().sum(axis=1)
    Gaming_usage_per_user = data.groupby('MSISDN/Number')[['Gaming DL (Bytes)', 'Gaming UL (Bytes)']].sum().sum(axis=1)
    Other_media_usage_per_user = data.groupby('MSISDN/Number')[['Other DL (Bytes)', 'Other DL (Bytes)']].sum().sum(axis=1)

    each_application_usage_per_user=pd.DataFrame({
    'social_media_usage_per_user': social_media_usage_per_user,
    'YouTube_usage_per_user': YouTube_usage_per_user,
    'Netflix_usage_per_user':Netflix_usage_per_user,
    'Google_usage_per_user': Google_usage_per_user,
    'Email_usage_per_user': Email_usage_per_user,
    'Gaming_usage_per_user': Gaming_usage_per_user,
    'Other_media_usage_per_user': Other_media_usage_per_user
   

})
    each_application_usage_per_user = each_application_usage_per_user.reset_index()

    data_long = each_application_usage_per_user.melt(id_vars='MSISDN/Number', var_name='application', value_name='usage')

# Find top 10 most engaged users per application
    top_users_per_app = data_long.groupby(['application', 'MSISDN/Number']).agg({
    'usage': 'sum'
}).reset_index().groupby('application').apply(lambda x: x.nlargest(10, 'usage')).reset_index(drop=True)

# Display top 10 most engaged users per application
    st.write("Top 10 Most Engaged Users per Application:")
    st.dataframe(top_users_per_app)

# Find the top 3 most used applications based on total usage
    top_apps = data_long.groupby('application').agg({
    'usage': 'sum'
}).nlargest(3, 'usage').reset_index()

# Plotting the top 3 most used applications
    plt.figure(figsize=(10, 6))
    sns.barplot(x='application', y='usage', data=top_apps, palette='viridis')

# Add title and labels to the plot
    plt.title('Top 3 Most Used Applications')
    plt.ylabel('Total Usage (Bytes)')
    plt.xlabel('Application')

# Display the plot in Streamlit
    st.pyplot(plt)


# Elbow Method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(normalized_data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    st.pyplot(plt)

# Interpret findings
    st.write("The elbow method plot helps determine the optimal number of clusters (k). The point where the plot bends (the 'elbow') indicates the ideal k value.")