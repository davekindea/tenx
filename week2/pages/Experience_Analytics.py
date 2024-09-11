import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
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
    st.title("Telecommunication User Experience Analytics")

# Task 3.1: Aggregate information per customer
    st.header(" Aggregated Information per Customer")
    numerical_cols = data.select_dtypes(include=[np.number]).columns 

    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean()) 
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns  
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

    Average_TCP_retransmission_per_user=data.groupby('MSISDN/Number')[['TCP DL Retrans. Vol (Bytes)', 'TCP DL Retrans. Vol (Bytes)']].mean().sum(axis=1)
    Average_RTT_retransmission_per_user=data.groupby('MSISDN/Number')[['Avg RTT DL (ms)', 'Avg RTT UL (ms)']].mean().sum(axis=1)
    handset_type_per_user = data.groupby('MSISDN/Number')['Handset Type'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    Average_throughput_retransmission_per_user=data.groupby('MSISDN/Number')[['DL TP < 50 Kbps (%)', '50 Kbps < DL TP < 250 Kbps (%)',"DL TP > 1 Mbps (%)","250 Kbps < DL TP < 1 Mbps (%)",'UL TP < 10 Kbps (%)', '10 Kbps < UL TP < 50 Kbps (%)',"50 Kbps < UL TP < 300 Kbps (%)","UL TP > 300 Kbps (%)"]].mean().sum(axis=1)
    
    customer_aggregates =pd.DataFrame({
    "TCP_Retransmission": Average_TCP_retransmission_per_user,
    "RTT": Average_RTT_retransmission_per_user,
    "Handset_Type": handset_type_per_user,
    "Throughput": Average_throughput_retransmission_per_user


    
    
   

}).reset_index()

# Display data
    st.dataframe(customer_aggregates.head())

# Task 3.2: Top, Bottom, and Most Frequent values
    st.header("Top, Bottom, and Most Frequent Values")

# Top 10, Bottom 10, and Most Frequent
    top_10_tcp = customer_aggregates['TCP_Retransmission'].nlargest(10)
    bottom_10_tcp = customer_aggregates['TCP_Retransmission'].nsmallest(10)
    most_frequent_tcp = customer_aggregates['TCP_Retransmission'].mode().iloc[0]

    top_10_rtt = customer_aggregates['RTT'].nlargest(10)
    bottom_10_rtt = customer_aggregates['RTT'].nsmallest(10)
    most_frequent_rtt = customer_aggregates['RTT'].mode().iloc[0]

    top_10_throughput = customer_aggregates['Throughput'].nlargest(10)
    bottom_10_throughput = customer_aggregates['Throughput'].nsmallest(10)
    most_frequent_throughput = customer_aggregates['Throughput'].mode().iloc[0]

# Display results
    st.write("### Top 10 TCP Retransmission")
    st.write(top_10_tcp)

    st.write("### Bottom 10 TCP Retransmission")
    st.write(bottom_10_tcp)

    st.write(f"### Most Frequent TCP Retransmission: {most_frequent_tcp}")

    st.write("### Top 10 RTT")
    st.write(top_10_rtt)

    st.write("### Bottom 10 RTT")
    st.write(bottom_10_rtt)

    st.write(f"### Most Frequent RTT: {most_frequent_rtt}")

    st.write("### Top 10 Throughput")
    st.write(top_10_throughput)

    st.write("### Bottom 10 Throughput")
    st.write(bottom_10_throughput)

    st.write(f"### Most Frequent Throughput: {most_frequent_throughput}")

# Task 3.3: Distribu of throughput and TCP retransmission per handset type
    st.header(" Throughput and TCP Retransmission per Handset Type")

# Distribution of average throughput per handset type
    throughput_per_handset = customer_aggregates.groupby('Handset_Type')['Throughput'].mean().sort_values(ascending=False)
    tcp_retransmission_per_handset = customer_aggregates.groupby('Handset_Type')['TCP_Retransmission'].mean().sort_values(ascending=False)

# Plot distribution of throughput per handset type
    fig1 = px.bar(throughput_per_handset, x=throughput_per_handset.index, y=throughput_per_handset.values, 
              labels={'x':'Handset Type', 'y':'Average Throughput'}, title="Average Throughput per Handset Type")
    st.plotly_chart(fig1)

# Plot TCP retransmission per handset type
    fig2 = px.bar(tcp_retransmission_per_handset, x=tcp_retransmission_per_handset.index, y=tcp_retransmission_per_handset.values, 
              labels={'x':'Handset Type', 'y':'Average TCP Retransmission'}, title="Average TCP Retransmission per Handset Type")
    st.plotly_chart(fig2)

# Task 3.4: K-means clustering (k = 3)
    st.header("Task 3.4: K-means Clustering for User Experience Segmentation")

# Selecting features for clustering
    features = customer_aggregates[['TCP_Retransmission', 'RTT', 'Throughput']]

# Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

# K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    customer_aggregates['Cluster'] = kmeans.fit_predict(features_scaled)

# Display cluster results
    st.write(customer_aggregates[['MSISDN/Number', 'Cluster']].head())

# Plot clustering results
    fig3 = px.scatter_3d(customer_aggregates, x='TCP_Retransmission', y='RTT', z='Throughput', color='Cluster',
                     title="Customer Segmentation Based on Network Experience")
    st.plotly_chart(fig3)

# Description of clusters
    st.write("""
### Cluster Interpretation:
- **Cluster 0**: Customers with high throughput and low TCP retransmission, indicating a good user experience.
- **Cluster 1**: Customers with moderate performance in all metrics, representing average user experience.
- **Cluster 2**: Customers with low throughput and high TCP retransmission, indicating poor user experience.
""")
