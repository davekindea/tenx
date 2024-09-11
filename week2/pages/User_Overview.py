import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
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
    data_cleaned = data.dropna(subset=['Handset Type'])
    data_cleaned = data_cleaned[data_cleaned['Handset Type'] != 'undefined']
    data_cleaned_man = data.dropna(subset=['Handset Manufacturer'])
    data_cleaned_man = data_cleaned[data_cleaned['Handset Manufacturer'] != 'undefined'] 

    top_handsets = data['Handset Type'].value_counts().head(10)
    st.subheader("Top 10 Handsets")
    st.bar_chart(top_handsets)
    top_manufacturers = data['Handset Manufacturer'].value_counts().head(3)
    st.subheader("Top 3 Handset Manufacturers")
    st.bar_chart(top_manufacturers)
    top_3_manufacturers =top_manufacturers.index.tolist()
    print(top_3_manufacturers)
    top_3_data = data_cleaned[data_cleaned['Handset Manufacturer'].isin(top_3_manufacturers)]
    grouped = top_3_data.groupby(['Handset Manufacturer', 'Handset Type']).size().reset_index(name='Count')

   
    


    top_5_handsets_per_manufacturer = grouped.groupby('Handset Manufacturer').apply(lambda x: x.nlargest(5, 'Count')).reset_index(drop=True)


    for manufacturer in top_3_manufacturers:
        print(f"Top 5 handsets for {manufacturer}:")
        manufacturer_data = top_5_handsets_per_manufacturer[top_5_handsets_per_manufacturer['Handset Manufacturer'] == manufacturer]
        st.subheader(f"Top 5 Handsets for {manufacturer}")
        st.bar_chart(manufacturer_data)


    # top_5_handsets_per_manufacturer = grouped.groupby('Handset Manufacturer').apply(lambda x: x.nlargest(5, 'Count')).reset_index(drop=True)
    # for manufacturer in top_manufacturers:
    #     manufacturer_data = top_5_handsets_per_manufacturer[top_5_handsets_per_manufacturer['Handset Manufacturer'] == manufacturer]
    #     st.subheader(f"Top 5 Handsets for {manufacturer}")
    #     st.bar_chart(manufacturer_data)
    st.subheader("Recommendations")
    st.write("Provide insights and recommendations here.")

    data=data.reset_index()
   # Count of 'Bearer id'
    

    st.title("Exploratory Data Analysis")

# Session Duration Chart
    st.subheader("Session Duration")
    st.bar_chart(data['Dur. (ms)'])

# Total Download and Upload Data Chart
    st.subheader("Total Download and Upload Data")
    st.bar_chart(data[['Total DL (Bytes)', 'Total UL (Bytes)']])

# Clean the data by dropping rows with missing values
    df_cleaned_agg = data.dropna(subset=['Bearer Id', 'Handset Manufacturer', 'Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)'])

# Group by 'MSISDN/Number' and aggregate data
    app_data = df_cleaned_agg.groupby('MSISDN/Number').agg(
    num_sessions=('Bearer Id', 'count'),
    total_session_duration=('Dur. (ms)', 'sum'),
    total_download_data=('Total DL (Bytes)', 'sum'),
    total_upload_data=('Total UL (Bytes)', 'sum'),

    social_media_download_data=("Social Media DL (Bytes)", "sum"),
    social_media_upload_data=("Social Media UL (Bytes)", "sum"),

    youtube_download_data=("Youtube DL (Bytes)", "sum"),
    youtube_upload_data=("Youtube UL (Bytes)", "sum"),

    netflix_download_data=("Netflix DL (Bytes)", "sum"),
    netflix_upload_data=("Netflix UL (Bytes)", "sum"),

    email_download_data=("Email DL (Bytes)", "sum"),
    email_upload_data=("Email UL (Bytes)", "sum"),

    google_download_data=("Google DL (Bytes)", "sum"),
    google_upload_data=("Google UL (Bytes)", "sum"),

    gaming_download_data=("Gaming DL (Bytes)", "sum"),
    gaming_upload_data=("Gaming UL (Bytes)", "sum"),

    other_download_data=("Other DL (Bytes)", "sum"),
    other_upload_data=("Other UL (Bytes)", "sum")
).reset_index()

# Exploratory Data Analysis - Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

# Missing Values and Outliers
    st.subheader("Missing Values and Outliers")
    st.write(data.isnull().sum())

# Univariate Analysis - Histogram of Session Duration
    st.subheader("Univariate Analysis")
    st.write("Histogram of Session Duration")
    st.bar_chart(data['Dur. (ms)'])

# Boxplot of Download Data using Seaborn
    st.subheader("Boxplot of Download Data")
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x='Total DL (Bytes)', ax=ax)
    st.pyplot(fig)

# Bivariate Analysis - Scatter Plot of Download vs Upload Data
    st.subheader("Bivariate Analysis")
    st.write("Scatter Plot of Download vs Upload Data")
    fig, ax = plt.subplots()
    ax.scatter(data['Total DL (Bytes)'], data['Total UL (Bytes)'])
    ax.set_xlabel('Download Data')
    ax.set_ylabel('Upload Data')
    st.pyplot(fig)

# Correlation Matrix
    st.subheader("Correlation Matrix")
    correlation_matrix = data[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']].corr()
    st.write(correlation_matrix)

# Dimensionality Reduction (PCA)
    st.subheader("Dimensionality Reduction (PCA)")
    pca = PCA(n_components=2)
    pca_data = data[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']].dropna()

# Standardizing data (if needed)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data)

# Fit PCA
    pca_result = pca.fit_transform(scaled_data)

# Display PCA Results
    st.write("PCA Result")
    st.write(pca_result)
   
   