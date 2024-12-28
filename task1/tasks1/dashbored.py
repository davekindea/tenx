import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as pe

# Set up Streamlit page
st.set_page_config(page_title="Solar Data", page_icon=":bar_chart:", layout="wide")

# Load data
data1 = pd.read_csv("./data/benin-malanville.csv")
data2 = pd.read_csv("./data/sierraleone-bumbuna.csv")
data3 = pd.read_csv("./data/togo-dapaong_qc.csv")

# Combine datasets into a dictionary
datasets = {
    "Benin": data1,
    "Sierra Leone": data2,
    "Togo": data3
}

st.title("Solar Data Analysis")

for name, data in datasets.items():
    st.header(f"{name} Dataset")
    
    # Convert the 'Timestamp' column to datetime format
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
    
    # Extract the date part from the timestamp
    data['Date'] = data['Timestamp'].dt.date
    
    # Display Raw Data
    st.subheader("Raw Data")
    st.dataframe(data)
    
    # Dataset Information
    st.subheader("Dataset Head")
    st.write(data.head())
    
    st.subheader("Dataset Dimensions")
    num_rows, num_cols = data.shape
    st.write(f"**Number of Rows:** {num_rows}")
    st.write(f"**Number of Columns:** {num_cols}")
    
    st.subheader("Dataset Description")
    st.write(data.describe())
    
    st.subheader("Missing Values in Dataset")
    st.write(data.isnull().sum())
    
    # Box plot to visualize outliers
    st.subheader("Box Plot of 'Global Horizontal Irradiance'")
    if 'GHI' in data.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x=data['GHI'], ax=ax)
        st.pyplot(fig)
    else:
        st.write("**'GHI' column not available for box plot.**")
    
    # Time Series Plot
    if 'Timestamp' in data.columns and 'GHI' in data.columns:
        st.subheader("GHI Time Series Plot")
        fig = pe.line(
            data, 
            x='Timestamp', 
            y='GHI', 
            title=f'{name} - GHI Time Series',
            labels={'Timestamp': 'Time', 'GHI': 'GHI (W/m²)'}
        )
        st.plotly_chart(fig)
    else:
        st.write("**Timestamp or GHI column not available for time series plot.**")
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    data_numeric = data.select_dtypes(include=['float64', 'int64']).fillna(0)
    if not data_numeric.empty:
        corr = data_numeric.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
        st.pyplot(plt)
    else:
        st.write("**No numeric data available for heatmap.**")
    
    # Polar Plot
    st.subheader("Wind Polar Plot")
    if 'wind_speed' in data.columns and 'wind_direction' in data.columns:
        fig = pe.scatter_polar(
            data,
            r="wind_speed",
            theta="wind_direction",
            color="wind_speed",
            size="wind_speed",
            color_continuous_scale=pe.colors.sequential.Viridis,
            title=f"{name} - Wind Polar Plot"
        )
        st.plotly_chart(fig)
    else:
        st.write("**Wind speed or wind direction data not available for polar plot.**")
    
    st.write("---")

st.write("### End of Solar Data Analysis")
