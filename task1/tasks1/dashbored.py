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

# Sidebar for dataset selection
st.sidebar.header("Dataset Selection")
dataset_option = st.sidebar.selectbox(
    "Select Dataset:",
    ["Benin", "Sierra Leone", "Togo"]
)

# Select dataset based on user input
if dataset_option == "Benin":
    data = data1
elif dataset_option == "Sierra Leone":
    data = data2
elif dataset_option == "Togo":
    data = data3

# Convert the 'Timestamp' column to datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Extract the date part from the timestamp
data['Date'] = data['Timestamp'].dt.date

# Display the data
st.title("Solar Data Analysis")
st.header("Raw Data")
st.dataframe(data)

# Sidebar for date filtering
st.sidebar.header("Date Filtering")
dates = st.sidebar.multiselect(
    "Select Date:", 
    options=sorted(data['Date'].unique()),  # Sort the dates for better UX
    default=sorted(data['Date'].unique())
)

# Filter data based on selected dates
if dates:
    data_selection = data[data['Date'].isin(dates)]
else:
    data_selection = data

# Display the filtered data
st.header("Filtered Data")
st.dataframe(data_selection)

# Dataset Information
st.subheader("Dataset Head")
st.write(data.head(10))

st.subheader("Dataset Dimensions")
num_rows, num_cols = data.shape
st.write(f"**Number of Rows:** {num_rows}")
st.write(f"**Number of Columns:** {num_cols}")

st.subheader("Dataset Description")
st.write(data.describe())

st.subheader("Missing Values in Dataset")
st.write(data.isnull().sum())

# Box plot to visualize outliers
st.header("Outlier Visualization")
st.subheader("Box Plot of 'Global Horizontal Irradiance':")
fig, ax = plt.subplots()
sns.boxplot(x=data_selection['GHI'], ax=ax)
st.pyplot(fig)

# Time Series Plot
st.subheader("GHI Time Series Plot")
fig = pe.line(
    data_selection, 
    x='Timestamp', 
    y='GHI', 
    title='GHI Time Series',
    labels={
        'Timestamp': 'Time',
        'GHI': 'GHI (W/m²)'
    }
)
st.plotly_chart(fig)

# Numeric data for heatmap
st.subheader("Correlation Heatmap")
data_numeric = data.select_dtypes(include=['float64', 'int64']).fillna(0)
if not data_numeric.empty:
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_numeric, annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(plt)
else:
    st.write("**No numeric data available for heatmap**")

# Polar Plot (Ensure these columns exist in the dataset)
st.subheader("Wind Polar Plot")
if 'wind_speed' in data.columns and 'wind_direction' in data.columns:
    fig = pe.scatter_polar(
        data,
        r="wind_speed",
        theta="wind_direction",
        color="wind_speed",
        size="wind_speed",
        color_continuous_scale=pe.colors.sequential.Viridis,
        title="Wind Polar Plot"
    )
    st.plotly_chart(fig)
else:
    st.write("**Wind speed or wind direction data not available for polar plot**")

# Histogram
st.subheader("GHI Histogram")
plt.figure(figsize=(10, 6))
sns.histplot(data['GHI'], bins=30, kde=True)
plt.title('Global Horizontal Irradiance (GHI) Histogram')
plt.xlabel('GHI')
plt.ylabel('Frequency')
st.pyplot(plt)
