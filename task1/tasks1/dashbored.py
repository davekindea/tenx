import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as pe
from scipy.stats import zscore
import numpy as np
import os

# Build the file path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "./data/benin-malanville.csv")
data_path_data2=os.path.join(current_dir, "./data/sierraleone-bumbuna.csv")
data_path_data3=os.path.join(current_dir, "./data/togo-dapaong_qc.csv")
# Set up Streamlit page
st.set_page_config(page_title="Solar Data", page_icon=":bar_chart:", layout="wide")

# Load data
data1 = pd.read_csv(data_path)
data2 = pd.read_csv(data_path_data2)
data3 = pd.read_csv(data_path_data3)

# Combine datasets into a dictionary
datasets = {
    "Benin": data1,
    "Sierra Leone": data2,
    "Togo": data3
}

# Sidebar for dataset selection
st.sidebar.header("Dataset Selection")
dataset_option = st.sidebar.selectbox(
    "Select Dataset:",
    ["Benin", "Sierra Leone", "Togo"]
)

# Load selected dataset
data = datasets[dataset_option]

# Page Title
st.title(f"{dataset_option} Dataset Analysis")

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




# Dynamic Selection for Year, Month, and Hourly Data Visualization
st.subheader("Analyze Yearly, Monthly, and Hourly Patterns of GHI, DNI, DHI, and Tamb")
data["Date"] = pd.to_datetime(data["Date"], errors='coerce')
data.set_index("Date",inplace=True)
data["year"]=data.index.year
data["month"]=data.index.month
data["day"]=data.index.day
data["Hour"]=data.index.hour
data["week"]=data.index.weekday
if not pd.api.types.is_datetime64_any_dtype(data.index):
    data.index = pd.to_datetime(data.index)

# User selects the year
available_years = data.index.year.unique()
selected_year = st.selectbox('Select Year:', available_years)

# Filter data by selected year
yearly_data = data.copy()

# User selects time aggregation level
aggregation_level = st.radio("Select Aggregation Level:", ['Yearly', 'Monthly', 'Hourly'])

# Aggregation based on user selection
if aggregation_level == 'Yearly':
    yearly_avg = yearly_data[['GHI', 'DNI', 'DHI', 'Tamb']].resample('Y').mean()
    st.write("### Yearly Average Patterns")
    fig, ax = plt.subplots(figsize=(15, 6))
    yearly_avg.plot(ax=ax)
    ax.set_title(f'Yearly Average Patterns of GHI, DNI, DHI, and Tamb ({selected_year})')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Value')
    st.pyplot(fig)

elif aggregation_level == 'Monthly':
    monthly_avg = yearly_data[['GHI', 'DNI', 'DHI', 'Tamb']].resample('M').mean()
    st.write("### Monthly Average Patterns")
    fig, ax = plt.subplots(figsize=(15, 6))
    monthly_avg.plot(ax=ax)
    ax.set_title(f'Monthly Average Patterns of GHI, DNI, DHI, and Tamb ({selected_year})')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Value')
    st.pyplot(fig)

elif aggregation_level == 'Hourly':
    hourly_avg = yearly_data[['GHI', 'DNI', 'DHI', 'Tamb']].groupby(yearly_data.index.hour).mean()
    st.write("### Hourly Average Patterns")
    fig, ax = plt.subplots(figsize=(15, 6))
    hourly_avg.plot(ax=ax)
    ax.set_title(f'Hourly Average Patterns of GHI, DNI, DHI, and Tamb ({selected_year})')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Average Value')
    st.pyplot(fig)

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
        title=f'{dataset_option} - GHI Time Series',
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
    plt.figure(figsize=(15, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    st.pyplot(plt)
else:
    st.write("**No numeric data available for heatmap.**")

# Polar Plot
st.subheader("Wind Polar Plot")
if 'WS' in data.columns and 'WD' in data.columns:
    fig = pe.scatter_polar(
        data,
        r="WS",
        theta="WD",
        color="WS",
        size="WS",
        color_continuous_scale=pe.colors.sequential.Viridis,
        title=f"{dataset_option} - Wind Polar Plot"
    )
    st.plotly_chart(fig)
else:
    st.write("**Wind speed or wind direction data not available for polar plot.**")
# Bubble Chart Visualization

st.subheader("Bubble Chart: GHI vs Temperature (Tamb) with RH and WS")
if all(col in data.columns for col in ['GHI', 'Tamb', 'RH', 'WS']):
    fig, ax = plt.subplots(figsize=(12, 8))
    bubble_chart = ax.scatter(
        data['GHI'], 
        data['Tamb'], 
        s=data['RH'] * 10,  # Bubble size based on Relative Humidity (RH)
        c=data['WS'],       # Color based on Wind Speed (WS)
        cmap='plasma', 
        alpha=0.7,
        edgecolors='w'
    )
    cbar = plt.colorbar(bubble_chart, ax=ax)
    cbar.set_label('Wind Speed (WS)')
    
    # Titles and labels
    ax.set_title('Bubble Chart: GHI vs Temperature (Tamb) with RH and WS')
    ax.set_xlabel('Global Horizontal Irradiance (GHI)')
    ax.set_ylabel('Temperature (°C)')
    
    st.pyplot(fig)
else:
    st.write("**Required columns ('GHI', 'Tamb', 'RH', 'WS') are not available for the bubble chart.**")


# List of columns to analyze for outliers
columns_to_analyze = ['GHI', 'DNI', 'DHI', 'WS', 'Tamb']
threshold = 3

# Check if the columns exist in the dataset
for col in columns_to_analyze:
    if col in data.columns:
        # Calculate Z-Score and Add Outlier Column
        data[f'{col}_zscore'] = zscore(data[col].fillna(data2[col].mean()))
        data[f'{col}_outlier'] = data[f'{col}_zscore'].abs() > threshold
    else:
        st.write(f"**Column '{col}' not found in the dataset. Skipping...**")
# Select Column for Outlier Visualization
selected_column = st.selectbox(
    "Select a column to visualize outliers:",
    columns_to_analyze
)

if selected_column in data.columns and f'{selected_column}_outlier' in data.columns:
    st.subheader(f"Outlier Detection in {selected_column} Using Z-Score")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        x=data.index, 
        y=data[selected_column], 
        hue=data[f'{selected_column}_outlier'], 
        palette={True: 'red', False: 'blue'}, 
        ax=ax
    )
    ax.set_title(f'Outlier Detection in {selected_column} Using Z-Score')
    ax.set_xlabel('Index')
    ax.set_ylabel(selected_column)
    ax.legend(title='Outlier')
    st.pyplot(fig)
else:
    st.write(f"**Outlier data for '{selected_column}' is not available for visualization.**")



# Distribution Plots for Selected Variables
st.subheader("Distribution of Key Variables")

variables = ['GHI', 'DNI', 'DHI', 'WS', 'Tamb']

# Check if the selected variables exist in the dataset
available_variables = [var for var in variables if var in data.columns]

if available_variables:
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, var in enumerate(available_variables):
        sns.histplot(data3[var], bins=30, kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribution of {var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Frequency')
    
    # Remove empty subplot if variables are less than subplot slots
    if len(available_variables) % 2 != 0:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.write("**None of the selected variables are available in the dataset.**")



# Wind Speed and Direction Polar Plot
st.subheader("Wind Speed and Direction Polar Plot")

if 'WD' in data.columns and 'WS' in data.columns:
   
    
    # Convert wind direction to radians
    data["WD_rad"] = np.deg2rad(data["WD"])
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    sc = ax.scatter(
        data["WD_rad"], 
        data["WS"], 
        c=data["WS"], 
        cmap='viridis', 
        alpha=0.75, 
        edgecolors='k'
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Wind Speed (m/s)')
    ax.set_theta_zero_location('N')  
    ax.set_theta_direction(-1)       
    ax.set_title('Wind Speed and Direction Polar Plot')
    
    st.pyplot(fig)
else:
    st.write("**Wind Direction (WD) or Wind Speed (WS) column not available for polar plot.**")

# Moving Average for ModA and ModB

data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
st.subheader("Moving Average for ModA and ModB Before and After Cleaning")



# Check if ModA and ModB exist in the dataset
if 'ModA' in data.columns and 'ModB' in data.columns:
    # Create two sample datasets (Before and After Cleaning)
    before_cleaning = data[data['Cleaning'] == 0]
    after_cleaning = data[data['Cleaning'] == 1]
    
    # Compute moving average (7-day window)
    before_cleaning['ModA_MA'] = before_cleaning['ModA'].rolling(window=7).mean()
    after_cleaning['ModA_MA'] = after_cleaning['ModA'].rolling(window=7).mean()
    
    before_cleaning['ModB_MA'] = before_cleaning['ModB'].rolling(window=7).mean()
    after_cleaning['ModB_MA'] = after_cleaning['ModB'].rolling(window=7).mean()
    
    # Plot ModA Moving Average
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    
    ax[0].plot(before_cleaning.index, before_cleaning['ModA_MA'], label='Before Cleaning', color='blue')
    ax[0].plot(after_cleaning.index, after_cleaning['ModA_MA'], label='After Cleaning', color='green')
    ax[0].set_title('ModA Sensor Readings with Moving Average Before and After Cleaning')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('ModA (Moving Average)')
    ax[0].legend()
    
    # Plot ModB Moving Average
    ax[1].plot(before_cleaning.index, before_cleaning['ModB_MA'], label='Before Cleaning', color='blue')
    ax[1].plot(after_cleaning.index, after_cleaning['ModB_MA'], label='After Cleaning', color='green')
    ax[1].set_title('ModB Sensor Readings with Moving Average Before and After Cleaning')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('ModB (Moving Average)')
    ax[1].legend()
    
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.write("**'ModA' or 'ModB' columns not found in the dataset.**")

# Time Series Plot
fig, ax = plt.subplots(figsize=(15, 6))
for col in ["GHI", "DNI", "DHI", "Tamb"]:
    ax.plot(data.index, data[col], label=col)
ax.set_title('Time Series Analysis of GHI, DNI, DHI, and Tamb')
ax.set_xlabel('Time')
ax.set_ylabel('Values')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

st.write("---")
st.write("### End of Solar Data Analysis")
