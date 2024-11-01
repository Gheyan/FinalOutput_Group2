#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
import os
import io
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

#######################
# Page configuration
st.set_page_config(
    page_title="Dashboard Template", # Replace this with your Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Dashboard Template')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Elon Musk\n2. Jeff Bezos\n3. Sam Altman\n4. Mark Zuckerberg")

#######################
# Data

# Load data
dataset = pd.read_csv("data/vgsales.csv")

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("Video Game Sales Dataset")
    st.write("")

    st.markdown("""

    The **Video Game Sales** dataset was formulated by Gregory Smith in 2016, it is generally focused on presenting varying data that encompasses multiple sales figures of various video game titles, with the video games included having a minimum cap of atleast generating 100,000 sales of individual copies.

    In general, the dataset contains 11 columns that have their own distinct attributes, but these columns can be seperated into two primary categories which are the descriptive and numerical columns. Descriptive columns involve data such as ranking, name, platform, year genre, and publisher. The other column comprising of numerical data consist of data such as the different sales figures for regions such as North America, Europe, Japan, Other coupled countries, and a Global collection.

    **Content**      
    The dataset as a whole contains 16,598 rows, as originally 2 additional rows were present but the creator of the dataset itself has removed these rows due to too many incomplete information being present within them. There are varying columns that help to seperate the rows distinctinctly which are mainly divided into two categories which are mentioned above. In addition the sales figures of the dataset are in millions in that (1 is equivalent to 1,000,000 sales).

    `Link:` https://www.kaggle.com/datasets/gregorut/videogamesales/data           
                
    """)

    # Your content for your DATASET page goes here
    
    # Display the dataset
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(dataset, use_container_width=True, hide_index=True)

    # Describe Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(dataset.describe(), use_container_width=True)

    st.markdown(""" 
    The results from `df.describe()` mainly shows the descriptive statistics about the dataset. The main highlights are the key trends in game releases and regional sales patterns:

    **Release Year**: Games in this dataset average a release year of 2006, with a standard deviation of 5.83, spanning from 1980 to 2020. The concentration around the 2000s and 2010s (as shown by the 25th, 50th, and 75th percentiles: 2003, 2007, and 2010) reflects an emphasis on more recent titles.

    Sales by Region:
    - **NA_Sales**: The North American market averages 0.26 million units per game, with high variability (standard deviation of 0.82) and a maximum of 41.49 million units, underscoring a strong demand for blockbuster games.
    - **EU_Sales**: European sales average 0.15 million units with a standard deviation of 0.51, typically lower than in North America but still showing steady performance.
    - **JP_Sales**: Japan shows an average of 0.08 million units (standard deviation of 0.31), indicating it as a smaller, more niche market.
    - **Other_Sales**: Other regions average 0.08 million units with a standard deviation of 0.19. However, a few games reach up to 10.22 million units in these markets.

    **Global Sales**: Aggregating all regions, the average global sales per game reach 0.54 million units, with a high standard deviation of 1.56. The top-selling game hit 82.74 million units, reflecting a skewed distribution where only a few titles achieve high global sales.
                
    The 25th, 50th, and 75th percentiles across all sales regions illustrate a skewed distribution, with the majority of games falling into lower sales brackets and only a few achieving high sales. This distribution suggests the dataset’s usefulness for studying sales patterns and factors that contribute to high-performing games across different markets and release periods. summarize
    """)



    # Display DataFrame information
    buffer = io.StringIO()
    dataset.info(buf=buffer)
    info_output = buffer.getvalue()
    st.header("Dataset Information")
    st.text(info_output)

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((1.5, 4.5, 2), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('#### Graphs Column 1')


    with col[1]:
        st.markdown('#### Graphs Column 2')
        
    with col[2]:
        st.markdown('#### Graphs Column 3')

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here