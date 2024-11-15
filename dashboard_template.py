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
    st.title('Video Game Sales')

    # Page Button Navigation
    st.subheader("Pages: ")

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
    st.subheader("Members: ")
    st.markdown("1. Aleah Balagao\n2. Sean Cabantog\n3. John Garina \n4. Felipe Panugan III\n5. Gian Mateo")

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of a training three models using the Video Game Sales dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/gregorut/videogamesales/data)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1PinmgyIyVgvNG0V0cNRMxhJbwuC02iPe?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Gheyan/FinalOutput_Group2.git)")
    

    

#######################
# Data

# Load data
dataset = pd.read_csv("data/vgsales.csv")

#######################

# Plots

def Platform_Distribution():
    platform_counts = dataset['Platform'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(platform_counts.index, platform_counts.values, color='skyblue')
    plt.xlabel('Platform')
    plt.ylabel('Games Published')
    plt.title('Platform Distribution')
    plt.xticks(rotation=90)
    plt.tight_layout()

    st.pyplot(plt)
    plt.clf()

def Publisher_Distribution():
    Publisher_counts = dataset['Publisher'].value_counts(dropna=True).head(10)
    plt.figure(figsize=(10, 6))
    plt.bar(Publisher_counts.index, Publisher_counts.values, color='skyblue')
    plt.xlabel('Publisher')
    plt.ylabel('Games Published')
    plt.title('Top 10 Publishers Distribution')
    plt.xticks(rotation=90)
    plt.tight_layout()
    

    st.pyplot(plt)
    plt.clf()

def Genre_Distribution():
    genre_counts = dataset['Genre'].value_counts()
    plt.figure(figsize=(10, 8))
    plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Distribution of Video Game Genres')
    plt.axis('equal')
    
    st.pyplot(plt)
    plt.clf()

def Region_Distribution():
  total_na_sales = dataset['NA_Sales'].sum()
  total_eu_sales = dataset['EU_Sales'].sum()
  total_jp_sales = dataset['JP_Sales'].sum()
  total_other_sales = dataset['Other_Sales'].sum()
  sales_totals = [total_na_sales, total_eu_sales, total_jp_sales, total_other_sales]
  sales_labels = ['NA Sales', 'EU Sales', 'JP Sales', 'Other Sales']
  plt.figure(figsize=(10, 8))
  plt.pie(sales_totals, labels=sales_labels, autopct='%1.1f%%', startangle=90)
  plt.axis('equal')
  plt.title('Global Sales Distribution by Region')
  
  st.pyplot(plt)
  plt.clf()
#Felipe

def na_sales_distribution():
    genre_df = dataset.groupby('Genre')['NA_Sales'].sum()  # Sum NA_Sales by genre
    colors = plt.cm.tab20.colors[:len(genre_df)]
    
    plt.figure(figsize=(8, 6))
    plt.pie(
        genre_df,
        labels=genre_df.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.85,
        labeldistance=1.1
    )
    plt.title('NA Sales Distribution by Genre')
    st.pyplot(plt)
    plt.clf()

def eu_sales_distribution():
    genre_df = dataset.groupby('Genre')['EU_Sales'].sum()  # Sum EU_Sales by genre
    colors = plt.cm.tab20.colors[:len(genre_df)]
    
    plt.figure(figsize=(8, 6))
    plt.pie(
        genre_df,
        labels=genre_df.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.85,
        labeldistance=1.1
    )
    plt.title('EU Sales Distribution by Genre')
    st.pyplot(plt)
    plt.clf()

def jp_sales_distribution():
    genre_df = dataset.groupby('Genre')['JP_Sales'].sum()  # Sum JP_Sales by genre
    colors = plt.cm.tab20.colors[:len(genre_df)]
    
    plt.figure(figsize=(8, 6))
    plt.pie(
        genre_df,
        labels=genre_df.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.85,
        labeldistance=1.1
    )
    plt.title('JP Sales Distribution by Genre')
    st.pyplot(plt)
    plt.clf()

def other_sales_distribution():
    genre_df = dataset.groupby('Genre')['Other_Sales'].sum()  # Sum Other_Sales by genre
    colors = plt.cm.tab20.colors[:len(genre_df)]
    
    plt.figure(figsize=(8, 6))
    plt.pie(
        genre_df,
        labels=genre_df.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.85,
        labeldistance=1.1
    )
    plt.title('Other Region Sales Distribution by Genre')
    st.pyplot(plt)
    plt.clf()



def bar_graph1():
    dataset['Global_Sales'] = pd.to_numeric(dataset['Global_Sales'], errors='coerce')
    df = dataset.dropna(subset=['Global_Sales'])
    plt.figure(figsize=(12, 6))
    platform_sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)
    platform_sales.plot(kind='bar', color='purple')
    plt.title('Total Global Sales by Platform')
    plt.xlabel('Platform')
    plt.ylabel('Global Sales (in Millions)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()  # Clear figure after displaying

def bar_graph2():
    dataset['Global_Sales'] = pd.to_numeric(dataset['Global_Sales'], errors='coerce')
    df = dataset.dropna(subset=['Global_Sales'])
    plt.figure(figsize=(12, 6))
    genre_sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
    genre_sales.plot(kind='bar', color='pink')
    plt.title('Total Global Sales by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Global Sales (in Millions)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()  # Clear figure after displaying

def heat_map():
  dataset.dropna(subset=['Genre', 'Platform', 'Global_Sales'], inplace=True)
  top_platforms = dataset['Platform'].value_counts().nlargest(10).index.tolist()
  filtered_df = dataset[dataset['Platform'].isin(top_platforms)]
  average_sales = filtered_df.groupby(['Genre', 'Platform'])['Global_Sales'].mean().reset_index()
  pivot_table = average_sales.pivot(index='Genre', columns='Platform', values='Global_Sales')
  plt.figure(figsize=(12, 10))
  sns.heatmap(pivot_table, annot=True, cmap='YlGnBu')
  plt.title('Average Global Sales by Genre in the Top 10 Platforms(1 = 1million)')
  plt.xlabel('Platform')
  plt.ylabel('Genre')
  plt.show()
  st.pyplot(plt)
  plt.clf()

def bar_chart():
  top_platforms = dataset.groupby('Platform').sum()[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum(axis=1).nlargest(10).index
  sales_by_region_platform = dataset[dataset['Platform'].isin(top_platforms)].groupby('Platform').sum()[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
  sales_by_region_platform.plot(kind='bar', stacked=True, figsize=(10, 6))
  plt.title('Sales by Region and Platform (Top 10 Platforms)')
  plt.xlabel('Platform')
  plt.ylabel('Sales (in millions)')
  plt.xticks(rotation=90)
  plt.show()
  st.pyplot(plt)
  plt.clf()
#Felipe
######
#cabantog
mydf = dataset.dropna(subset=['Year'])
mydf['Year'] = mydf['Year'].astype(int)

def gamegenretime():
  genredata = {'Year': mydf['Year'], 'Genre': mydf['Genre'], 'Global Sales': mydf['Global_Sales']}
  genre_df = pd.DataFrame(genredata)

  grouped_data = genre_df.groupby(['Year', 'Genre']).sum().reset_index()

  pivot_data = grouped_data.pivot(index='Year', columns='Genre', values='Global Sales').fillna(0)

  num_genres = len(pivot_data.columns)
  cmap = plt.get_cmap('tab20', num_genres)
  colors = [cmap(i) for i in range(num_genres)]

  plt.figure(figsize=(15, 8))
  for genre, color in zip(pivot_data.columns, colors):
      plt.plot(pivot_data.index, pivot_data[genre], marker='o', label=genre, color=color)

  plt.title('Game Genre Sales Over Time')
  plt.xlabel('Year')
  plt.ylabel('Global Sales (in Millions)')
  plt.grid()
  plt.legend(title='Genre')
  plt.xticks(pivot_data.index, rotation=75)
  plt.tight_layout()
  plt.show()
  st.pyplot(plt)
  plt.clf()
######

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here
    st.markdown(""" 

    A Streamlit web application that performs **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, as well as **Unsupervised and Supervised Machine Learning** to .

    #### Pages
    1. `Dataset` - Provides a brief description of the Video Game Sales data set that would mainly be used within the whole dashboard.
    2. `EDA` - Exploratory Data Analysis of the Video Game Sales dataset. This portion of the dashboard mainly highlights the different relationships between the various columns present within the dataset, providing multiple insights by the use of various graphs as visual representations of the said relationships.
    3. `Data Cleaning / Pre-processing` - Various data cleaning and pre-processing were dont as to properly prepare the data before use. Techniques such as null checking and handling, and label encoding were the main things that were used within the dataset.
    4. `Machine Learning` - Two models are currently present within the dashboard, which are a supervised model (linear regression) and unuspervised model (k-means clusering).
    5. `Prediction` - Within this page, users can input specific values which can map into predictions of global sales for video games.
    6. `Conclusion` - Summary of the all the insights gained and observations from the EDA and model training.


    """)

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
    st.subheader("Dataset Information")
    st.text(info_output)


    st.markdown("""
    The information presented directly above shows the different columns that are present within the dataset, while also highlighting the type of data that each of them contains. The data types present within the whole data set are floats, integers, and objects (considered as strings).
    
    """)

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    # Your content for the EDA page goes here
   

    col = st.columns((1,1), gap='medium')


    with col[0]:
        st.markdown('#### Platform Distribution Chart')
        Platform_Distribution();
        st.markdown('Within the chart above, it shows the different types of platforms that are present within the dataset, in which a total of 31 distinct platforms where identified, and within that distinction the platform with the most games attributed to it is the Nintendo DS with 2163 ganes published.')

        st.markdown('#### Genre Distribution Chart')
        Genre_Distribution();
        st.markdown('Within the contents of the genre distribution chart 12 distinct genres where identified, these genres describe and categorize the multiple games present within the dataset. In which the most common genre present within the dataset is Action, with 20% of the data set being under this genre.')
        #Felipe
        st.markdown('### North American Sales Distribution Chart')
        na_sales_distribution();
        st.markdown("""Action is the best-selling genre in NA (North America). The Sports genre is the second-best selling genre in the region by a relatively small margin, followed closely by Shooter games.
""")

        st.markdown('### Europe Sales Distribution Chart')
        eu_sales_distribution();
        st.markdown("""Action is the best-selling genre in EU (Europe). The Sports genre is the second-best selling genre in the region by a relatively small margin, followed closely by Shooter games.""")

        # BALAGAO - EDA first graphs
        st.markdown('### Global Sales Platform Chart')
        bar_graph1();
        st.markdown("""
                PS2 stands as the highest-selling platform, followed by the X360 in second place and the PS3 in third. This ranking highlights a clear preference among consumers for these particular platforms, suggesting they hold the largest market share in terms of sales compared to other gaming platforms.""")
        
        st.markdown('### Platform Sales by Region Chart')
        bar_chart();
        st.markdown("""
                In Japan, gaming sales data indicate a strong preference for portable gaming, with a significant portion of sales coming from the Nintendo DS, a popular handheld device. In contrast, North American gamers tend to prefer more powerful console systems, as reflected in the top-selling platforms, which include the PlayStation 2 (PS2) and Xbox 360 (X360). European consumers also show a particular affinity for the PlayStation series, with the PS2 and PS3 leading in sales in this region. Similarly, in markets outside of Europe, Japan, and North America, the PS2 stands out as the most favored gaming platform, with sales surpassing those of other gaming systems in these areas.""")

        ##cabantog
        st.markdown('### Game Genre Sales Over Time Chart')
        gamegenretime();
        st.markdown("""
        In the 1980s, the Platform genre has garnered the most sales, peaking in the year 1985 (which can be attributed to the game Super Mario Bros.). Shooter games sold well on the preceding year 1984, while Puzzle games sold well towards the end of the decade in 1989.
        While not selling as much until the mid-1990s, Action and Sports games have generated by far the most sales in history, seeing massive upward trajectories in the turn of the new millenium, dwarfing most of the other genres in sales.
        """)
        ##
    

    with col[1]:
        st.markdown('#### Publisher Distribution Chart')
        Publisher_Distribution();
        st.markdown('The chart above present the top 10 publishers within the dataset, sorted primarily by the most games published. In total we have identified 578 distinct publishers within the dataset, and ranking top above all of these is the publisher Electronic Arts with 1351 games published.')

        st.markdown('#### Region Distribution Chart')
        Region_Distribution();
        st.markdown('Presented above is the region distribution chart, in which it outlines the division of all global sales into specific regions. Within all the regions present the North America regions by far has the most marketshare with it encompassing 49.3% of all global sales.')

        #Felipe
        st.markdown('### Japan Sales Distribution Chart')
        jp_sales_distribution();
        st.markdown("""Role-PLaying is the best-selling genre in JP (Japan), outselling Action and Sports games by a wide margin. Shooter games also sell the least in this region.""")

        st.markdown('### Other Sales Distribution Chart')
        other_sales_distribution();
        st.markdown("""Action is the best-selling genre in other regions. The Shooter genre is the second-best selling genre in the region by a relatively small margin, followed closely by Shooter games.""")

        st.markdown('### Global Sales Genre Chart')
        bar_graph2();
        st.markdown("""Action games are the top-selling category, with sports games and shooter games sharing the second spot in terms of popularity. it suggests that action, sports, and shooter games are consistently favored by consumers, making them more likely to achieve higher sales compared to other genres. """)
        
        st.markdown('### Average Global Sales by Genre Chart')
        heat_map();
        st.markdown("""Certain combinations of platform and genre have a significant impact on game sales. The most popular combinations show that platform-based games on the Wii platform lead in average sales, reaching approximately 1.6 million units. Following closely, shooter games on the X360 platform achieve an average of 1.4 million units sold, while shooter games on the PS3 platform come in third with an average of 1.3 million units. On the other end of the spectrum, certain genre combinations on the PC platform report the lowest average sales. Puzzle games on the PC platform sell the least, averaging only 37,000 units, while fighting and platform genres on PC are similarly low, each averaging around 45,000 units. This data highlights a significant disparity in sales performance across different platform-genre combinations, with PC games generally underperforming compared to other platforms.""")
        #Felipe

    
    

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here


    #Data Cleaning and Pre-processing for Genre & Global Sales (K-Means Clustering Model)
    st.subheader('Genre & Global Sales (K-Means Clustering Model)')
    data = {'Genre': dataset['Genre'],'Global_Sales': dataset['Global_Sales']}
    df_data_genreAndGlobal = pd.DataFrame(data)
    st.dataframe(df_data_genreAndGlobal, use_container_width=True,hide_index=True)
    st.markdown('We have seperated two distinct columns from the main dataset that will be used specifically for training a K-means clustering model to determine clusters suited for Genre and Global Sales.')

    # Check for null values in 'Genre' and 'Global_Sales' columns
    null_counts = df_data_genreAndGlobal.isnull().sum()
    if null_counts.any():
        st.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
    else:
        st.success("No null values found in the selected columns.")

    st.markdown('Using the `.isnull().sum()` in our specific chunk of the dataset we will check if there are any null values that needs to be processed. Since no null values are found no extra processing in terms of removing or reprocessing the missing values.')

    label_encoder = LabelEncoder()
    df_data_genreAndGlobal['Encoded_Genre'] = label_encoder.fit_transform(df_data_genreAndGlobal['Genre'])

    display_data = df_data_genreAndGlobal[['Genre', 'Encoded_Genre']]
    st.subheader('Genre and Converted Label Table')
    st.dataframe(display_data, use_container_width=True)

    st.markdown('Now we converted the values of genre column to numerical values using `LabelEncoder`. The Encoded_Genre column can now be used and processed by our unsupervised training model.')

    distinct_labels = label_encoder.classes_  
    encoded_labels = range(len(distinct_labels)) 

    # Create a DataFrame for display
    label_mapping = pd.DataFrame({
        'Genre': distinct_labels,
        'Encoded_Genre': encoded_labels
        
    })

    # Display the distinct labels and their corresponding genres
    st.subheader('Distinct Encoded Labels and Corresponding Genres')
    st.dataframe(label_mapping, use_container_width=True, hide_index=True)
    st.markdown('This table above shows the different distinct genres that are present within the dataset and their corresponding encoded values after being processed.')

    
    st.divider()

    #Data Cleaning and Pre-processing for Regional Sales and Global Sales (Linear Regression Model)
    st.subheader('Regional Sales and Global Sales (Linear Regression Model)')
    LinearRegressiondata = {
    'NA_Sales': dataset['NA_Sales'],
    'EU_Sales': dataset['EU_Sales'],
    'JP_Sales': dataset['JP_Sales'],
    'Other_Sales': dataset['Other_Sales'],
    'Global_Sales': dataset['Global_Sales']
    }

    df_data_Linear = pd.DataFrame(LinearRegressiondata)
    st.dataframe(df_data_Linear, use_container_width=True,hide_index=True)
    st.markdown('We have seperated five distinct columns from the main dataset that will be used specifically for training a Linear Regression model to determine the relationship of the different regions and Global Sales.In addition to the information present all values are in the millions in that (1 is equivalent to 1,000,000 sales)')

    # Check for null values in 'Genre' and 'Global_Sales' columns
    null_counts = df_data_Linear.isnull().sum()
    if null_counts.any():
        st.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
    else:
        st.success("No null values found in the selected columns.")

    st.markdown('Using the `.isnull().sum()` in our specific chunk of the dataset we will check if there are any null values that needs to be processed. Since no null values are found no extra processing in terms of removing or reprocessing the missing values.')
    

    st.subheader('Splitting the Data into X (independent variable), and y (Dependent variable)')
    st.code("""
    regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

    plt.figure(figsize=(12, 10))
    for i, region in enumerate(regions):

        X = sales_data[[region]]  # Independent variable (region)
        y = sales_data['Global_Sales']  # Dependent variable (Global_Sales)

    """)

    X_Linear = df_data_Linear[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]  # Independent variables (regional sales)
    y_Linear= df_data_Linear['Global_Sales']

    st.subheader('X (independent variable)')
    st.dataframe(X_Linear, use_container_width=True,hide_index=True)
    st.subheader('y (dependent variable)')
    st.dataframe(y_Linear, use_container_width=True,hide_index=True)

    # BALAGAO - Data Cleaning for Linear Regression Model
    st.subheader('Actual vs Predicted Global Sales (Linear Regression Model)')

    # Data preparation
    X = dataset[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]  # Independent variables
    y = dataset['Global_Sales']  # Dependent variable

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training 
    # Prediction
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Results dataframe
    results_df = pd.DataFrame({
        'Actual Global Sales': y_test, 
        'Predicted Global Sales': y_pred  
    })


    LinearRegData = {
        'NA_Sales': dataset['NA_Sales'],
        'EU_Sales': dataset['EU_Sales'],
        'JP_Sales': dataset['JP_Sales'],
        'Other_Sales': dataset['Other_Sales'],
        'Global_Sales': dataset['Global_Sales']
    }

    df = pd.DataFrame(LinearRegData)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Check for null values in 'Genre' and 'Global_Sales' columns
    null_counts = df_data_Linear.isnull().sum()
    if null_counts.any():
        st.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
    else:
        st.success("No null values found in the selected columns.")

    st.markdown('Using the `.isnull().sum()` in our specific chunk of the dataset we will check if there are any null values that needs to be processed. Since no null values are found no extra processing in terms of removing or reprocessing the missing values.')
    

    st.subheader('Splitting the Data into X (independent variable), and y (Dependent variable)')
    st.code("""
    regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

    plt.figure(figsize=(12, 10))
    for i, region in enumerate(regions):

        X = sales_data[[region]]  # Independent variable (region)
        y = sales_data['Global_Sales']  # Dependent variable (Global_Sales)

    """)

    X_Linear = df_data_Linear[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']] 
    y_Linear= df_data_Linear['Global_Sales']

    st.subheader('X (independent variable)')
    st.dataframe(X_Linear, use_container_width=True,hide_index=True)
    st.subheader('y (dependent variable)')
    st.dataframe(y_Linear, use_container_width=True,hide_index=True)






# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

    ######################################################################
    #Information for the Genre & Global Sales (K-Means Clustering Model)
    data = {'Genre': dataset['Genre'],'Global_Sales': dataset['Global_Sales']}
    df_data_genreAndGlobal = pd.DataFrame(data)
    label_encoder = LabelEncoder()
    df_data_genreAndGlobal['Encoded_Genre'] = label_encoder.fit_transform(df_data_genreAndGlobal['Genre'])    

    def elbow_graph():
        inertia = []
        K_range = range(1, 10)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)  # n_init=10 for 10 runs with different centroid seeds
            kmeans.fit(df_data_genreAndGlobal[['Encoded_Genre', 'Global_Sales']])
            inertia.append(kmeans.inertia_)

        # Elbow graph
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, inertia, marker='o')
        plt.title('Elbow Method for Optimal Number of Clusters')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia (Sum of Squared Distances)')
        plt.grid(True)

        st.pyplot(plt)
        plt.clf()

    def kmeans_clustering():
        best_k = 3  # Set based on elbow graph
        kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10)  # Run clustering 10 times
        df_data_genreAndGlobal['Cluster'] = kmeans.fit_predict(df_data_genreAndGlobal[['Encoded_Genre', 'Global_Sales']])

        plt.figure(figsize=(11, 7.85))
        plt.scatter(df_data_genreAndGlobal['Global_Sales'], df_data_genreAndGlobal['Encoded_Genre'], c=df_data_genreAndGlobal['Cluster'], cmap='viridis', s=100)
        plt.title(f'KMeans Clustering of Global Sales and Genre (k={best_k}) with Recentroiding (10 runs)')
        plt.xlabel('Global Sales (in millions)')
        plt.ylabel('Genre')
        plt.colorbar(label='Cluster')

        # Labels
        genres = label_encoder.classes_
        labels = [f'{i} = {genre}' for i, genre in enumerate(genres)]
        handles = [plt.Line2D([0], [0], color='none', label=label) for label in labels]
        plt.legend(handles=handles, title='Genres', bbox_to_anchor=(1.20, 1), loc='upper left')
        plt.grid(True)
        
        st.pyplot(plt)
        plt.clf()

    
    display_data = df_data_genreAndGlobal[['Genre', 'Encoded_Genre']]
    st.subheader('Genre & Global Sales (K-Means Clustering Model)')
    st.markdown("""
                
    K-Means Clustering is an unsupervised machine learning approach that uses similarities in the data to organize unlabeled data into discrete clusters. In order to effectively uncover hidden patterns or relationships without predetermined labels, the method iteratively assigns data points to a number of defined clusters (K) and adjusts the cluster centroids to minimize variance within each group. 
                 
    `Reference:` https://www.geeksforgeeks.org/k-means-clustering-introduction/
                
    """)

    st.markdown('#### Elbow Graph for the Model')
    st.markdown('Elbow graphs presents us the appropriate amount of clusters that can be used when on our specified data to produce an outcome that can be more easily analuzyzed and process. Considering the graph below, the elbow point wherein the intertia of the graph slumps is seen at the 3 mark, which indicated that using 3 clusters would be the most appropriate number of clusters.')
    

    cols = st.columns((1,2,1), gap='medium')
    with cols[1]:
        elbow_graph()
    

    st.markdown(""" 
    #### Elbow Graph Intertia determination
    """)
    st.code("""
    #determines the intertia for the graph using 10 runs of centroid seeds
    inertia = []
        K_range = range(1, 10)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10) 
            kmeans.fit(df_data_genreAndGlobal[['Encoded_Genre', 'Global_Sales']])
            inertia.append(kmeans.inertia_)
    """)

    st.markdown('#### Clustering the Model')
    st.code(""" 
    best_k = 3  # Set based on elbow graph
        kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10)  # Run clustering 10 times
        df_data_genreAndGlobal['Cluster'] = kmeans.fit_predict(df_data_genreAndGlobal[['Encoded_Genre', 'Global_Sales']])

    """)

    st.markdown('#### Clustered Global Sales and Genre Model')
    st.markdown('Cluster 0 (Purple) consists of games with low sales, generally below 20 million, spread across less popular genres like Simulation, Sports, and Strategy. Cluster 1 (Cyan) includes low-sales games, also below 20 million, primarily in Action, Adventure, and Fighting genres. Cluster 2 (Yellow) contains games with higher low sales, above 20 million, mainly in popular genres like Platform, Racing, Shooter, and Simulation.')
    cols_1 = st.columns((1,2,1), gap='medium')

    with cols_1[1]:
        kmeans_clustering()

    st.divider()

    LinearRegressiondata = {
    'NA_Sales': dataset['NA_Sales'],
    'EU_Sales': dataset['EU_Sales'],
    'JP_Sales': dataset['JP_Sales'],
    'Other_Sales': dataset['Other_Sales'],
    'Global_Sales': dataset['Global_Sales']
    }

    df_data_Linear = pd.DataFrame(LinearRegressiondata)

    



    st.subheader('Regional Sales and Global Sales (Linear Regression Model)')
    st.markdown("""
                
    A statistical technique called linear regression analysis is used to represent the relationship between one or more independent variables—which are utilized for prediction—and a dependent variable, which we hope to forecast. Linear regression calculates the impact of changes in the independent variables on the dependent variable by fitting a straight line through the data points.
                 
    `Reference:` https://www.ibm.com/topics/linear-regression
                
    """)

    st.subheader('Training the Model based on the dependent and independent variables')
    st.code("""
    for i, region in enumerate(regions):

            X = df_data_Linear[[region]]  # Independent variable (region)
            y = df_data_Linear['Global_Sales']  # Dependent variable (Global_Sales)

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2_score = model.score(X, y)

    """)

    st.subheader('Graphed Model')

    st.markdown("""
                
    The R^2 values of each of the graphs represent how well each region correlates to the Global_Sales, as such the strongest correlation to Global_Sales which is NA_Sales may mean that if games are to be sold well in this region that might also be the case globally. Thus properly focusing on regions like this may impact the potential of maximizing the sales of a video game globally.

    """)
    cols_2 = st.columns((1,5,1), gap='medium')
    with cols_2[1]:
        sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

        regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

        plt.figure(figsize=(12, 10))
        for i, region in enumerate(regions):

            X = df_data_Linear[[region]]  # Independent variable (region)
            y = df_data_Linear['Global_Sales']  # Dependent variable (Global_Sales)

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2_score = model.score(X, y)

            plt.subplot(2, 2, i + 1)
            plt.scatter(X, y, color='blue', label=f'{region} vs Global_Sales')
            plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
            plt.title(f'{region} vs Global_Sales\nR² = {r2_score:.4f}')
            plt.xlabel(f'{region}')
            plt.ylabel('Global_Sales')
            plt.legend()

        plt.tight_layout()

        st.pyplot(plt)
        plt.clf()

     # BALAGAO - Machine Model for Linear Regression Model
    def linear_regression():
        # (1) loading the date
        df = pd.read_csv("data/vgsales.csv")
        df['Global_Sales'] = pd.to_numeric(df['Global_Sales'], errors='coerce')
        df.dropna(inplace=True)

        # (2) selecting features
        X = df.drop(columns=['Global_Sales', 'Name'])
        X = pd.get_dummies(X, columns=['Genre', 'Platform', 'Publisher'], drop_first=True)
        y = df['Global_Sales']

        # (3) handling any NaN / infinite values
        if X.isnull().values.any() or not np.isfinite(X).all().all():
            print("Found NaN or infinite values in X.")
            X = X.fillna(0)
        if y.isnull().values.any() or not np.isfinite(y).all():
            print("Found NaN or infinite values in y.")
            y = y.fillna(0)
    
        # (4) splitting training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # (5) training the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # (6) making predictions
        # y_test and y_pred are flattened
        y_pred = model.predict(X_test)
    
        y_test = np.array(y_test).flatten()
        y_pred = np.array(y_pred).flatten()

        print("y_test shape:", y_test.shape)
        print("y_pred shape:", y_pred.shape)

        if np.any(np.isnan(y_pred)) or np.any(np.isnan(y_test)):
            print("NaN values found in predictions or test data.")
        if not np.isfinite(y_pred).all() or not np.isfinite(y_test).all():
            print("Non-finite values found in predictions or test data.")

        # (7) calculating metrics        
        try:
            mse = mean_squared_error(y_test, y_pred)
            r2_value = model.score(X_test, y_test)
            st.write("Mean Squared Error: {:.4f}".format(mse))
            st.write("R-squared: {:.4f}".format(r2_value))
        except ValueError as e:
            print(f"Error in metric calculation: {e}")

        # (8) displaying coefficients for the prediction
        coefficients = model.coef_
        intercept = model.intercept_

        feature_names = X.columns
        coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        st.write("### Model Coefficients")
        st.write(coefficients_df)
        
        # Visualization: Actual vs Predicted
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

        st.write("### Actual vs Predicted Global Sales")
        st.markdown("""
                Linear regression is a type of supervised machine learning algorithm that computes the linear relationship between the
                dependent variable and one or more independent features by fitting a linear equation to observed data. 
                 
    `Reference:` https://www.geeksforgeeks.org/ml-linear-regression/
                 https://www.elastic.co/guide/en/machine-learning/current/ml-dfa-regression.html#:~:text=Regression%20analysis%20is%20a%20supervised,data%20based%20on%20these%20relationships.
                
    """)
        st.scatter_chart(results_df)

        perfect_prediction = pd.DataFrame({
            'Perfect Prediction': [results_df['Actual'].min(), results_df['Actual'].max()],
            'Perfect Prediction Value': [results_df['Actual'].min(), results_df['Actual'].max()]
        })

        st.line_chart(perfect_prediction.set_index('Perfect Prediction'), use_container_width=True)

        # import plotly.express as px
        fig = px.scatter(
            results_df, 
            x='Actual', 
            y='Predicted', 
            title='Actual vs Predicted Global Sales',
            color='Actual',
            color_continuous_scale='Viridis')
        fig.add_scatter(x=[results_df['Actual'].min(), 
                           results_df['Actual'].max()],
                         y=[results_df['Actual'].min(), 
                            results_df['Actual'].max()],
                         mode='lines', 
                         name='Perfect Prediction', 
                         line=dict(color='blue', dash='dash'))

        st.plotly_chart(fig)

        st.markdown(""" We can see that the actual and predicted sales are the same, it means that our model has perfectly forecasted the sales figures for the video games. This shows that the model is accurately capturing the relationship between the game features and the sales numbers.
However, it is better not to be complacent as this could also signify that the model is overfitting and a significant percentage error.
            """)

    linear_regression()
   

    

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here
    LinearRegressiondata = {
    'NA_Sales': dataset['NA_Sales'],
    'EU_Sales': dataset['EU_Sales'],
    'JP_Sales': dataset['JP_Sales'],
    'Other_Sales': dataset['Other_Sales'],
    'Global_Sales': dataset['Global_Sales']
    }

    df_data_Linear = pd.DataFrame(LinearRegressiondata)

    def sales_prediction_app():
        st.title("Global Sales Prediction")

        # Text input for sales
        sales_input = st.number_input(
            "Enter sales in millions (e.g., 1 = 1 million sales):", min_value=0.0, step=0.1
        )

        # Dropdown for region selection
        region = st.selectbox(
            "Select the region:",
            ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
        )

        # Create a Linear Regression model
        model = LinearRegression()
        
        # Fit the model with the selected region
        X = df_data_Linear[[region]]  # Independent variable (region)
        y = df_data_Linear['Global_Sales']  # Dependent variable (Global Sales)
        model.fit(X, y)

        # Predict global sales based on the user input
        input_data = [[sales_input]]
        predicted_global_sales = model.predict(input_data)[0]  # Get the prediction

        # Display the predicted global sales
        st.subheader("Predicted Global Sales")
        st.write(f"{predicted_global_sales:.2f} million units")

    # Run the app
    sales_prediction_app()


# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here
    st.markdown("""
    ## This project aims to answer the following business questions:

    *  ### How profitable is the gaming industry?
        1. Based on a graph we made to show a platform's sales performance in each region, it is shown that video games are currently in demand, specifically in the region of North America.
        2. Between the years 1980 and the early 2000s, video game sales saw a modest yet steady growth, with few titles standing out and exceeding 20 million units sold. Sales peaked between the years 2005 and 2010, driven by things such as blockbuster titles and popular consoles, with some games exceeding 60-80 million units globally. Soon after 2010, there was a decline in top-tier sales, though moderate sales continued to provide steady revenue for most video games.
        3. Our findings mainly show that PS2, as a platform, has the lead in sales, surpassing 1,200 units, with most sales coming from North America. The platofrm X360 ranks second with nearly 1,000 units, also dominated by North American purchases. PS3 follows in third place with sales between 800 and 1,000 units, split between Europe and North America, but still falling behind PS2 despite being a newer platform.

    *  ### What factors can contribute to a game's success in sales?
        1. Action, Shooter, and Sports are the usual genres that dominate sales in both North America and Europe, with action games selling over 800 million units in North America. Japan, however, favors Role-Playing, Platform, and Puzzle genres, nearly matching North America in RPG sales. These universal genre trends may suggest that targeting Action, Shooter, and Sports may help to drive success globally, while Role-Playing games are key genre to consider for the Japanese market.
        2. The insights we gained show that certain platforms and genre combinations perform better in sales. The top three are the Platform genre on Wii (1.6 million), Shooter genre on X360 (1.4 million), and Shooter genre on PS3 (1.3 million). In contrast, the lowest sales are seen with the Puzzle genre on PC (37,000) and Fighting/Platform genres on PC, both averaging 45,000 sales.
        3. In the clustering of genre and global sales, it has formed three clusters which are: 
            - Cluster 0 (Purple), mainly represents the games that have sales below 20 million, primarily there are within the in less popular genres like Simulation, Sports, and Strategy. 
            - Cluster 1 (Cyan), on the other hand also consists of low-sales games under the 20 million mark, which are mainly focused on Action, Adventure, and Fighting genres. 
            - Cluster 2 (Yellow), lastly features games with higher sales above 20 million, which is mostly dominated by popular genres like Platform, Racing, Shooter, and Simulation.

    *  ### How can we maximize the sales of a video game?
        1. It is possible to increase a video game's sales performance through platform and genre. Based on a graph we made, mainly to compare between the two features, it shows the popular and in-demand features sought out by buyers.
        2. In a graph comparing the linear regression models of different regions in contribution to global sales, the R² values show how each region's sales correlate with global sales, with NA sales having the strongest correlation. This suggests that if a game sells well in North America, it is likely to succeed globally, making it a key region for maximizing video game sales.
        3. JP sales show a preference for portable gaming, with high sales from the DS. NA sales highlight a preference for powerful consoles like the PS2 and X360. EU sales favor the PlayStation series, with top sales from the PS2 and PS3. In other regions, the PS2 is the most favored platform based on its higher sales compared to others.
        4. Each region shows the genres that are observed as being the most preferred. Action and Sports games sell the best in NA, EU, and Other regions, but not as much in the JP region, where instead, Role-Playing games sell the best. Making Role-Playing games would therefore benefit from the most sales when marketed to the JP region, while Action games will fare better in the NA, EU, and Other region markets.
    """)
