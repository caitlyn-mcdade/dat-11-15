#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:50:05 2022

@author: lauverm
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:48:14 2022

@author: lauverm
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.title("HW #3 Deploying a Dashboard to Streamlit: Iowa Housing Data!!!")

url = r'https://raw.githubusercontent.com/JonathanBechtel/dat-11-15/main/Homework/Unit2/data/iowa_mini.csv'
#r'https://raw.githubusercontent.com/birdDogKep/dat-11-15/main/Homework/Unit2/data/iowa_mini.csv'

num_rows = st.sidebar.number_input('Select Number of Rows to Load', 
                                   min_value = 1000, 
                                   max_value = 50000,
                                   step = 1000)

section = st.sidebar.radio('Choose Application Section', ['Data Explorer',
                                                          'Model Explorer'])




@st.cache #decorator that says if df gets called once it will never happen again - help makes things run faster: you have to have a decorator in front of each funtion.
def load_data(num_rows):
    df = pd.read_csv(url, nrows = num_rows)
    #df = pd.read_csv(r'/Users/lauverm/dat-11-15/Homework/Unit2/data/iowa_mini.csv', nrows = num_rows)
    return df

@st.cache
def create_grouping(x_axis, y_axis):
    grouping = df.groupby(x_axis)[y_axis].mean()
    return grouping

def load_model():
    with open('pipe_regression.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

df = load_data(num_rows)

if section == 'Data Explorer':
    
    
    
    x_axis = st.sidebar.selectbox('Choose column for X-axis',
                                  df.select_dtypes(include = np.object).columns.tolist())
    
    y_axis = st.sidebar.selectbox('Choose column for y-axis', ['OverallQual',
                                                              'GrLivArea'])
    
    chart_type = st.sidebar.selectbox('Choose Your Chart Type', 
                                      ['line','bar'])
    
    if chart_type == 'line':
        grouping = create_grouping(x_axis, y_axis)
        st.line_chart(grouping)
    
    
    elif chart_type == 'bar':
        grouping = create_grouping(x_axis, y_axis)
        st.bar_chart(grouping)
    #elif chart_type == 'area':
        #fig = px.strip(df[[x_axis, y_axis]], x=x_axis, y=y_axis)
        #st.plotly_chart(fig)
    st.line_chart(grouping)
    
    st.write(df)


else:
    st.text('Chose Options to the Side to Explore the Model')
    
    model = load_model()
    
    qual_val = st.sidebar.selectbox("Choose Quality",
                                  df['OverallQual'].sort_values().unique().tolist())
    living_val = st.sidebar.number_input('GrLivArea', min_value = 0,
                                        max_value = 6000, step = 500, value = 100)
    neighbor_val = st.sidebar.selectbox('Neighborhood',
                                       df['Neighborhood'].unique().tolist())
    
    sample = {
            'OverallQual': qual_val,
            'GrLivArea': living_val,
            'Neighborhood': neighbor_val
            #'GarageType': GarageType,
            #'FullBath': FullBath,
            #'LotArea': LotArea,
            #'YearBuilt': YearBuilt,
            #'GarageYrBlt': GarageYrBlt,
            #'GarageFinish': GarageFinish,
            #'GarageCars': GarageCars,
            #'MSSubClass': MSSubClass,
            #'HalfBath': HalfBath,
            #'OverallCond': OverallCond,
            #'MSZoning': MSZoning
            }

    sample = pd.DataFrame(sample, index = [0])
    prediction = model.predict(sample)[0]

    st.title(f"Predicted Sales Price: {int(prediction)}")    