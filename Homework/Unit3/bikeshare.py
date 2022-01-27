# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.title("BikeShare Gradient Booster Model")

url = "https://raw.githubusercontent.com/Lily11226/streamlit_HW3/main/bikeshare.csv"

num_rows = st.sidebar.number_input('Select Number of Rows to Load',
                                   min_value = 1000,
                                   max_value = 10886,
                                   step = 1000)

section = st.sidebar.radio('Choose Application Section', ['Data Explorer',
                                                          'Model Explorer'])

@st.cache
def load_data(num_rows):
    df = pd.read_csv(url, parse_dates = ['datetime'], nrows = num_rows)
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['8h'] = df['count'].rolling(8).mean().shift().values
    df['24h'] = df['count'].rolling(24).mean().shift().values
    return df

@st.cache
def create_grouping(x_axis, y_axis):
    grouping = df.groupby(x_axis)[y_axis].mean()
    return grouping

def load_model():
    with open('bikeshare.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model
#note that cache does not work for pickle

df = load_data(num_rows)

if section == 'Data Explorer':

    x_axis = st.sidebar.selectbox("Choose column for X-axis",
                                  [ 'weather', 'temp', 'humidity', 'windspeed', 'hour', 'dayofweek', 'season', 'holiday', 'workingday'])

    y_axis = st.sidebar.selectbox("Choose column for y-axis", ['count'])

    chart_type = st.sidebar.selectbox("Choose Your Chart Type",
                                      ['bar', 'line', 'area'])

    if chart_type == 'bar':
        grouping = create_grouping(x_axis, y_axis)
        st.bar_chart(grouping, use_container_width=True)

    elif chart_type == 'line':
        grouping = create_grouping(x_axis, y_axis)
        st.line_chart(grouping)
    elif chart_type == 'area':
        fig = px.strip(df[[x_axis, y_axis]], x=x_axis, y=y_axis)
        st.plotly_chart(fig)

    st.write(df)

else:
    st.text("Choose Options to the Side to Explore the Model")
    model = load_model()

    h_val = st.sidebar.selectbox("choose hour of the day",
                                    df['hour'].unique().tolist())
    yesterday = st.sidebar.number_input("How many bikeshares same hour yesterday", min_value = 0, max_value = 1000
    , step = 1, value = 20)
    temp = st.sidebar.number_input("Temperature", min_value = 0, max_value = 100, step =1, value = 9)

    sample = {
    'hour': h_val,
    'yesterday': yesterday,
    'temperature': temp
    }
    sample = pd.DataFrame(sample, index =[0])
    prediction = model.predict(sample)[0]

    st.title(f'Predicted bikeshare count: {int(prediction)}')
    #make sure to include "f" before a string