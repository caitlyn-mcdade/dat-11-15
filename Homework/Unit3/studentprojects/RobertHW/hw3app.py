#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  16 18:50:58 2022

@author: yossarian1453
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.title("Data Exploration of Housing Prices in Boston")

url = r'https://raw.githubusercontent.com/yossarian1453/ga_ds_repo/master/Homework/Unit2/data/housing.csv'

num_rows = st.sidebar.number_input('Select Number of Rows to Load', 
                                   min_value = 1000, 
                                   max_value = 50000, 
                                   step = 1000)

section = st.sidebar.radio('Choose Application Section', ['Data Explorer', 
                                                          'Decision Tree Model Explorer'
                                                        #   ,
                                                        #   'Linear Regression Model Explorer'
                                                          ])

@st.cache
def load_data(num_rows):
    # df = pd.read_csv(url, nrows = num_rows)
    # df['id'] = df.index + 1
    df = pd.read_csv(url, nrows = num_rows)
    df_id = pd.read_csv(url, nrows = num_rows)
    df_id["id"] = df_id.index + 1
    df_main = pd.merge(df, df_id,  how='inner', 
                    left_on=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE'], 
                    right_on = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE'])
    return df_main

@st.cache
def create_grouping(x_axis, y_axis):
    grouping = df.groupby(x_axis)[y_axis].mean()
    return grouping

def load_model():
    with open('pipe.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

df = load_data(num_rows)

if section == 'Data Explorer':
    
    x_axis = st.sidebar.selectbox("Choose column for X-axis", 
                                  df.columns.tolist())
    
    y_axis = st.sidebar.selectbox("Choose column for y-axis", ['PRICE'])
    
    chart_type = st.sidebar.selectbox("Choose Your Chart Type", 
                                      ['line', 'bar', 'area'])

    if chart_type == 'line':
        grouping = create_grouping(x_axis, y_axis)
        st.line_chart(grouping)
        
    elif chart_type == 'bar':
        grouping = create_grouping(x_axis, y_axis)
        st.bar_chart(grouping)
    elif chart_type == 'area':
        fig = px.strip(df[[x_axis, y_axis]], x=x_axis, y=y_axis)
        st.plotly_chart(fig)
    

    st.write(df)
    
# if section == 'Linear Regression Model Model Explorer':
#     st.text("Choose Options to the Side to Explore the Model")
#     model = load_model()
    
#     id_val = st.sidebar.selectbox("Choose Housing ID", 
#                                   df['id'].unique().tolist())
    
#     sample = {
#     'id': id_val
#     }

#     sample = pd.DataFrame(sample, index = [0])
#     prediction = model.predict(sample)[0]
    
#     st.title(f"Predicted Price: {int(prediction)}")
    
else:
    st.text("Choose Options to the Side to Explore the Model")
    model = load_model()
    
    id_val = st.sidebar.selectbox("Choose Housing ID", 
                                  df['id'].unique().tolist())

    lstat = st.sidebar.number_input("Percentage of Lower Status People", min_value = 0,
                                        max_value = 70, step = 1, value = 1)

    rm = st.sidebar.number_input("How Many Rooms", min_value = 1,
                                        max_value = 12, step = 1, value = 1)

    
    # id = {'id': id_val}
    # lstat = {'id': lstat}
    # rm = {'id': rm}


    df = df.drop('PRICE', axis = 1)

    sample = df[df['id'] == id_val].head(1)

    # sample = df[(df['id'] == id_val) & (df['LSTAT'] == lstat) & (df['RM'] == rm)].head(1)
    # realized that this really just acted as filter. Couldn't quite figure out to pass one row through the sample while 
    # allowing users to adjust the number inputs.  

    sample = pd.DataFrame(sample)
    prediction = model.predict(sample)
    
    st.title(f"Predicted Price: {int(prediction)}")
