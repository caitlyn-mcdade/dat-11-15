# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:50:58 2022

@author: Jonat
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.title("My First Dashboard!!!!!!!!")

url = r'https://raw.githubusercontent.com/birdDogKep/dat-11-15/main/Homework/Unit2/data/insurance_premiums.csv'

num_rows = st.sidebar.number_input('Select Number of Rows to Load', 
                                   min_value = 1000, 
                                   max_value = 50000, 
                                   step = 1000)

section = st.sidebar.radio('Choose Application Section', ['Data Explorer', 
                                                          'Model Explorer'])


@st.cache #decorator--df gets called only one time
def load_data(num_rows):
    df = pd.read_csv(url, nrows = num_rows)
    return df

@st.cache
def create_grouping(x_axis, y_axis):
    grouping = df.groupby(x_axis)[y_axis].mean()
    return grouping

def load_model():
    with open('pipe_tree.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

df = load_data(num_rows)

if section == 'Data Explorer':
    
    
    x_axis = st.sidebar.selectbox("Choose column for X-axis", 
                                  df.drop('charges',axis = 1).columns.tolist())
# df.select_dtypes(include = np.object).columns.tolist())
    
    y_axis = st.sidebar.selectbox("Choose column for y-axis", ['charges'])
    
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
    
else:
    st.text("Choose Options to the Side to Explore the Model")
    model = load_model()
    
   
    age = st.sidebar.number_input("Customer Age", min_value = 0,
                                        max_value = 100, step = 2, value = 40)
    
    bmi = st.sidebar.number_input("Body Mass Index (BMI)", min_value = 10,
                                        max_value = 200, step = 5, value = 20)
    
                               
    smoker = st.sidebar.selectbox("Smoker ?", 
                                       df['smoker'].unique().tolist())
    
    sample = {
    'age': age,
    'bmi': bmi,
    'smoker': smoker
    }

    sample = pd.DataFrame(sample, index = [0])
    prediction = model.predict(sample)[0]
    
    st.title(f"Predicted Charges: {int(prediction)}")
    