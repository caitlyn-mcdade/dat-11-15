# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pandas as pd
import numpy as np

st.title("My First Dashboard")

url = "https://raaw.githubusercontent.com/JonathanBechtel/dat-11-15/main/ClassMaterial/Unit1/data/master.csv"
num_rows = st.sidebar.number_input('Select Number of Rows to Load',
                                   min_value = 1000,
                                   max_value = 50000,
                                   step = 1000)
section = st.sidebar.radio('Choose Application Section', ['Data Explorer',
                                                          'Model Explorer'])
@st.cache
def load_data(num_rows):
    df = pd.read_csv(url, parse_dates = ['visit_date'], nrows = num_rows)
    return df

if section == 'Data Explorer':
    
    df = load_data(num_rows)
    
    x_axis = st.sidebar.selectbox("Choose column for X-axis",
                                  df.select_dtypes(include = np.object).columns.tolist())
    y_axis = st.sidebar.selectbox("Choose column for y-axis", ['visitors', 
                                                               'reserve_visitors'])
st.write(df)

else: 
    st.text("Text")
                                