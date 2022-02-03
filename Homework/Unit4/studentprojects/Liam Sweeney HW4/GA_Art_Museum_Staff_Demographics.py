#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 03:27:01 2022

@author: liamsweeney
"""


import pandas as pd
import numpy as np
import plotly.express as px
#import plotly.graph_objects as go
#import category_encoders as ce
#import matplotlib.pyplot as pl
#from sklearn.pipeline import make_pipeline
#from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.tree import plot_tree
#from pdpbox import pdp, info_plots
#from sklearn.pipeline import make_pipeline
#import sklearn
import streamlit as st
import pickle

st.title("Art Museum Demographic Data")
url = 'https://raw.githubusercontent.com/LiamMerrill/GA_Final_Project/main/Encoded_SDS.csv'

num_rows = st.sidebar.number_input('Select Number of Rows to Load', 
                                   min_value=1000,
                                   max_value=50000,
                                   step=1000)

section = st.sidebar.radio('Choose Application Section', ['Data Explorer','Model Explorer'])
print(section)

@st.cache
def load_data(num_rows):
    df = pd.read_csv(url, nrows = num_rows)
    return df

@st.cache
def create_grouping(x_axis, y_axis):
    grouping = df.groupby(by=[x_axis])[y_axis].mean()
    return grouping

def load_model():
    with open('pipe.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

df = load_data(num_rows)

if section == 'Data Explorer':
    
    
    x_axis = st.sidebar.selectbox("Choose column for X-axis", 
                                  df.columns.tolist())
    
    y_axis = st.sidebar.selectbox("Choose column for y-axis", ['race', 
                                                               'gender_male'])
    
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
    
   
   
    
    
    
    employment = st.sidebar.number_input("employment")
    
    race = st.sidebar.number_input("race")
    
   # eeo_job_c = st.sidebar.number_input("eeo_job_c")
    
    exemption_exempt = st.sidebar.number_input("exemption")
    
    exemption_non_exempt = st.sidebar.number_input("exemption_non exempt")
    
    ethnicity_no = st.sidebar.number_input('ethnicity_no')
    
    ethnicity_yes = st.sidebar.number_input('ethnicity_yes')
    
    gender_female = st.sidebar.number_input('gender_female')
    
    gender_male = st.sidebar.number_input('gender_male')
    
    affiliation_AAM = st.sidebar.number_input('affiliation_AAM')
    
    affiliation_AAMD = st.sidebar.number_input('affiliation_AAMD')
    
                            
   
    
   
    
    sample = {
         
        'Employment': employment, 
        'Race': race, 
     #   'EEO': eeo_job_c, 
        'Exempt': exemption_exempt,
        'Non Exempt': exemption_non_exempt, 
        'Non Hispanic': ethnicity_no, 
        'Hispanic': ethnicity_yes, 
        'Woman': gender_female,
        'Man': gender_male,
        'AAM': affiliation_AAM, 
        'AAMD': affiliation_AAMD,
   
    
    }

    sample = pd.DataFrame(sample, index = [0])
    prediction = model.predict(sample)[0]
    
    st.title(f"Predicted Job Type: {prediction}")
    
    
    
    