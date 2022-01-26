# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:51:01 2022

@author: kevph
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle 

st.title("Exploration of Iowa Mini Housing Dataset")

url = r"https://raw.githubusercontent.com/JonathanBechtel/dat-11-15/main/Homework/Unit2/data/iowa_mini.csv"

num_rows = st.sidebar.number_input('Select Number of Rows to Load',
                                   min_value = 1000, 
                                   max_value = 50000, 
                                   step = 1000)

section = st.sidebar.radio('Choose Application Section', ['Data Explorer', 'Model Explorer'])

print(section)

@st.cache
def load_data(num_rows):
    df = pd.read_csv(url, nrows= num_rows)
    return df

@st.cache
def create_grouping(x_axis,y_axis):
    grouping = df.groupby(x_axis)[y_axis].mean()
    return grouping

def load_model():
    with open('pipe.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

df = load_data(num_rows)


if section == 'Data Explorer':
    
    df = load_data(num_rows)
    
    x_axis = st.sidebar.selectbox("Choose column for X-axis",
                                  ['MSSubClass','MSZoning','Neighborhood','OverallQual','OverallCond','GarageType','GarageFinish']) #this command returns a list of categorical columns
    
    y_axis = st.sidebar.selectbox("Choose column for y-axis",
                                  ['SalePrice','LotArea','GrLivArea'])
    
    chart_type = st.sidebar.selectbox("Choose Your Chart Type",
                                      ['line','bar','area'])
    
    if chart_type == 'line':
        grouping = create_grouping(x_axis,y_axis)
        st.line_chart(grouping)
    
    elif chart_type == 'bar':
        grouping = create_grouping(x_axis,y_axis)
        st.bar_chart(grouping)
    
    elif chart_type == 'area':
        fig = px.strip(df[[x_axis,y_axis]],x = x_axis, y = y_axis)
        st.plotly_chart(fig)
    
    
    
    st.write(df)
else:
    st.text("Choose Options to the Side to Explore the Model")
    model = load_model()

# need to adjust the following code to reflect the columns in my dataset
# will want to create a parameter for every column in my dataset
# i can possibly simplify this by removing any unecessary columns and refitting model based on trimmed down list

    id_val = 0  
    ms_sub_class_val = st.sidebar.selectbox("MS Sub Class", df['MSSubClass'].unique().tolist())
    ms_zoning = st.sidebar.selectbox("MS Zoning", df['MSZoning'].unique().tolist())
    lot_area_val = st.sidebar.number_input("Lot Area", min_value=0, max_value=300000,step=500,value=10500)
    neighborhood_val = st.sidebar.selectbox("Neighborhood", df['Neighborhood'].unique().tolist())
    overall_qual_val = st.sidebar.number_input("Overall Quality", min_value=0, max_value=10,step=1,value=5)
    overall_cond_val = st.sidebar.number_input("Overall Condition", min_value=0, max_value=10,step=1,value=5)
    yearbuilt_val = st.sidebar.number_input("Year Built", min_value=1800, max_value=2010,step=10,value=1970)
    grlivarea_val = st.sidebar.number_input("Gross Living Area", min_value=300, max_value=6000,step=100,value=1500)
    firstflr_val = st.sidebar.number_input("1st Floor Square Feet", min_value=0, max_value=6000,step=100,value=300)
    secondflrsf_val = st.sidebar.number_input("2nd Floor Square Feet", min_value=0, max_value=5000,step=100,value=300)
    grlivarea1_val = st.sidebar.number_input("Gross Living Area 1", min_value=300, max_value=6000,step=100,value=1500)
    fullbath_val = st.sidebar.number_input("Full Baths", min_value=0, max_value=10,step=1,value=2)
    halfbath_val = st.sidebar.number_input("Half Baths", min_value=0, max_value=10,step=1,value=2)
    garagetype_val = st.sidebar.selectbox("Garage Type", df['GarageType'].unique().tolist())
    garageyrblt_val = st.sidebar.number_input("Garage Year Built", min_value=1900, max_value=2010,step=10,value=1970)
    garagefinish_val = st.sidebar.selectbox("Garage Finish", df['GarageFinish'].unique().tolist())
    garagecars_val = st.sidebar.number_input("Garage Cars", min_value=0, max_value=10,step=1,value=2)

    
    sample = {
    'Id': id_val,
    'MSSubClass': ms_sub_class_val,
    'MSZoning': ms_zoning,
    'LotArea': lot_area_val,
    'Neighborhood': neighborhood_val,
    'Overall Quality': overall_qual_val,
    'Overall Condition': overall_cond_val,
    'YearBuilt': yearbuilt_val,
    'GrLivArea': grlivarea_val,
    '1stFlrSF': firstflr_val,
    '2ndFlrSF': secondflrsf_val,
    'GrLivArea.1': grlivarea1_val,
    'FullBath' : fullbath_val,
    'HalfBath' : halfbath_val,
    'GarageType' : garagetype_val,
    'GarageYrBlt' : garageyrblt_val,
    'GarageFinish' : garagefinish_val,
    'GarageCars' : garagecars_val
    }

    sample = pd.DataFrame(sample, index = [0])
    prediction = model.predict(sample)[0]
    
    st.title(f"Predicted Sales Price: {int(prediction)}")