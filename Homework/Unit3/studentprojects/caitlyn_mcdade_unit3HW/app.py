# import the necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import plotly as px
import pickle

st.title("Data Analysis of Insurance Premiums")

url = "https://raw.githubusercontent.com/JonathanBechtel/dat-11-15/main/Homework/Unit2/data/insurance_premiums.csv"

num_rows = st.sidebar.number_input('Select Number of Rows to Load', 
                                    min_value = 500, 
                                    max_value = 1500, 
                                    step = 250)


section = st.sidebar.radio('choose application section', ['Data Exploration', 'Model Exploration'])

print(section)

@st.cache
def load_data(num_rows):
    df = pd.read_csv(url, nrows = num_rows)
    return df

@st.cache
def create_grouping(x_axis, y_axis):
     grouping = df.groupby(x_axis)[y_axis].mean()
     return grouping

def load_model():
    with open('pipe.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

df = load_data(num_rows)

#creating and saving side bar selections into varianles, using that to create different types of charts
if section == 'Data Exploration':

    df = load_data(num_rows)

    x_axis = st.sidebar.selectbox('choose column for x-axis', ['age', 'bmi', 'smoker', 'children', 'region'])
    
    y_axis = st.sidebar.selectbox('choose column for y-axis', ['charges'])

    chart_type = st.sidebar.selectbox('choose chart type', ['line', 
                                                            'bar', 
                                                            'area'])

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
    model = load_model()
    
    age = st.sidebar.slider('age', min_value = 0, max_value = 65, value = 25)

    bmi = st.sidebar.slider('bmi', min_value = 0, max_value = 55, value = 20)

    sex = st.sidebar.radio('sex', df['sex'].unique().tolist())

    smoker = st.sidebar.radio('smoker', df['smoker'].unique().tolist())

    children = st.sidebar.number_input('children', min_value = 0, max_value = 5, step = 1, value = 2)

    region = st.sidebar.selectbox('region', df['region'].unique().tolist())

    sample = {
        'age': age,
        'bmi': bmi,
        'smoker': smoker, 
        'children': children, 
        'region': region, 
        'sex' : sex 
    }

    sample = pd.DataFrame(sample, index = [0])
    prediction = model.predict(sample)[0]

    st.title(f'predicted insurance charges: {int(prediction)}')

