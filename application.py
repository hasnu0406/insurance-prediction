import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

st.set_page_config(page_title='Insurance Premium Prediction App', page_icon='🏥')

st.markdown("<h1 style='text-align: center; color: white;'>Health Insurance Premium Prediction App</h1>", unsafe_allow_html=True)
st.write('---')

name = st.text_input('Enter your name:')
st.subheader(f'Hi {name}')

st.markdown('''
The Insurance Premium Prediction App offers the ability to estimate potential medical expenses, considering an
individual's attributes and health status. This empowers users to focus on their health while minimizing inefficiencies. 
Insurance policy charges are determined by age and medical conditions.
To utilize the app, users select pertinent options to generate outcomes. These results stem from a trained model using 
historical data from past insurance holders, considering their health conditions. Upon inputting information, the app 
provides an estimate of the user's prospective medical expenses.
''')

# inputs
sex = st.selectbox("What is your gender?", ('male', 'female'))
age = st.slider("What is your age?", min_value=18, max_value=64)
bmi = st.slider("What is your BMI?", min_value=16.0, max_value=53.1, help='BMI stands for Body Mass Index. You can calculate your by using, BMI = weight/height^2')
children = st.selectbox("How many children do you have?", ('0', '1', '2', '3', '4', '5'))
smoker = st.selectbox("Do you have smoking tendencies?", ('yes', 'no'))
region = st.selectbox("What region do you belong?", ('northwest', 'northeast', 'southwest', 'southeast'))

# Column names and created DataFrame for inputs
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns=cols)

def preprocess_inputs(df):
    df = df.copy()

    # numerical columns
    num_cols = ['age', 'bmi', 'children']

    # feature scaling
    df[num_cols] = scaler.transform(df[num_cols])

    # categorical columns
    cat_cols = ['sex', 'smoker', 'region']

    # feature encoding
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # ensure input has the same columns as the training data
    df = df.reindex(columns=columns, fill_value=0)

    return df

def predict_premium(df):
    df = preprocess_inputs(df)
    prediction = model.predict(df)
    return prediction

if st.button('Predict'):
    result = predict_premium(data)
    st.write('---')
    st.success('Your Estimated Medical Expenses Bill is {} USD'.format(np.around(result[0], decimals=2)))
    st.write('---')

# remove main menu and footer note of Streamlit
hide_menu_style = '''
       <style>
       #MainMenu {visibility: hidden;}
       footer {visibility: hidden;}
       </style>
'''
st.markdown(hide_menu_style, unsafe_allow_html=True)

