import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained models and encoders
model = pickle.load(open('modelhc.pkl', 'rb'))
scaler = pickle.load(open('scalerhc.pkl', 'rb'))
ohe = pickle.load(open('ohehc.pkl', 'rb'))

# Function to preprocess user input
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    # One-hot encode categorical features
    cat_features = ['Gender', 'Occupation', 'BMI Category']
    df_ohe = ohe.transform(df[cat_features])
    df.drop(columns=cat_features, inplace=True)
    df_ohe = pd.DataFrame(df_ohe, columns=ohe.get_feature_names_out(cat_features))
    df = pd.concat([df, df_ohe], axis=1)
    # Scale numerical features
    df_scaled = scaler.transform(df)
    return df_scaled

# Streamlit app
st.title('Sleep Disorder Prediction')

# Collect user input
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=1, max_value=100, value=30)
occupation = st.selectbox('Occupation', ['Lawyer', 'Salesperson', 'Nurse', 'Engineer', 'Accountant', 'Teacher', 'Doctor'])
sleep_duration = st.number_input('Sleep Duration (hours)', min_value=0.0, max_value=24.0, value=7.0)
quality_of_sleep = st.slider('Quality of Sleep', min_value=1, max_value=10, value=5)
physical_activity_level = st.number_input('Physical Activity Level', min_value=0, max_value=100, value=50)
stress_level = st.slider('Stress Level', min_value=1, max_value=10, value=5)
bmi_category = st.selectbox('BMI Category', ['Normal', 'Overweight'])
systolic_bp = st.number_input('Systolic Blood Pressure', min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=50, max_value=150, value=80)
heart_rate = st.number_input('Heart Rate', min_value=40, max_value=200, value=70)
daily_steps = st.number_input('Daily Steps', min_value=0, max_value=30000, value=8000)

# Prepare input data
input_data = {
    'Age': age,
    'Sleep Duration': sleep_duration,
    'Quality of Sleep': quality_of_sleep,
    'Physical Activity Level': physical_activity_level,
    'Stress Level': stress_level,
    'Systolic BP': systolic_bp,
    'Diastolic BP': diastolic_bp,
    'Heart Rate': heart_rate,
    'Daily Steps': daily_steps,
    'Gender': gender,
    'Occupation': occupation,
    'BMI Category': bmi_category
}

# Predict button
if st.button('Predict'):
    input_data_processed = preprocess_input(input_data)
    prediction = model.predict(input_data_processed)
    label_map = {0: 'Insomnia', 1: 'No Sleep Disorder', 2: 'Sleep Apnea'}
    prediction_label = label_map[prediction[0]]
    st.write(f'Predicted Sleep Disorder: **{prediction_label}**')

# Run the app
if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.write('Interactive web interface for Sleep Disorder Prediction Model')
