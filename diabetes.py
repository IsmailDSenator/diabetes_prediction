import numpy as np
import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings 
warnings.filterwarnings('ignore')
data = pd.read_csv('Diabetes_Prediction (Project).csv')
df = data.copy()



st.markdown("<h1 style = 'color:#0802A3; text-align: center; font-family: Arial Black; font-size:42px'>DIABETES PREDICTION</h1>", unsafe_allow_html=True)

st.markdown("<h4 style = 'margin: -30px; color: #000000; text-align: center; font-family: cursive;font-size:32px'>Built By Ismail Ibitoye</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html = True)
st.image('pngwing.com (1).png', width = 350, use_column_width = True)
st.markdown("<br>", unsafe_allow_html = True)
st.markdown("<p style=font-family:Comic Sans>Diabetes prediction involves analyzing various factors such as BMI, age, and other relevant health indicators to assess the likelihood of developing diabetes. By examining these columns in a dataset, predictive models can identify patterns and correlations indicative of diabetes risk. Features like BMI, age, blood glucose levels, and family medical history are crucial predictors utilized in model development. Leveraging machine learning algorithms, such as logistic regression or decision trees, enables the creation of predictive models capable of accurately forecasting diabetes risk. The incorporation of additional factors, such as lifestyle choices and dietary habits, enhances the predictive power of the model, aiding in early detection and intervention strategies.Ultimately, the goal of diabetes prediction is to notify individual of his or her health status.</p>", 
             unsafe_allow_html = True)
st.markdown('<br>', unsafe_allow_html = True)
st.dataframe(data, use_container_width = True)  

st.sidebar.image('pngwing.com (2).png', caption = 'welcome user')



blood_pressure = st.sidebar.number_input('Blood_pressure_level', data['BloodPressure'].min(), data['BloodPressure'].max())
skin_thickness = st.sidebar.number_input('Skin_thickness', data['SkinThickness'].min(), data['SkinThickness'].max())
Insulin = st.sidebar.number_input('Insulin_level', data['Insulin'].min(), data['Insulin'].max())
bmi = st.sidebar.number_input('BMI', data['BMI'].min(), data['BMI'].max())
age = st.sidebar.number_input('Age', data['Age'].min(), data['Age'].max())
Diabetic_functionality = st.sidebar.number_input('Diabetic_Functionality', data['DiabetesPedigreeFunction'].min(), data['DiabetesPedigreeFunction'].max())
glucose = st.sidebar.number_input('Glucose_level', data['Glucose'].min(), data['Glucose'].max())


input_var = pd.DataFrame({ 'BloodPressure':[blood_pressure], 
                          'SkinThickness':[skin_thickness], 'Insulin':[Insulin], 'BMI':[bmi], 
                           'Age': [age], 'DiabetesPedigreeFunction' :[Diabetic_functionality], 'Glucose' : glucose})
st.dataframe(input_var)
model = joblib.load('diabetes.pkl')
prediction = st.button('Press to predict')

if prediction:
    predicted = model.predict(input_var)
    output = None
    if predicted == 1:
        output = 'Diabetic'
    else:
        output = 'Non-Diabetic'
    st.success(f'The result of this analysis shows that this individual is {output}')
    st.balloons()