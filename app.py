import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## Streamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])       
gender = st.selectbox("Gender", label_encoder_gender.classes_)                 
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_number = st.selectbox("Is Active Member", [0, 1])

# Example input data
input_data = pd.DataFrame({
    'CreditScore': [int(credit_score)],       
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [int(age)],                          
    'Tenure': [int(tenure)],                    
    'Balance': [float(balance)],                
    'NumOfProducts': [int(num_of_products)],    
    'HasCrCard': [int(has_cr_card)],            
    'IsActiveMember': [int(is_active_number)],  
    'EstimatedSalary': [float(estimated_salary)] 
})

# One-hot encode 'Geography'
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()              
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

## Concatenation one hot encoded
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Scaling the input_data
input_data_scaled = scaler.transform(input_data)

## Predict Churn
prediction = model.predict(input_data_scaled)
prediction_probs = prediction[0][0]

## The Decision Set Through the Threshold
if prediction_probs > 0.5:
    st.write(f"The customer is likely to churn (probability: {prediction_probs:.2f})")
else:
    st.write(f"The customer is not likely to churn (probability: {prediction_probs:.2f})")