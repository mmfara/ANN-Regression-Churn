import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model
import pandas as pd
import pickle

# Load the trained model
model = load_model('model.h5')

# Load the encoder and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title('Estimated Salary Prediction (Regression)')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0, 1])          # keep only if used as input in training
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore':    [credit_score],
    'Gender':         [label_encoder_gender.transform([gender])[0]],
    'Age':            [age],
    'Tenure':         [tenure],
    'Balance':        [balance],
    'NumOfProducts':  [num_of_products],
    'HasCrCard':      [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited':         [exited],   # remove if not used in training
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]])
if hasattr(geo_encoded, "toarray"):
    geo_encoded = geo_encoded.toarray()

geo_feature_names = onehot_encoder_geo.get_feature_names_out(['Geography'])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_feature_names)

# Combine one-hot encoded columns with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Optional: check column alignment
# st.write("Model input feature columns:", list(input_data.columns))

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict (regression)
prediction = model.predict(input_data_scaled)
predicted_salary = float(prediction[0][0])

st.write(f'Predicted Estimated Salary: ${predicted_salary:.2f}')
