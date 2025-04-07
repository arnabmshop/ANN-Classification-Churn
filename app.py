import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

#Load the trained model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scalers
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

#Streamlit app
st.title("Customer Churn Prediction")

#Provision for inputs
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Num of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

#Input Data
input_data = {'CreditScore':credit_score,'Geography':geography,'Gender':gender,'Age':age,'Tenure':tenure,'Balance':balance,
              'NumOfProducts':num_of_products,'HasCrCard':has_cr_card,'IsActiveMember':is_active_member,
              'EstimatedSalary':estimated_salary}

input_df = pd.DataFrame([input_data])

# Label encode Gender
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform(input_df[['Geography']]).toarray()
# Create DataFrame for encoded geography
geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
geo_df = pd.DataFrame(geo_encoded, columns=geo_cols)

# Drop original Geography and concatenate
input_df = input_df.drop('Geography', axis=1)
input_df = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)

# Scale the input
input_scaled = scaler.transform(input_df)

# Predict using the trained ANN model
prediction = model.predict(input_scaled)

# Display result in the Streamlit app
if prediction > 0.5:
    st.warning("⚠️ Prediction: Customer will churn")
else:
    st.success("✅ Prediction: Customer will not churn")

