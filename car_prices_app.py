#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.write("""
# Car Value Prediction App

This app predicts the value of a used car based on 2021 May Alabama Car data
""")

st.sidebar.header('User Input Features')

# collects user input features into dataframe
def car_details():
    age = st.sidebar.number_input('Age of vehicle is:', min_value=0, max_value=30)
    odometer = st.sidebar.number_input('mileage on the vehicle is:', min_value=1000, max_value=5000000)
    condition = st.sidebar.selectbox('condition',('good', 'like new', 'fair'))
    cylinders = st.sidebar.selectbox('cylinders',('3 cylinders', '4 cylinders', '5 cylinders', '6 cylinders', '8 cylinders', '10 cylinders', '12 cylinders'))
    fuel = st.sidebar.selectbox('fuel',('gas', 'hybrid', 'diesel', 'electric'))
    transmission = st.sidebar.selectbox('transmission',('automatic', 'manual'))
    drive = st.sidebar.selectbox('drive',('rwd', '4wd', 'fwd'))
    size = st.sidebar.selectbox('size',('full-size', 'mid-size', 'sub-compact', 'compact'))
    ctype = st.sidebar.selectbox('type',('bus', 'convertible', 'coupe', 'hatchback', 'mini-van', 'offroad', 'pickup', 'sedan', 'SUV', 'truck', 'van', 'wagon'))
    data = {'age': age,
            'odometer' : odometer,
            'condition' : condition,
            'cylinders' : cylinders,
            'fuel' : fuel,
            'transmission' : transmission,
            'drive' : drive,
            'size' : size,
            'type' : ctype }
    features = pd.DataFrame(data, index=[0])
    return features
df = car_details()


#Encoding categorical features
encode = ['condition', 'cylinders', 'fuel', 'transmission', 'drive', 'size', 'type']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
    df = df[:1] #selects only the first row(user input data)


#Display user input features
st.subheader('User Input Features')
st.write(df)

#Read saved prediction model
load_clf = pickle.load(open('cars_model.pkl', 'rb')) #import model
sc = pickle.load(open('scaler.pkl', 'rb')) #import scaler

# Ensure the order of encoding columns is the same as during training

cols = ['age', 'odometer', 'condition_fair', 'condition_good', 'condition_like new', 'cylinders_10 cylinders', 'cylinders_12 cylinders', 'cylinders_3 cylinders', 'cylinders_4 cylinders', 'cylinders_5 cylinders', 'cylinders_6 cylinders', 'cylinders_8 cylinders', 'fuel_diesel', 'fuel_electric', 'fuel_gas', 'fuel_hybrid', 'transmission_automatic', 'transmission_manual', 'drive_4wd', 'drive_fwd', 'drive_rwd', 'size_compact', 'size_full-size', 'size_mid-size', 'size_sub-compact', 'type_SUV', 'type_bus', 'type_convertible', 'type_coupe', 'type_hatchback', 'type_mini-van', 'type_offroad', 'type_pickup', 'type_sedan', 'type_truck', 'type_van', 'type_wagon']
df = df.reindex(columns=cols, fill_value=False)
df = sc.transform(df)


#Apply model to make predictions
prediction = load_clf.predict(df)

st.subheader('Prediction')
price = np.round(np.exp(prediction), 0)
st.write('Your car value in USD is approximately:', price)

