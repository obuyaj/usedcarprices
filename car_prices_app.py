#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write('if using mobile device, tap on top left arrow to input values')

st.write("""
# Car Value Prediction App

This app predicts the value of a used car based on 2021 May Alabama Car data
""")

st.sidebar.header('User Input Features')

# collects user input features into dataframe
def car_details():
    age = st.sidebar.number_input('Age of vehicle in years is:', min_value=0, max_value=30)
    odometer = st.sidebar.number_input('mileage on the vehicle is:', min_value=1000, max_value=5000000)
    condition = st.sidebar.selectbox('condition of the vehigle is',('good', 'like new', 'fair'))
    cylinders = st.sidebar.selectbox('vehicle engine cylinders',('3 cylinders', '4 cylinders', '5 cylinders', '6 cylinders', '8 cylinders', '10 cylinders', '12 cylinders'))
    fuel = st.sidebar.selectbox('fuel used',('gas', 'hybrid', 'diesel', 'electric'))
    transmission = st.sidebar.selectbox('engine transmission type',('automatic', 'manual'))
    drive = st.sidebar.selectbox('vehicle drive type',('rwd', '4wd', 'fwd'))
    size = st.sidebar.selectbox('size of vehicle',('full-size', 'mid-size', 'sub-compact', 'compact'))
    ctype = st.sidebar.selectbox('type of vehicle',('bus', 'convertible', 'coupe', 'hatchback', 'mini-van', 'offroad', 'pickup', 'sedan', 'SUV', 'truck', 'van', 'wagon'))
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

price = np.format(np.round(np.exp(prediction), 0), f'{number:,}')

st.write('The predicted car value is approximately ', **price**)


st.divider()
st.write("""
#### About the features
Value — Predicted value of the car in US dollar based on May 2021 prices.
\nAge — The age of the car since it was manufactured in years
\nCondition — The condition of the car; good, fair, like new
\nCylinders — The number of cylinders in the car engine ranging from 3 to 12. 
\nFuel — The type of fuel the car uses: ‘diesel’, ‘gas’, ‘electric’, and ‘hybrid’.
\nOdometer — This is the distance in miles that the car has traveled since it was manufactured.
\nTransmission – Transmission type of the vehicle: ‘automatic’ or ‘manual’
\nDrive — There are 3 types of drive transmissions; ‘4WD, ‘FWD’ and ‘RWD’. (Four wheel drive, forward wheel drive and rear wheel drive.)
\nSize – Details on the size of the vehicle: 'full-size', 'compact', 'mid-size', 'sub-compact'
\nType — This feature identifies if a vehicle is a SUV or a mini-van. There 13 unique values in this feature.
""")

st.divider()
st.write("""
\nVehicle sizes glosary:
\nSubcompact — Have 85 to 89 cubit feet between their passenger and cargo areas, and they are 157 to 165 inches long. Some fit up to four passengers, and some fit up to five. Are normally cheapest to buy and maintain.
\nCompact — Is also called a small car, but bigger than subcompact cars. Compacts have between 100 and 109 cubic feet of interior space. 
\nMidsize — vehicles are generally the most popular because they aren’t too big or too small. Cars in this category have between 110 and 120 cubic feet of combined passenger and cargo space, and hatchbacks have between 130 and 159 cubic feet. They have more head and leg room than compact vehicles. 
\nFull-Size — The biggest kind of car is a full-size, which is also known as large. To earn this classification, a car just has to have more than 120 cubic feet of interior space. Station Wagons have over 160 cubic feet. 
""")
st.divider()
st.write("Created by [Joshua Obuya](https://www.linkedin.com/in/joshua-obuya-80849956/)") 
