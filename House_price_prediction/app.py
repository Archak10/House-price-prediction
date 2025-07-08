import streamlit as st
import numpy as np
import joblib

# Load model and features
model=joblib.load("House_price_prediction/house_price_model.pkl")
columns=joblib.load("House-price_prediction/features.pkl")

# Page UI
st.title("🏠 House Price Prediction App")
st.write("Enter the required house details to predict the price")

# Input fields
area = st.number_input("Area (in square feet)", min_value=500, max_value=10000, value=1500, step=100)

bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=10, value=2)

stories = st.selectbox("Number of Stories", options=[1, 2, 3, 4], index=1)

mainroad = st.selectbox("Is the house on the main road?", ["yes", "no"])
guestroom = st.selectbox("Guest Room Available?", ["yes", "no"])
basement = st.selectbox("Has Basement?", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating?", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning?", ["yes", "no"])
parking = st.slider("Number of Parking Spaces", min_value=0, max_value=5, value=1)

prefarea = st.selectbox("Is it in a Preferred Area?", ["yes", "no"])
furnishingstatus = st.selectbox("Furnishing Status", ["unfurnished", "semi-furnished", "furnished"])

# Create input dictionary based on expected feature columns

input_dict = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking,
    'mainroad': 1 if mainroad == 'yes' else 0,
    'guestroom': 1 if guestroom == 'yes' else 0,
    'basement': 1 if basement == 'yes' else 0,
    'hotwaterheating': 1 if hotwaterheating == 'yes' else 0,
    'airconditioning': 1 if airconditioning == 'yes' else 0,
    'prefarea': 1 if prefarea == 'yes' else 0,

        # One-hot encode furnishingstatus

    'furnishingstatus_semi-furnished': 1 if furnishingstatus == 'semi-furnished' else 0,
    'furnishingstatus_unfurnished': 1 if furnishingstatus == 'unfurnished' else 0,

           # furnished is baseline → both above = 0
}

# Build the final input vector in correct order

input_data = []
for feature in columns:
    input_data.append(input_dict.get(feature, 0))   # default 0 if not present

# Predict

if st.button("💰 Predict Price"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"🏡 Estimated House Price: ₹{prediction:,.2f}")
