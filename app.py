# =========================
# STREAMLIT APP
# =========================

import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD MODEL
# =========================

model = pickle.load(open("model.pkl", "rb"))

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title("🏠 House Price Prediction")

st.write("Enter house details below")

st.divider()

# =========================
# USER INPUTS
# =========================

sqft = st.number_input(
    "Enter Total Square Feet",
    min_value=500,
    max_value=10000,
    value=1000
)

bath = st.number_input(
    "Enter Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

bhk = st.number_input(
    "Enter Number of BHK",
    min_value=1,
    max_value=10,
    value=2
)

# =========================
# PREDICTION
# =========================

if st.button("Predict Price"):

    input_data = pd.DataFrame(
        [[sqft, bath, bhk]],
        columns=['total_sqft', 'bath', 'bhk']
    )

    prediction = model.predict(input_data)

    st.success(
        f"Estimated House Price: ₹ {prediction[0]:.2f} Lakhs"
    )