import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Airbnb Price Predictor", page_icon="🏙️", layout="centered")

@st.cache_resource
def load_model():
    # Build an absolute path next to THIS script — works regardless of CWD
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_pipeline.pkl")
    return joblib.load(model_path)

model = load_model()

st.title("🏙️ NYC Airbnb Nightly Price Predictor")
st.write("Enter listing details to estimate the nightly price (USD).")

col1, col2 = st.columns(2)

with col1:
    neighbourhood_group = st.selectbox(
        "Borough",
        ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    )
    room_type = st.selectbox(
        "Room Type",
        ["Entire home/apt", "Private room", "Shared room"]
    )
    latitude = st.number_input("Latitude", value=40.7500, format="%.4f")
    longitude = st.number_input("Longitude", value=-73.9800, format="%.4f")

with col2:
    availability_365 = st.slider("Availability (days/year)", 1, 365, 180)
    reviews_per_month = st.slider("Reviews per Month", 0.0, 20.0, 1.0, 0.1)
    minimum_nights = st.number_input("Minimum Nights", 1, 365, 3)
    number_of_reviews = st.number_input("Total Number of Reviews", 0, 1000, 10)
    host_listings = st.number_input("Host's Total Listings", 1, 200, 1)

neighbourhood = st.text_input("Neighbourhood (optional)", value="")

if st.button("Predict Price 💰"):

    # Recreate the host_type bucket exactly like in training
    if host_listings == 1:
        host_type = 'Single'
    elif host_listings <= 5:
        host_type = 'Small'
    elif host_listings <= 50:
        host_type = 'Medium'
    else:
        host_type = 'Commercial'

    input_df = pd.DataFrame([{
        'latitude': latitude,
        'longitude': longitude,
        'availability_365': availability_365,
        'reviews_per_month': reviews_per_month,
        'calculated_host_listings_count': host_listings,
        'minimum_nights': minimum_nights,
        'number_of_reviews': number_of_reviews,
        'neighbourhood_group': neighbourhood_group,
        'neighbourhood': neighbourhood if neighbourhood else neighbourhood_group,
        'room_type': room_type,
        'host_type': host_type
    }])

    pred = model.predict(input_df)[0]
    st.success(f"### Estimated Nightly Price: **${pred:.2f}**")
    st.caption("Estimate based on a Gradient Boosting model trained on 2019 NYC Airbnb data.")
