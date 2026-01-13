import streamlit as st
import requests
import pandas as pd
import numpy as np
import warnings

from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor
)
from sklearn.metrics import mean_absolute_error

import folium
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ML Restaurant Recommender",
    page_icon="🍽️",
    layout="wide"
)

# --------------------------------------------------
# Utilities
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def get_coordinates(pincode):
    geolocator = Nominatim(user_agent="restaurant_streamlit_app")
    try:
        loc = geolocator.geocode(f"{pincode}, India", timeout=10)
        if loc:
            return (loc.latitude, loc.longitude), loc.address
    except:
        pass
    return None, None


@st.cache_data(show_spinner=False)
def fetch_restaurants(search_coords, user_coords, radius_km):
    overpass_url = "http://overpass-api.de/api/interpreter"

    query = f"""
    [out:json][timeout:35];
    (
      node["amenity"~"restaurant|cafe|fast_food|food_court"](around:{radius_km*1000},{search_coords[0]},{search_coords[1]});
      way["amenity"~"restaurant|cafe|fast_food|food_court"](around:{radius_km*1000},{search_coords[0]},{search_coords[1]});
    );
    out center;
    """

    try:
        response = requests.get(overpass_url, params={"data": query}, timeout=40)
        data = response.json()
    except:
        return pd.DataFrame()

    rows = []
    for el in data.get("elements", []):
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        tags = el.get("tags", {})

        if not lat or not lon:
            continue

        dist = geodesic(user_coords, (lat, lon)).km

        rows.append({
            "name": tags.get("name", "Local Eatery"),
            "category": tags.get("amenity", "food"),
            "lat": lat,
            "lon": lon,
            "distance_km": round(dist, 2),
            "review_count": np.random.randint(20, 500),
            "rating": np.random.uniform(3.5, 5.0)
        })

    return pd.DataFrame(rows)


def train_best_model(df):
    X = df[["distance_km", "review_count"]]
    y = df["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "KNN": KNeighborsRegressor(n_neighbors=3),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
        "Gradient Boost": GradientBoostingRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=150, random_state=42)
    }

    best_model, best_name, best_mae = None, "", float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name

    return best_model, best_name


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🍽️ ML-Powered Restaurant Recommendation System")
st.caption("Geolocation + Machine Learning + OpenStreetMap")

with st.sidebar:
    st.header("📍 Location Settings")
    pincode = st.text_input("Enter your Pincode", "")
    radius = st.slider("Search Radius (km)", 1, 20, 5)

    search_btn = st.button("🔍 Find Restaurants")

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if search_btn:
    if not pincode:
        st.warning("Please enter a pincode")
        st.stop()

    user_coords, user_addr = get_coordinates(pincode)

    if not user_coords:
        st.error("Invalid pincode or location not found")
        st.stop()

    st.success(f"Location: {user_addr}")

    with st.spinner("Fetching nearby restaurants..."):
        df = fetch_restaurants(user_coords, user_coords, radius)

    if df.empty:
        st.warning("No restaurants found. Increase radius.")
        st.stop()

    st.info(f"Found {len(df)} restaurants")

    # ML
    if len(df) >= 6:
        model, model_name = train_best_model(df)
        df["predicted_rating"] = model.predict(df[["distance_km", "review_count"]])
        st.success(f"Best ML Model: {model_name}")
    else:
        df["predicted_rating"] = df["rating"]
        st.info("Insufficient data — using base ratings")

    # Final ranking
    df["final_score"] = (
        0.75 * df["predicted_rating"] -
        0.25 * df["distance_km"]
    )

    top = df.sort_values("final_score", ascending=False).head(5)

    # --------------------------------------------------
    # Map
    # --------------------------------------------------
    st.subheader("🗺️ Restaurant Map")

    m = folium.Map(location=user_coords, zoom_start=14)

    folium.Marker(
        user_coords,
        tooltip="You are here",
        icon=folium.Icon(color="blue", icon="user")
    ).add_to(m)

    for _, row in top.iterrows():
        folium.Marker(
            [row["lat"], row["lon"]],
            tooltip=row["name"],
            popup=f"""
            <b>{row['name']}</b><br>
            ⭐ {row['predicted_rating']:.2f}<br>
            📍 {row['distance_km']} km
            """,
            icon=folium.Icon(color="red", icon="cutlery")
        ).add_to(m)

    st_folium(m, width=900, height=500)

    # --------------------------------------------------
    # Table
    # --------------------------------------------------
    st.subheader("🏆 Top Recommendations")

    display_df = top[[
        "name", "category", "predicted_rating", "distance_km"
    ]].rename(columns={
        "name": "Restaurant",
        "category": "Category",
        "predicted_rating": "ML Rating",
        "distance_km": "Distance (km)"
    })

    st.dataframe(display_df, use_container_width=True)

