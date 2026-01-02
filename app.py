import os
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix

# ==============================
# Load trained artifact (model + encoder + metadata)
# ==============================
MODEL_PATH = "models/RandomForest_crop_model.pkl"
DATA_PATH = "Crop_recommendation.csv"  # needed for radar chart and fertilizer module

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please train and save Random Forest model first.")
    st.stop()

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
le = artifact["label_encoder"]
features = artifact["features"]

# Load dataset for radar chart and fertilizer recommendations
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = None

# ==============================
# Streamlit Layout
# ==============================
st.set_page_config(page_title="Smart Crop & Fertilizer Recommendation", page_icon="🌱", layout="wide")
st.title("🌱 Smart Crop & Fertilizer Recommendation System")
st.caption("Trained on Kaggle Crop Recommendation dataset (22 crops)")

# Sidebar: extra visualizations
st.sidebar.header("🔎 Extra Visualizations")
show_cm = st.sidebar.checkbox("Show Confusion Matrix (Test Data)")

# ==============================
# Input form
# ==============================
with st.form("crop_input_form"):
    col1, col2 = st.columns(2)
    with col1:
        N  = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50, step=1)
        P  = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50, step=1)
        K  = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50, step=1)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    with col2:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
        humidity    = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
        rainfall    = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=150.0, step=0.1)

    submitted = st.form_submit_button("🔮 Predict Crop")

st.sidebar.subheader("🌤️ Get Live Weather Data")
use_live_weather = st.sidebar.checkbox("Fetch live temperature & humidity")

city = st.sidebar.text_input("Enter City Name (for weather data)", "Chennai")

if use_live_weather:
    api_key = "65e300897cacbef7fd4bbdda70160cc5"
    base_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(base_url)
    if response.status_code == 200:
        weather = response.json()
        temperature = weather['main']['temp']
        humidity = weather['main']['humidity']
        st.sidebar.success(f"🌡️ Temperature: {temperature}°C | 💧 Humidity: {humidity}%")
    else:
        st.sidebar.error("Could not fetch weather data. Please check city name.")

# ==============================
# Prediction + Visualizations
# ==============================
if submitted:
    # Prepare input with feature names
    X_one = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=features)

    # Predict class index & crop name
    pred_idx = model.predict(X_one)[0]
    crop_name = le.inverse_transform([pred_idx])[0]

    st.success(f"🌾 Recommended Crop: **{crop_name}**")

    # Top-3 probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_one)[0]
        idx_sorted = np.argsort(proba)[::-1][:3]
        top3_crops = le.inverse_transform(idx_sorted)
        top3_probs = proba[idx_sorted]

        st.subheader("📊 Top-3 Crop Probabilities")
        for i in range(3):
            st.write(f"**{top3_crops[i]}:** {top3_probs[i]*100:.2f}%")

    # Display soil & climate input summary
    st.subheader("🌍 Your Soil & Climate Inputs")
    st.table(pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=features))

    # Radar chart: Input vs Average conditions
    if df is not None:
        st.subheader(f"🌐 Input vs Average Conditions for {crop_name}")
        crop_means = df[df['label'] == crop_name][features].mean()

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[N, P, K, temperature, humidity, ph, rainfall],
            theta=features,
            fill='toself',
            name="Your Input"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=crop_means.values,
            theta=features,
            fill='toself',
            name=f"Avg {crop_name}"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title=f"🌍 Comparison: You vs Avg {crop_name}"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ==============================
    # 🧪 Fertilizer Recommendation (Dynamic)
    # ==============================
    if df is not None:
        st.subheader("🧪 Fertilizer Recommendation")

        # Get ideal nutrient levels for predicted crop
        ideal = df[df['label'] == crop_name][['N','P','K']].mean()
        diff_N = round(ideal['N'] - N, 1)
        diff_P = round(ideal['P'] - P, 1)
        diff_K = round(ideal['K'] - K, 1)

        # Suggestion logic
        def fert_suggestion(n, p, k):
            recs = []
            if n > 0:
                recs.append(f"🟢 Add {abs(int(n))} kg/acre Nitrogen (Urea)")
            elif n < 0:
                recs.append(f"🔴 Reduce {abs(int(n))} kg/acre Nitrogen fertilizer")

            if p > 0:
                recs.append(f"🟢 Add {abs(int(p))} kg/acre Phosphorus (DAP)")
            elif p < 0:
                recs.append(f"🔴 Reduce {abs(int(p))} kg/acre Phosphorus fertilizer")

            if k > 0:
                recs.append(f"🟢 Add {abs(int(k))} kg/acre Potassium (MOP)")
            elif k < 0:
                recs.append(f"🔴 Reduce {abs(int(k))} kg/acre Potassium fertilizer")

            return recs

        # Generate recommendations
        fert_recs = fert_suggestion(diff_N, diff_P, diff_K)
        for rec in fert_recs:
            st.write(rec)

        # Bar Chart (Matplotlib)
        comp_df = pd.DataFrame({
            "Nutrient": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"],
            "Your Soil": [N, P, K],
            f"Ideal for {crop_name}": [ideal['N'], ideal['P'], ideal['K']]
        })

        comp_melt = comp_df.melt(id_vars="Nutrient", var_name="Source", value_name="Value")
        fig, ax = plt.subplots(figsize=(7,5))
        sns.barplot(data=comp_melt, x="Nutrient", y="Value", hue="Source", palette="Set2", ax=ax)
        ax.set_title(f"Soil vs Ideal N-P-K Levels for {crop_name}", fontsize=13)
        ax.set_ylabel("Nutrient Level")
        ax.set_xlabel("Nutrient Type")
        st.pyplot(fig)

# ==============================
# Feature Importance (Random Forest)
# ==============================
if hasattr(model, "feature_importances_"):
    st.subheader("📈 Feature Importance (Random Forest)")
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=True)

    fig2, ax2 = plt.subplots(figsize=(4,2))
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis", ax=ax2)
    ax2.set_title("Feature Importance")
    st.pyplot(fig2)

# ==============================
# Confusion Matrix (Sidebar Option)
# ==============================
if show_cm and df is not None:
    st.subheader("📊 Confusion Matrix (Test Data)")

    from sklearn.model_selection import train_test_split
    X = df[features]
    y = le.transform(df["label"])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=list(le.classes_),
        y=list(le.classes_),
        colorscale="Blues",
        showscale=True
    )
    fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig_cm, use_container_width=True)
