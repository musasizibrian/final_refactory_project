import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ------------------------------
# 1️⃣ Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.markdown("Choose a model and enter passenger details to predict survival chances.")

# ------------------------------
# 2️⃣ Model Selector
# ------------------------------
model_choice = st.selectbox(
    "🔽 Select Model",
    ["AdaBoost", "Feedforward Neural Network (FNN)"]
)

# Load models from 'models/' folder
if model_choice == "AdaBoost":
    model = joblib.load("models/best_adaboost.pkl")
else:
    model = load_model("models/best_fnn_model.keras")

# ------------------------------
# 3️⃣ Input Form
# ------------------------------
with st.form("prediction_form"):
    pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 30)
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.slider("Fare Paid (£)", 0.0, 512.0, 32.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
    
    submitted = st.form_submit_button("Predict Survival")

# ------------------------------
# 4️⃣ Preprocess Input + Prediction
# ------------------------------
if submitted:
    # Encode categorical inputs
    sex_encoded = 1 if sex == "female" else 0
    embarked_map = {"S": 0, "C": 1, "Q": 2}
    embarked_encoded = embarked_map[embarked]

    # Feature vector
    features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

    # Make prediction
    if model_choice == "AdaBoost":
        prob = model.predict_proba(features)[:, 1][0]
    else:
        prob = model.predict(features)[0][0]

    prediction = "✅ Survived" if prob >= 0.5 else "❌ Did Not Survive"

    # ------------------------------
    # 5️⃣ Display Results
    # ------------------------------
    st.subheader("🔎 Prediction Result")
    st.write(f"**{prediction}** (Probability: {prob:.2f})")
    st.progress(float(prob))

    if prob >= 0.5:
        st.success("Passenger likely to **Survive** 🚑")
    else:
        st.error("Passenger likely to **Not Survive** ⚠️")
