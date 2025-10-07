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

# ------------------------------
# 3️⃣ Load Models
# ------------------------------
try:
    if model_choice == "AdaBoost":
        model = joblib.load("models/best_adaboost_model.pkl")
    else:
        model = load_model("models/best_fnn_model.h5")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# ------------------------------
# 4️⃣ Input Form
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
# 5️⃣ Preprocess Input & Make Prediction
# ------------------------------
if submitted:
    try:
        # Encode categorical inputs
        sex_encoded = 1 if sex == "female" else 0
        embarked_map = {"S": 0, "C": 1, "Q": 2}
        embarked_encoded = embarked_map[embarked]

        # Engineered features
        family_size = sibsp + parch
        is_alone = 1 if family_size == 0 else 0
        deck = 8          # default / most common deck in training
        age_group = 0     # default age group
        title = 2         # default title

        # Combine all 12 features
        features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded,
                              family_size, deck, is_alone, age_group, title]])

        # Make prediction
        if model_choice == "AdaBoost":
            prob = model.predict_proba(features)[:, 1][0]   # AdaBoost only uses first 7 features
        else:
            prob = model.predict(features)[0][0]

        prediction = "✅ Survived" if prob >= 0.5 else "❌ Did Not Survive"

        # ------------------------------
        # 6️⃣ Display Results
        # ------------------------------
        st.subheader("🔎 Prediction Result")
        st.write(f"**{prediction}** (Probability: {prob:.2f})")
        st.progress(float(prob))

        if prob >= 0.5:
            st.success("Passenger likely to **Survive** 🚑")
        else:
            st.error("Passenger likely to **Not Survive** ⚠️")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
