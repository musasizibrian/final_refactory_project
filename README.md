# 🚢 Titanic Survival Predictor

This Streamlit app predicts whether a passenger survived the Titanic disaster based on demographic and travel details.

## 🧠 Models Used
- **AdaBoost Classifier** (`best_adaboost.pkl`)
- **Feedforward Neural Network (FNN)** (`best_fnn_model.keras`)

## 🧩 Features
- Simple interactive web app built with Streamlit
- Supports multiple ML models
- Real-time prediction with visual feedback

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py



Here’s an enhanced, more detailed version of your **README.md** — polished, well-structured, and ready for GitHub or documentation 👇

---

```markdown
# 🚢 Titanic Survival Predictor

A **Streamlit-based Machine Learning web app** that predicts whether a passenger would have survived the **Titanic disaster** based on their demographic, socio-economic, and travel details.

---

## 🧠 Machine Learning Models

This project integrates and compares multiple trained models for prediction:

| Model | Description | File |
|--------|--------------|------|
| **AdaBoost Classifier** | Ensemble model that combines weak learners (decision trees) for strong predictive performance. | `best_adaboost.pkl` |
| **Feedforward Neural Network (FNN)** | Deep learning model built using TensorFlow/Keras for complex pattern recognition. | `best_fnn_model.keras` |

---

## 🧩 Features

- 🖥️ **Interactive Web Interface** — Built with **Streamlit** for easy input and instant feedback.  
- 🔍 **Multiple Model Support** — Choose between classical ML and neural network models.  
- 📊 **Real-time Prediction** — See instant survival probability and classification.  
- 🌈 **Clean, Responsive UI** — Simple yet elegant design optimized for all devices.  
- 💾 **Model Persistence** — Pre-trained models saved and loaded efficiently.  
- 🧮 **Feature Inputs** — Age, Gender, Passenger Class, Fare, Family Size, Embarked Port, etc.

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend / ML** | Python, scikit-learn, TensorFlow / Keras |
| **Visualization** | Matplotlib, Seaborn |
| **Data Handling** | Pandas, NumPy |
| **Model Persistence** | joblib, h5 (Keras format) |

---

## 📁 Project Structure

```

titanic-survival-predictor/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
├── models/                 # Saved trained models
│   ├── best_adaboost.pkl
│   └── best_fnn_model.keras
├── data/                   # (Optional) Dataset folder
│   └── titanic.csv
└── assets/                 # Images, icons, etc.

````

---

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/titanic-survival-predictor.git
   cd titanic-survival-predictor
````

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   venv\Scripts\activate     # On Windows
   source venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

5. **Open the app**
   The app will open automatically in your browser (usually at [http://localhost:8501](http://localhost:8501)).

---

## 🧪 Example Prediction

| Feature         | Example Value   |
| --------------- | --------------- |
| Passenger Class | 2               |
| Gender          | Female          |
| Age             | 28              |
| Fare            | 12.50           |
| Embarked        | S (Southampton) |
| Family Size     | 1               |

💡 **Prediction Output:**
🟢 *Survived (Probability: 87%)*

---

## 🧭 Future Improvements

* ✅ Add feature importance visualization
* ✅ Display model performance metrics (Accuracy, F1-score)
* ⚙️ Integrate dataset upload for custom testing
* 📈 Add dashboard with insights and graphs
* 🌐 Deploy to Streamlit Cloud or Hugging Face Spaces

---

## 👨‍💻 Author

**Brian Musasizi**
Kabale University • Department of Computer Science
📧 musasizibrian759@gmail.com
🌐 https://github.com/musasizibrian

---

## 🪪 License

This project is released under the **MIT License** — free to use, modify, and distribute with attribution.

---

✨ *"Predicting survival — where data meets destiny."*

```

---

Would you like me to **add a section for the dataset details** (e.g., features, preprocessing steps, missing value handling, encoding, scaling)? It would make your README look more professional and research-friendly.
```
