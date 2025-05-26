# HEA Phase Prediction Model

A machine learning web application for predicting the phase of High Entropy Alloys (HEAs) using physicochemical features and an XGBoost classifier.  
This project provides a user-friendly Streamlit interface for real-time phase prediction.

---

## 🚀 Demo

Try the deployed app: [heaphasepredictionmodel-bzoanj8gzm7fvzaeskvjjv.streamlit.app](https://heaphasepredictionmodel-bzoanj8gzm7fvzaeskvjjv.streamlit.app/)

---

## 📂 Repository Structure
.
├── app.py                    # Streamlit web app for phase prediction
├── hea_phase_predictor.pkl  # Trained XGBoost model
├── hea_scaler.pkl           # Feature scaler (StandardScaler)
├── label_encoder.pkl        # Label encoder for phase names
├── requirements.txt         # Python dependencies
├── HEA Phase DataSet v1d.csv# (Optional) Dataset for reference
└── README.md                # Project documentation

---

## 🧑‍💻 Getting Started

### 1. Clone the Repository

# Clone the repository
git clone https://github.com/Itzraj786iul/HEA_Phase_Prediction_Model.git

# Navigate into the project directory
cd HEA_Phase_Prediction_Model


### 2. Install Dependencies
pip install -r requirements.txt


pip install -r requirements.txt
streamlit run app.py


The app will open in your browser. Enter alloy features to predict the phase.

---

## 🧬 Features Used

- Mixing enthalpy (`dHmix`)
- Mixing entropy (`dSmix`)
- Valence Electron Concentration (`VEC`)
- Atomic size difference (`Atom.Size.Diff`)
- Average atomic radius (`Avg_Atomic_Radius`)
- Atomic size mismatch (`Delta`)
- Configurational entropy (`Mixing_Entropy`)
- Calculated VEC (`True_VEC`)
- Stability parameter (`Stability_Param`)
- Number of elements (`Num_Elements`)

---

## 📊 Model

- **Algorithm:** XGBoost Classifier
- **Preprocessing:** StandardScaler, LabelEncoder
- **Output:** Predicted phase (e.g., BCC_SS, FCC_SS, etc.) and top-3 class probabilities

---

## 📝 Results

- The model achieves high accuracy on test data.
- The web app displays predicted phase and confidence.

---

## 📄 License

[MIT License](LICENSE)  

---
  

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/), and [XGBoost](https://xgboost.ai/)
- Raziullah Ansari

---

**For questions or contributions, please open an issue or pull request.**
