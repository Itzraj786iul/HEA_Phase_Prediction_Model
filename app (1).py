import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import joblib




# Load model and preprocessing objects
@st.cache_resource
def load_model():
    model = joblib.load("hea_phase_predictor.pkl")
    scaler = joblib.load("hea_scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, scaler, le

model, scaler, le = load_model()

# Feature information dictionary
feature_info = {
    'dHmix': {'desc': 'Mixing enthalpy (kJ/mol). Typical range: -20 to 5', 'range': (-20, 5)},
    'dSmix': {'desc': 'Mixing entropy (J/mol·K). Typical range: 10 to 16', 'range': (10, 16)},
    'VEC': {'desc': 'Valence Electron Concentration. Typical range: 4.5 to 8.5', 'range': (4.5, 8.5)},
    'Atom.Size.Diff': {'desc': 'Atomic size difference (%). Typical range: 0 to 7', 'range': (0, 7)},
    'Avg_Atomic_Radius': {'desc': 'Average atomic radius (Å). Typical range: 1.2 to 1.6', 'range': (1.2, 1.6)},
    'Delta': {'desc': 'Atomic size mismatch (dimensionless). Typical range: 0.01 to 0.08', 'range': (0.01, 0.08)},
    'Mixing_Entropy': {'desc': 'Configurational entropy (J/mol·K). Typical range: 1.3 to 1.7', 'range': (1.3, 1.7)},
    'True_VEC': {'desc': 'Calculated VEC (dimensionless). Typical range: 4.5 to 8.5', 'range': (4.5, 8.5)},
    'Stability_Param': {'desc': 'Stability parameter (dHmix/Tm). Typical range: -0.02 to 0.02', 'range': (-0.02, 0.02)},
    'Num_Elements': {'desc': 'Number of elements in alloy. Typical range: 4 to 7', 'range': (4, 7)}
}

st.title("HEA Phase Predictor")
st.markdown("Enter your alloy features below. All values should be numerical.")

with st.form("input_form"):
    st.caption("ℹ️ Hover over input fields to see feature descriptions and valid ranges.")

    dHmix = st.number_input("Mixing enthalpy (dHmix in kJ/mol)", value=0.0,
                            help=feature_info['dHmix']['desc'])
    dSmix = st.number_input("Mixing entropy (dSmix in J/mol·K)", value=12.0,
                            help=feature_info['dSmix']['desc'])
    VEC = st.number_input("Valence Electron Concentration (VEC)", value=7.0,
                          help=feature_info['VEC']['desc'])
    Atom_Size_Diff = st.number_input("Atomic size difference (%)", value=5.0,
                                     help=feature_info['Atom.Size.Diff']['desc'])
    Avg_Atomic_Radius = st.number_input("Average atomic radius (Å)", value=1.3,
                                        help=feature_info['Avg_Atomic_Radius']['desc'])
    Delta = st.number_input("Atomic size mismatch (Δ)", value=0.05,
                            help=feature_info['Delta']['desc'])
    Mixing_Entropy = st.number_input("Mixing entropy (S_mix in J/K·mol)", value=1.5,
                                     help=feature_info['Mixing_Entropy']['desc'])
    True_VEC = st.number_input("True VEC", value=7.0,
                               help=feature_info['True_VEC']['desc'])
    Stability_Param = st.number_input("Stability parameter (dHmix/Tm)", value=-0.003,
                                      help=feature_info['Stability_Param']['desc'])
    Num_Elements = st.select_slider("Number of elements in alloy", options=[4, 5, 6, 7],
                                    value=5, help=feature_info['Num_Elements']['desc'])

    submit = st.form_submit_button("Predict Phase")

if submit:
    # Input validation
    input_dict = {
        'dHmix': dHmix,
        'dSmix': dSmix,
        'VEC': VEC,
        'Atom.Size.Diff': Atom_Size_Diff,
        'Avg_Atomic_Radius': Avg_Atomic_Radius,
        'Delta': Delta,
        'Mixing_Entropy': Mixing_Entropy,
        'True_VEC': True_VEC,
        'Stability_Param': Stability_Param,
        'Num_Elements': Num_Elements
    }
    valid = True
    for feat, value in input_dict.items():
        min_val, max_val = feature_info[feat]['range']
        if not (min_val <= value <= max_val):
            st.error(f"❗ {feat} must be between {min_val} and {max_val}. You entered {value}.")
            valid = False

    if valid:
        features = pd.DataFrame([input_dict.values()], columns=input_dict.keys())
        try:
            scaled = scaler.transform(features)
            pred = model.predict(scaled)
            phase = le.inverse_transform(pred)[0]
            proba = model.predict_proba(scaled)[0]
            confidence = np.max(proba) * 100

            # Top 3 predictions
            top3_idx = np.argsort(proba)[::-1][:3]
            st.success(f"Predicted Phase: **{phase}** (Confidence: {confidence:.1f}%)")
            st.markdown("#### Top 3 Phase Probabilities:")
            for idx in top3_idx:
                st.write(f"- {le.inverse_transform([idx])[0]}: {proba[idx]*100:.1f}%")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
