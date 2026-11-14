
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="BreastCare AI | Detection System",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #f4f7fb;
}

[data-testid="stSidebar"] {
    background: #10243a;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #10243a;
    margin-bottom: 4px;
}

.subtitle {
    color: #64748b;
    font-size: 15px;
}

.hero {
    background: linear-gradient(120deg, #10243a, #176b75);
    padding: 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 25px;
}

.hero h2 {
    color: white;
    font-size: 27px;
    margin-bottom: 8px;
}

.hero p {
    color: #d8f1f0;
    font-size: 15px;
}

.section-title {
    color: #10243a;
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.result-value {
    font-size: 30px;
    font-weight: 800;
}

.small-note {
    color: #64748b;
    font-size: 13px;
}

.stButton > button {
    background: #087e83;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 12px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: #065e63;
    color: white;
}

div[data-testid="stFormSubmitButton"] button {
    background: #087e83;
    color: white;
    border-radius: 9px;
    font-weight: 700;
    min-height: 48px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "best_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    artifact = load_model()
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    model_name = artifact["model_name"]

except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.stop()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎗️ BreastCare AI")
    st.caption("Machine Learning Project")

    st.divider()

    st.markdown("### Project Information")
    st.write("**Domain:** Healthcare AI")
    st.write("**Dataset:** Wisconsin Breast Cancer")
    st.write("**Algorithm:** " + str(model_name))

    st.divider()

    st.markdown("### Navigation")
    st.write("🏠 Dashboard")
    st.write("🧪 Prediction")
    st.write("📊 Model Information")

    st.divider()

    st.caption("CODE-A-NOVA Internship Project")
    st.caption("Educational demonstration only")


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">Breast Cancer Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered classification using Machine Learning</div>',
    unsafe_allow_html=True
)

st.write("")

st.markdown("""
<div class="hero">
    <h2>Welcome to BreastCare AI 🎗️</h2>
    <p>
        Explore a machine learning demonstration that classifies
        breast cancer measurement patterns into malignant or benign
        categories using the trained Wisconsin dataset model.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- OVERVIEW CARDS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Records", "569")

with col2:
    st.metric("Input Features", "30")

with col3:
    st.metric("ML Models", "2")


st.divider()

st.markdown(
    '<div class="section-title">🧪 Patient Measurement Analysis</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter all 30 numerical features below. "
    "Default values are dataset medians for demonstration."
)


# ---------------- INPUT GROUPS ----------------
groups = {
    "Mean Measurements": [
        f for f in feature_names if f.startswith("mean ")
    ],
    "Standard Error Measurements": [
        f for f in feature_names if f.endswith(" error")
    ],
    "Worst Measurements": [
        f for f in feature_names if f.startswith("worst ")
    ]
}

# Get dataset reference ranges
from sklearn.datasets import load_breast_cancer

reference_data = load_breast_cancer()

reference_df = pd.DataFrame(
    reference_data.data,
    columns=reference_data.feature_names
)


# ---------------- PREDICTION FORM ----------------
input_values = {}

with st.form("prediction_form"):

    for group_name, features in groups.items():

        with st.expander(
            group_name + " (" + str(len(features)) + " features)",
            expanded=True
        ):

            cols = st.columns(3)

            for i, feature in enumerate(features):

                with cols[i % 3]:

                    minimum = float(reference_df[feature].min())
                    maximum = float(reference_df[feature].max())
                    default = float(reference_df[feature].median())

                    input_values[feature] = st.number_input(
                        feature.title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        format="%.4f",
                        key="input_" + feature
                    )

    st.write("")

    submitted = st.form_submit_button(
        "🔍 Analyze & Predict",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------
if submitted:

    input_df = pd.DataFrame(
        [[input_values[f] for f in feature_names]],
        columns=feature_names
    )

    prediction = int(model.predict(input_df)[0])

    probabilities = model.predict_proba(input_df)[0]

    malignant_probability = float(probabilities[0])
    benign_probability = float(probabilities[1])

    predicted_probability = float(probabilities[prediction])

    st.session_state["prediction_result"] = {
        "prediction": prediction,
        "malignant": malignant_probability,
        "benign": benign_probability,
        "confidence": predicted_probability
    }


# ---------------- DISPLAY RESULTS ----------------
if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Prediction Results</div>',
        unsafe_allow_html=True
    )

    result_col, confidence_col = st.columns(2)

    with result_col:

        if result["prediction"] == 0:
            st.error("Prediction: MALIGNANT")
            st.markdown(
                "The model classified this input as malignant."
            )
        else:
            st.success("Prediction: BENIGN")
            st.markdown(
                "The model classified this input as benign."
            )

    with confidence_col:

        st.metric(
            "Predicted Class Probability",
            f'{result["confidence"] * 100:.2f}%'
        )

    st.subheader("Class Probability Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.write("🔴 Malignant probability")
        st.progress(result["malignant"])
        st.write(f'{result["malignant"] * 100:.2f}%')

    with col2:
        st.write("🟢 Benign probability")
        st.progress(result["benign"])
        st.write(f'{result["benign"] * 100:.2f}%')

    st.info(
        "These values represent the model's predicted probabilities, "
        "not medically validated diagnostic confidence."
    )


# ---------------- MODEL INFORMATION ----------------
st.divider()

with st.expander("📚 About This Machine Learning Project"):

    st.write("**Dataset:** Wisconsin Diagnostic Breast Cancer Dataset")
    st.write("**Features:** 30 numerical measurements")
    st.write("**Classes:** Malignant and Benign")
    st.write("**Selected Model:** " + str(model_name))

    st.write(
        "The model was trained using a machine learning pipeline "
        "with preprocessing and classification."
    )


# ---------------- DISCLAIMER ----------------
st.divider()

st.warning(
    "⚕️ MEDICAL DISCLAIMER: This application is developed "
    "for educational and internship demonstration purposes only. "
    "It is not a medical diagnostic device. Do not use its "
    "predictions for healthcare decisions. Always consult "
    "qualified healthcare professionals."
)

st.markdown(
    """
    <div style="text-align:center; padding:20px; color:#64748b;">
        <b>BreastCare AI</b><br>
        CODE-A-NOVA | ML Developer Internship Project<br>
        Built using Python, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
