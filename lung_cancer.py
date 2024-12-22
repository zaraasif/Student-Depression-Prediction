import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .css-1v3fvcr {
        background-color: #1E3A8A;
        color: white;
    }
    h1 {
        color: #1D4ED8;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    .css-1d391kg {
        background-color: #ffffff;
    }
    .stMarkdown {
        font-size: 18px;
        line-height: 1.5;
    }
    .stImage img {
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load logo
logo_path = "lung_cancer_logo.jpg"  # Replace with your actual logo path
try:
    logo = Image.open(logo_path)
    st.image(logo, width=200)
except FileNotFoundError:
    st.warning("Logo file not found!")

# Title of the app
st.title("Lung Cancer Prediction")
st.markdown("<p style='text-align: center; color: #1D4ED8;'>by Zara Asif</p>", unsafe_allow_html=True)

# Load dataset
file_path = 'dataset.csv'  # Replace with the actual dataset path
dataset = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
for col in ['GENDER', 'LUNG_CANCER']:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    label_encoders[col] = le

# Split features and target
X = dataset.drop(columns=['LUNG_CANCER'])
y = dataset['LUNG_CANCER']

# Standardize numerical features
scaler = StandardScaler()
numerical_columns = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                     'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
                     'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
                     'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Save model components
with open('lung_cancer_svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_model, model_file)
with open('lung_cancer_label_encoders.pkl', 'wb') as encoders_file:
    pickle.dump(label_encoders, encoders_file)
with open('lung_cancer_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Use session_state to prevent unnecessary re-runs and store prediction results
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
    st.session_state.prediction_proba = None

# Input fields
st.header("Provide the Following Details:")
age = st.number_input("Age", min_value=15, max_value=100, step=1, value=40)
smoking = st.radio("Smoking", ["No", "Yes"])
yellow_fingers = st.radio("Yellow Fingers", ["No", "Yes"])
anxiety = st.radio("Anxiety", ["No", "Yes"])
peer_pressure = st.radio("Peer Pressure", ["No", "Yes"])
chronic_disease = st.radio("Chronic Disease", ["No", "Yes"])
fatigue = st.radio("Fatigue", ["No", "Yes"])
allergy = st.radio("Allergy", ["No", "Yes"])
wheezing = st.radio("Wheezing", ["No", "Yes"])
alcohol_consuming = st.radio("Alcohol Consuming", ["No", "Yes"])
coughing = st.radio("Coughing", ["No", "Yes"])
shortness_of_breath = st.radio("Shortness of Breath", ["No", "Yes"])
swallowing_difficulty = st.radio("Swallowing Difficulty", ["No", "Yes"])
chest_pain = st.radio("Chest Pain", ["No", "Yes"])

# Prepare inputs
input_data = [
    age,
    1 if smoking == "Yes" else 0,
    1 if yellow_fingers == "Yes" else 0,
    1 if anxiety == "Yes" else 0,
    1 if peer_pressure == "Yes" else 0,
    1 if chronic_disease == "Yes" else 0,
    1 if fatigue == "Yes" else 0,
    1 if allergy == "Yes" else 0,
    1 if wheezing == "Yes" else 0,
    1 if alcohol_consuming == "Yes" else 0,
    1 if coughing == "Yes" else 0,
    1 if shortness_of_breath == "Yes" else 0,
    1 if swallowing_difficulty == "Yes" else 0,
    1 if chest_pain == "Yes" else 0,
]

# Standardize input data
input_array = np.array(input_data).reshape(1, -1)

# Load the scaler and model from the saved files (these should be done at the top level of your script)
with open('lung_cancer_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('lung_cancer_svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Standardize input data
input_scaled = scaler.transform(input_array)

# Prediction Button
if st.button("Predict"):
    # This prevents the prediction from happening on every click
    with st.spinner('Running prediction...'):
        st.session_state.prediction = svm_model.predict(input_scaled)[0]
        st.session_state.prediction_proba = svm_model.predict_proba(input_scaled)[0]

# Display prediction results
if st.session_state.prediction is not None:
    prediction = st.session_state.prediction
    prediction_proba = st.session_state.prediction_proba
    result = "Yes" if prediction == 1 else "No"
    st.subheader("Prediction Result")
    st.write(f"Lung Cancer Detected: **{result}**")
    st.write(f"Confidence: {prediction_proba[1]*100:.2f}%")

    if result == "Yes":
        st.error("Lung cancer detected. Please consult a doctor immediately.")
    else:
        st.success("No signs of lung cancer detected. Maintain a healthy lifestyle!")
