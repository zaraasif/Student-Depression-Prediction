import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
import pickle
import matplotlib.pyplot as plt
import sys


# Custom CSS for the blue and white theme
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

# Load the logo
logo_path = "A_modern_and_uplifting_logo_design_for_a_mental_he.jpg"
logo = Image.open(logo_path)
st.image(logo, width=200)

# Title of the app
st.title("Student Depression Prediction")
st.markdown("<p style='text-align: center; color: #1D4ED8;'>by Zara Asif</p>", unsafe_allow_html=True)

# Load dataset
file_path = 'Depression Student Dataset.csv'
dataset = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
for col in ['Gender', 'Sleep Duration', 'Dietary Habits',
            'Have you ever had suicidal thoughts ?',
            'Family History of Mental Illness', 'Depression']:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    label_encoders[col] = le

# Splitting features and target
X = dataset.drop(columns=['Depression'])
y = dataset['Depression']

# Standardize numerical features
scaler = StandardScaler()
X[['Age', 'Academic Pressure', 'Study Satisfaction',
   'Study Hours', 'Financial Stress']] = scaler.fit_transform(
    X[['Age', 'Academic Pressure', 'Study Satisfaction',
       'Study Hours', 'Financial Stress']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural network model using Scikit-learn
model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=200, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
st.write(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoders['Depression'].classes_)
disp.plot(ax=ax, cmap='Blues', values_format='d')
st.pyplot(fig)

# Save the model and components
with open('mlp_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('label_encoders.pkl', 'wb') as encoders_file:
    pickle.dump(label_encoders, encoders_file)
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Input fields
st.header("Provide the Following Details:")
gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
age = st.slider("Age", 15, 35, 20)
academic_pressure = st.slider("Academic Pressure (1-5)", 1.0, 5.0, 3.0)
study_satisfaction = st.slider("Study Satisfaction (1-5)", 1.0, 5.0, 3.0)
sleep_duration = st.selectbox("Sleep Duration", label_encoders['Sleep Duration'].classes_)
dietary_habits = st.selectbox("Dietary Habits", label_encoders['Dietary Habits'].classes_)
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", label_encoders['Have you ever had suicidal thoughts ?'].classes_)
study_hours = st.slider("Study Hours", 1, 12, 6)
financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
family_history = st.selectbox("Family History of Mental Illness", label_encoders['Family History of Mental Illness'].classes_)

# Encode inputs
def encode_input(column, value):
    return label_encoders[column].transform([value])[0]

# Process categorical data
categorical_data = [
    encode_input('Gender', gender),
    encode_input('Sleep Duration', sleep_duration),
    encode_input('Dietary Habits', dietary_habits),
    encode_input('Have you ever had suicidal thoughts ?', suicidal_thoughts),
    encode_input('Family History of Mental Illness', family_history)
]

# Prepare numerical data
numerical_data = np.array([age, academic_pressure, study_satisfaction, study_hours, financial_stress]).reshape(1, -1)

# Standardize numerical data
try:
    numerical_data_scaled = scaler.transform(numerical_data)
except ValueError as e:
    st.error(f"Error in data processing: {e}")
    st.stop()

# Combine numerical and categorical data
categorical_data_reshaped = np.array(categorical_data).reshape(1, -1)  # Convert to 2D array
data_combined = np.hstack((numerical_data_scaled, categorical_data_reshaped))

# Prediction
if st.button("Predict"):
    prediction = model.predict(data_combined)[0]
    result = label_encoders['Depression'].inverse_transform([prediction])[0]

    st.subheader("Prediction Result")
    st.write(f"Are you Depressed: **{result}**")

    if result == "Yes":  # Adjust based on your label encoding for "Depression"
        st.error("It seems like you might be experiencing depression. Please reach out for support. You can call the helpline: **(92) 0311 7786264** for immediate assistance.")
    else:
        st.success("You are not currently showing signs of depression. Keep maintaining a healthy lifestyle! Remember to take care of your mental health and seek support if you ever need it.")
