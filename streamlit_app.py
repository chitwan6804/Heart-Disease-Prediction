#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from keras import models, layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set the title and icon for the Streamlit app
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🚑",
    layout="wide"
)

# Load and prepare the data
@st.cache_data
def load_data():
    data = pd.read_csv('heart.csv')
    return data

# Load the model if it exists
@st.cache_data
def load_trained_model():
    try:
        return load_model('heart_disease_model.h5')
    except Exception as e:
        st.warning(f"Model not found or failed to load: {e}")
        return None

# Build the model
def build_model(input_shape):
    try:
        model = models.Sequential()
        model.add(layers.Input(shape=(input_shape,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error while building the model: {e}")
        return None

# Sidebar for navigation
st.sidebar.header("Navigation")
menu_options = ["Home", "Going Through Data", "Predict for a Patient"]
selected_option = st.sidebar.selectbox("Choose an option", menu_options)

# Load data
data = load_data()

# Split data into features and target
Features = data.iloc[:, :-1]
Target = data.iloc[:, -1]

# Splitting data into training and test sets
train_Features, test_Features, train_target, test_target = train_test_split(
    Features, Target, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_Features)
X_test = scaler.transform(test_Features)

# Load or build the model
model = load_trained_model()

# Train and save the model if not already loaded
if model is None:
    model = build_model(train_Features.shape[1])
    model.fit(X_train, train_target, epochs=50, batch_size=10, validation_split=0.2)
    # Save the model after training
    model.save('heart_disease_model.h5')

# Home page
if selected_option == "Home":
    st.title("🏠 Heart Disease Prediction App")
    
    st.header("Introduction")
    st.write("""
    Welcome to the **Heart Disease Prediction App**!
    
    This application is designed to help predict the likelihood of heart disease based on several health indicators. It uses a trained neural network model for the predictions.
    """)

# Going Through Data
elif selected_option == "Going Through Data":
    st.title("📊 Going Through Data")
    
    st.write("### Dataset Shape:", data.shape)
    st.write(data.describe())
    
    st.subheader("Understanding the Parameters:")
    st.write("""
    The dataset contains several key health indicators which are used to assess heart disease risk.
    """)
    
    st.subheader("Correlation Matrix")
    correlation_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=90)
    ax.set_yticklabels(correlation_matrix.columns)
    for (i, j), val in np.ndenumerate(correlation_matrix.values):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
    st.pyplot(fig)

# Prediction Section
elif selected_option == "Predict for a Patient":
    st.title("🔧 Making Prediction")
    
    st.subheader("Predict Heart Disease for a New Patient")
    input_data = {
        'age': st.number_input("Enter Age", min_value=0, max_value=100, value=50),
        'sex': st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1]),
        'cp': st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0),
        'trestbps': st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120),
        'chol': st.number_input("Cholesterol Level", min_value=0, max_value=600, value=200),
        'fbs': st.selectbox("Fasting Blood Sugar (1 = >120mg/dl, 0 = otherwise)", [0, 1]),
        'restecg': st.number_input("Resting ECG Result (0-2)", min_value=0, max_value=2, value=1),
        'thalach': st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150),
        'exang': st.selectbox("Exercise-induced Angina (1 = Yes, 0 = No)", [0, 1]),
        'oldpeak': st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0),
        'slope': st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=1),
        'ca': st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0),
        'thal': st.number_input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", min_value=1, max_value=3, value=1),
    }

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    if st.button("Predict"):
        predicted_probability = model.predict(input_scaled)[0][0]
        prediction = "Yes! Patient is predicted to be suffering from heart disease." if predicted_probability > 0.5 else "No! Patient is predicted not to be suffering from heart disease."
        st.write(f"### Prediction Result: {prediction}")
        st.write(f"Predicted Probability of Heart Disease: {predicted_probability:.2f}")
        st.progress(float(predicted_probability))
        if predicted_probability > 0.5:
            st.warning("The model suggests that the patient may have heart disease. Consider consulting a healthcare professional.")
        else:
            st.success("The model suggests that the patient is unlikely to have heart disease.")
