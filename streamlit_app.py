#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
from keras import models, layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns  # <-- Add this line

# Load and prepare the data
@st.cache_data
def load_data():
    data = pd.read_csv('heart.csv')
    return data

# Build the model
def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Sidebar for navigation
st.sidebar.header("Navigation")
menu_options = ["Home", "Going Through Data", "Predict for a Patient"]
selected_option = st.sidebar.selectbox("Choose an option", menu_options)

# Home page
if selected_option == "Home":
    st.title("🏠 Heart Disease Prediction App")
    
    # Introduction to the app
    st.header("Introduction")
    st.write("""
    Welcome to the **Heart Disease Prediction App**!
    
    This application is designed to help predict the likelihood of heart disease based on several health indicators. It uses machine learning algorithms to analyze patient data and provides a probability estimate of heart disease risk.
    """)
    
    # How it works
    st.subheader("How it Works:")
    st.write("""
    - The app uses a **neural network** model trained on the **Heart Disease dataset**.
    - Users can input various health-related factors such as age, cholesterol levels, blood pressure, and more to get an assessment of their heart disease risk.
    - After training the model, you can provide new data to make predictions about heart disease likelihood.
    """)
    
    # Features of the App
    st.subheader("Features of the App:")
    st.write("""
    1. **Data Overview**: Explore the heart disease dataset and visualize important trends and statistics.
    2. **Model Training**: Train a machine learning model on the dataset and view the model’s performance through metrics like accuracy and loss.
    3. **Heart Disease Prediction**: Enter your own health information and get a personalized prediction on the risk of heart disease.
    """)
    
    # Who can use the app
    st.subheader("Who Can Use This App:")
    st.write("""
    - **Healthcare professionals**: To analyze patient data and provide a quick assessment.
    - **Researchers**: To explore machine learning in healthcare.
    - **General users**: To assess their heart disease risk based on their health data.
    """)
    
    # Disclaimer
    st.subheader("Disclaimer:")
    st.write("""
    This app is intended for educational purposes and should not be used as a substitute for professional medical advice. Always consult with a healthcare provider for accurate diagnosis and treatment.
    """)


# Going Through Data
# Going Through Data
elif selected_option == "Going Through Data":
    st.title("📊 Going Through Data")
    
    # Load data
    data = load_data()
    
    # Display Dataset Shape
    st.write("### Dataset Shape:", data.shape)
    
    # Display the dataset description
    st.write(data.describe())
    
    # Explain the dataset parameters (features)
    st.subheader("Understanding the Parameters:")
    st.write("""
    The dataset contains several key health indicators which are used to assess heart disease risk. Here's a brief explanation of each parameter:

    - **Age**: The age of the patient.
    - **Sex**: Gender of the patient (1 = male, 0 = female).
    - **Chest Pain Type (cp)**: Indicates the type of chest pain experienced (0-3), where:
        - 0: Typical Angina
        - 1: Atypical Angina
        - 2: Non-Anginal Pain
        - 3: Asymptomatic
    - **Resting Blood Pressure (trestbps)**: The patient's resting blood pressure (in mm Hg).
    - **Cholesterol (chol)**: Serum cholesterol level (in mg/dl).
    - **Fasting Blood Sugar (fbs)**: Whether the patient's fasting blood sugar is above 120 mg/dl (1 = true; 0 = false).
    - **Resting ECG Results (restecg)**: Results of the resting electrocardiogram (0-2), where:
        - 0: Normal
        - 1: ST-T wave abnormality
        - 2: Left ventricular hypertrophy
    - **Maximum Heart Rate Achieved (thalach)**: Maximum heart rate achieved during exercise.
    - **Exercise-Induced Angina (exang)**: Whether the patient experiences angina as a result of exercise (1 = yes; 0 = no).
    - **ST Depression (oldpeak)**: Depression of the ST segment induced by exercise relative to rest.
    - **Slope of the Peak Exercise ST Segment (slope)**: Slope of the ST segment during peak exercise (0-2).
    - **Number of Major Vessels (ca)**: Number of major vessels (0-3) colored by fluoroscopy.
    - **Thalassemia (thal)**: A blood disorder involving the hemoglobin (0-3), where:
        - 0: Normal
        - 1: Fixed Defect
        - 2: Reversible Defect
    """)
    
    # Correlation Matrix
    st.subheader("Feature Correlations")
    st.write("""
    Below is the correlation matrix that shows how each parameter correlates with others. Correlation values range from -1 to 1:
    - A value of **1** means a perfect positive correlation.
    - A value of **-1** means a perfect negative correlation.
    - A value close to **0** indicates no correlation.
    
    This helps to understand the relationships between different health indicators and how they might affect heart disease risk.
    """)
    
    # Show correlation matrix
    correlation_matrix = data.corr()
    st.write("### Correlation Matrix Heatmap")
    fig, ax = plt.subplots()
    ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(fig)
    
    # # Optional: Display raw correlation data
    # st.write("### Raw Correlation Data")
    # st.write(correlation_matrix)


# Training and Prediction
elif selected_option == "Predict for a Patient":
    st.title("🔧 Predict for a Patient")
    
    # Load data
    data = load_data()
    
    # Split data into features and target
    Features = data.iloc[:, :-1]
    Target = data.iloc[:, -1]
    
    # Splitting data into training and test sets
    train_Features, test_Features, train_target, test_target = train_test_split(Features, Target, test_size=0.2, random_state=42)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_Features)
    X_test = scaler.transform(test_Features)
    
    # Build and train the model
    model = build_model(train_Features.shape[1])
    model.fit(X_train, train_target, epochs=50, batch_size=10, validation_split=0.2)

    st.write("### Model Training Completed")

    # Prediction Section
    st.subheader("Predict Heart Disease for a New Patient")
    
    # Input data for prediction
    input_data = {
        'Age': st.number_input("Enter Age", min_value=0, max_value=100, value=50),
        'Sex': st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1]),
        'ChestPainType': st.number_input("Chest Pain Type (1-4)", min_value=1, max_value=4, value=1),
        'RestingBP': st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120),
        'Cholesterol': st.number_input("Cholesterol Level", min_value=0, max_value=600, value=200),
        'FastingBS': st.selectbox("Fasting Blood Sugar (1 = >120mg/dl, 0 = otherwise)", [0, 1]),
        'RestingECG': st.number_input("Resting ECG Result (0-2)", min_value=0, max_value=2, value=1),
        'MaxHR': st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150),
        'ExerciseAngina': st.selectbox("Exercise-induced Angina (1 = Yes, 0 = No)", [0, 1]),
        'Oldpeak': st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0),
    }
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Normalize the input data
    input_scaled = scaler.transform(input_df)
    
    # Make predictions
    if st.button("Predict"):
        predicted_probability = model.predict(input_scaled)[0][0]
        prediction = "Yes, patient is likely to have heart disease." if predicted_probability > 0.5 else "No, patient is unlikely to have heart disease."
        
        # Show prediction results
        st.write(f"### Prediction Result: {prediction}")
        st.write(f"Predicted Probability: {predicted_probability:.2f}")
        
        # Visualize the prediction probability
        st.subheader("Prediction Probability")
        st.progress(predicted_probability)
        
        # Interpretation
        if predicted_probability > 0.5:
            st.warning("This suggests a higher risk of heart disease. Consult a healthcare professional.")
        else:
            st.success("The patient is predicted to be at low risk for heart disease.")
