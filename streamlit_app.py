import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import models, layers
from sklearn.metrics import accuracy_score

# Load data
def load_data():
    data = pd.read_csv('heart.csv')
    return data

# Build model
def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.3))  # Dropout for regularization
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.3))  # Another dropout layer
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Data preprocessing
def preprocess_data(data):
    X = data.drop('target', axis=1)
    y = data['target']

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# Train and evaluate model
def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = build_model(X_train.shape[1])

    # Handle class imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # Train model with validation split and class weights
    history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2, class_weight=class_weights)

    # Evaluate on test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    st.write(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

    return model, history

# Plot training and validation loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)

# Predict heart disease
def predict_heart_disease(model, scaler, user_input):
    input_data = np.array([user_input])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    predicted_prob = prediction[0][0]
    
    # Output results
    if predicted_prob >= 0.5:
        st.write("Prediction Result: Yes! Patient is predicted to be suffering from heart disease.")
    else:
        st.write("Prediction Result: No! Patient is not predicted to be suffering from heart disease.")
    
    st.write(f"Predicted Probability of Heart Disease: {predicted_prob:.2f}")

# Streamlit App
st.title("Heart Disease Prediction")

# Load and preprocess data
data = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

# Train the model
if st.button("Train Model"):
    model, history = train_and_evaluate(X_train, y_train, X_test, y_test)
    plot_loss(history)

# Get user input for prediction
st.write("Enter patient data for prediction:")
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", options=[0, 1])
cp = st.selectbox("Chest Pain Type (0 to 3)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=240)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", options=[0, 1])
restecg = st.selectbox("Resting ECG Results (0 to 2)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, step=0.1, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (0 to 2)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0 to 4)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", options=[1, 2, 3])

# Collect input into a list
user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Make prediction
if st.button("Predict Heart Disease"):
    predict_heart_disease(model, scaler, user_input)
