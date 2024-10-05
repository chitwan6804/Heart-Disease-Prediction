#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
from keras import models
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and prepare the data
@st.cache
def load_data():
    data = pd.read_csv('heart.csv')
    return data

# Build the model
def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Sidebar for navigation
st.sidebar.header("Navigation")
menu_options = ["Home", "Going Through Data", "Training and Prediction"]
selected_option = st.sidebar.selectbox("Choose an option", menu_options)

# Home page
if selected_option == "Home":
    st.title("🏠 Home")
    st.write("Welcome to the Heart Disease Prediction App! Use the navigation menu to explore different sections of the application.")

# Going Through Data
elif selected_option == "Going Through Data":
    st.title("📊 Going Through Data")
    data = load_data()
    st.write("Dataset Shape:", data.shape)
    st.write(data.describe())

# Training and Prediction
elif selected_option == "Training and Prediction":
    st.title("💻 Training and Prediction")
    
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
    model = build_model((train_Features.shape[1],))
    history = model.fit(X_train, train_target, epochs=50, batch_size=10, validation_split=0.2)

    # Plotting the training and validation loss
    st.subheader("Training and Validation Loss")
    fig, ax = plt.subplots()
    loss_values = history.history['loss']
    val_loss_values = history.history['val_loss']
    epochs = range(1, len(loss_values) + 1)
    ax.plot(epochs, loss_values, 'bo', label='Training Loss')
    ax.plot(epochs, val_loss_values, 'b', label='Validation Loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

    # Plotting the training and validation accuracy
    st.subheader("Training and Validation Accuracy")
    fig, ax = plt.subplots()
    acc_values = history.history['accuracy']
    val_acc_values = history.history['val_accuracy']
    ax.plot(epochs, acc_values, 'bo', label='Training Accuracy')
    ax.plot(epochs, val_acc_values, 'b', label='Validation Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, test_target)
    st.subheader("Model Evaluation")
    st.write(f'Test Loss: {loss:.4f}')
    st.write(f'Test Accuracy: {accuracy:.4f}')

    # Input data for prediction
    st.subheader("Predict Heart Disease")
    input_data = {}
    for column in Features.columns:
        input_data[column] = st.number_input(f"Enter {column}", min_value=float(Features[column].min()), 
                                              max_value=float(Features[column].max()), 
                                              value=float(Features[column].mean()))
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Normalize the input data
    input_scaled = scaler.transform(input_df)

    # Display input values in a table
    st.write("### Input Values")
    st.table(input_df)

    # Make predictions
    if st.button("Predict"):
        predicted_probability = model.predict(input_scaled)[0][0]
        prediction = "Yes! Patient is predicted to be suffering from heart disease." if predicted_probability > 0.5 else "No! Patient is predicted not to be suffering from heart disease."
        
        # Show prediction results
        st.write(f"### Prediction Result: {prediction}")
        
        # Display the predicted probability
        st.write(f"Predicted Probability of Heart Disease: {predicted_probability:.2f}")

        # Visualize the prediction probability
        st.subheader("Prediction Probability")
        
        # Ensure predicted_probability is between 0 and 1 for the progress bar
        progress_value = max(0.0, min(predicted_probability, 1.0))
        st.progress(progress_value)  # Progress bar for probability
        
        # Optional: Add some interpretation based on predicted probability
        if predicted_probability > 0.5:
            st.warning("The model suggests that the patient may have heart disease. Consider consulting a healthcare professional.")
        else:
            st.success("The model suggests that the patient is unlikely to have heart disease.")
