import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import models, layers, regularizers  # Include regularizers import
from keras.optimizers import RMSprop  # Import RMSprop
import matplotlib.pyplot as plt


# Set page configuration
st.set_page_config(page_title='Heart Disease Prediction', layout='wide')

# Title of the app
st.title('Heart Disease Prediction App')
st.markdown("""
This application predicts the presence of heart disease using various health metrics. 
You can explore the dataset, visualize correlations, and evaluate the model's performance.
""")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('heart.csv')
    return data

data = load_data()

# Display data if checkbox is selected
if st.checkbox('Show Raw Data'):
    st.subheader('Raw Dataset')
    st.write(data)

# Correlation matrix display
if st.checkbox('Show Correlation with Target'):
    corr_matrix = data.corr()
    st.subheader('Correlation Matrix')
    st.write(corr_matrix['target'].sort_values(ascending=False))

    # Display a heatmap of the correlation matrix
    plt.figure(figsize=(10, 6))
    st.write(plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest'))
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    st.pyplot(plt.gcf())

# Splitting features and target
Features = data.iloc[:, :-1]
Target = data.iloc[:, -1]

# Train-test split
train_Features, test_Features, train_target, test_target = train_test_split(Features, Target, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_Features)
X_test = scaler.transform(test_Features)

def build_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(train_Features.shape[1],)))
    model.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Train the model
if st.button('Train Model'):
    with st.spinner('Training the model...'):
        history = model.fit(X_train, train_target, epochs=100, batch_size=10, validation_split=0.2, verbose=0)
    st.success('Training complete!')

    # Plot training and validation accuracy
    st.subheader('Training and Validation Accuracy')
    acc_values = history.history['accuracy']
    val_acc_values = history.history['val_accuracy']
    epochs = range(1, len(acc_values) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc_values, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc_values, 'b-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    st.pyplot(plt.gcf())

# Predicting heart disease based on test set
if st.button('Evaluate Model'):
    
    # Debugging outputs
    st.write(f'Training set shape: {X_train.shape}, {train_target.shape}')
    st.write(f'Test set shape: {X_test.shape}, {test_target.shape}')

    train_loss, train_accuracy = model.evaluate(X_train, train_target)
    st.write(f'**Training Loss:** {train_loss:.4f}')
    st.write(f'**Training Accuracy:** {train_accuracy:.4f}')
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, test_target, verbose=1)
    st.write(f'**Test Loss:** {loss:.4f}')
    st.write(f'**Test Accuracy:** {accuracy:.4f}')
    
    # Make predictions on test set
    predicted_probability = model.predict(X_test)
    # Show predictions
    predictions = ['Heart Disease' if prob > 0.5 else 'No Heart Disease' for prob in predicted_probability]

    # Compare with actual values
    comparison_df = pd.DataFrame({
        'Actual': ['Heart Disease' if x == 1 else 'No Heart Disease' for x in test_target.values],
        'Predicted': predictions
    })
    st.write(comparison_df.head(10))  # Display the first 10 predictions
