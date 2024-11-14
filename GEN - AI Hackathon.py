import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler  # Handle class imbalance
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load dataset
data = pd.read_csv('ecommerce_data.csv')

# Display the first few rows
print(data.head())

# Get dataset summary
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Fill or drop missing values
data.fillna('', inplace=True)

# Example for text data preprocessing
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Assuming 'product' column is the text column you want to clean
data['cleaned_text'] = data['product'].apply(clean_text)

# Convert categorical data to numerical
categorical_columns = ['user id', 'product id']
data = pd.get_dummies(data, columns=categorical_columns)

# Convert 'rating' to binary target
data['binary_rating'] = (data['rating'] >= 1000).astype(int)

# Check class distribution
print(data['binary_rating'].value_counts())

X = data.drop(['rating', 'product', 'cleaned_text', 'binary_rating'], axis=1)
y = data['binary_rating']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

# Check the new class distribution
print(pd.Series(y_train_bal).value_counts())

# Dummy variables
vocab_size = 5000
embedding_dim = 128
max_length = 100

# Define the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_bal, y_train_bal, epochs=50, batch_size=16, validation_split=0.2)

# Make predictions
predictions = model.predict(X_test)

# Adjust the threshold
threshold = 0.5  # Start with default, then adjust if needed
predictions = (predictions > threshold).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Visualize the results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
model.save('product_recommendation_model.h5')