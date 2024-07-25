import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate simple time series data
def generate_data(n_samples=100):
    X = np.arange(n_samples)
    y = np.sin(X * 0.1) + np.random.normal(scale=0.1, size=n_samples)
    return X, y

# Prepare data for LSTM
def prepare_data(y, time_steps):
    X, y_out = [], []
    for i in range(len(y) - time_steps):
        X.append(y[i:i + time_steps])
        y_out.append(y[i + time_steps])
    return np.array(X), np.array(y_out)

# Normalize data to range [0, 1]
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Denormalize data
def denormalize(data, original_data):
    return data * (np.max(original_data) - np.min(original_data)) + np.min(original_data)

# Parameters
time_steps = 5
n_samples = 100

# Generate and prepare data
_, y = generate_data(n_samples)
X, y = prepare_data(y, time_steps)

# Normalize data
y_min, y_max = np.min(y), np.max(y)
X = normalize(X)
y = normalize(y)

# Reshape for LSTM [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define and train the model
model = Sequential([
    LSTM(10, activation='relu', input_shape=(time_steps, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, verbose=0)

# Make predictions
y_pred = model.predict(X)

# Denormalize for output
y = denormalize(y, y)
y_pred = denormalize(y_pred.reshape(-1), y)

# Print results
for i in range(len(y)):
    print(f"True: {y[i]:.2f}, Predicted: {y_pred[i]:.2f}")

