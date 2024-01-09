import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

# Load historical stock price data
data = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\BAJAJFINSV.csv")
prices = data["Close"].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
prices = scaler.fit_transform(prices)

# Split data into training and testing sets
train_size = int(len(prices) * 0.8)
train_data, test_data = prices[:train_size], prices[train_size:]

# Prepare data sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

seq_length = 10
X_train = create_sequences(train_data, seq_length)
y_train = train_data[seq_length:]

# Build an LSTM model
model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    keras.layers.LSTM(50, return_sequences=False),
    keras.layers.Dense(25),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions on the test data
X_test = create_sequences(test_data, seq_length)
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size+seq_length:], data['Close'][train_size+seq_length:], label='Actual Price', color='b')
plt.plot(data.index[train_size+seq_length:], predicted_prices, label='Predicted Price', color='r')
plt.legend()
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
