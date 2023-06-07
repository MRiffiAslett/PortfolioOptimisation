# Import necessary libraries
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt


# Set a random seed for reproducibility
np.random.seed(7)

# Set environment variable to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Ignore warning messages
warnings.filterwarnings("ignore") 

# Read data from HDF5 file into a pandas DataFrame
df = pd.read_hdf('DeepLearning.h5', 'Data_Index')

# Select the target index to be predicted
target_index = '^DJI'

# Create a list of stock columns
stocks = list(df.columns)
stocks.remove(target_index)

# Extract the target index series and perform normalization
target_series = df[target_index].dropna()
scaler = Normalizer()
target_series.loc[:] = scaler.fit_transform(target_series)[0]

# Define the number of previous time steps to consider
look_back = 2

# Convert the target series to a 2D numpy array
timeseries = np.asarray(target_series)
timeseries = np.atleast_2d(timeseries)
if timeseries.shape[0] == 1:
    timeseries = timeseries.T

# Reshape the target series into the required 3D input format for the LSTM model
input_data = np.atleast_3d(np.array([timeseries[start:start + look_back] for start in range(0, timeseries.shape[0] - look_back)]))

# Set the target variable for the LSTM model (in this case, it is the input data itself)
target_variable = input_data

# Define the predictors (features) to be used for training the LSTM model
predictors = ['Gold']

# Create an instance of the sequential model
model = Sequential()

# Add LSTM layers to the model
model.add(LSTM(5, input_dim=1, return_sequences=True))
model.add(LSTM(1, return_sequences=True))

# Add an activation function to the output layer
model.add(Activation('linear'))

# Compile the model by specifying the loss function and optimizer
model.compile(loss="mse", optimizer="rmsprop")

# Train the LSTM model using the input data and target variable
model.fit(input_data, target_variable, epochs=1000, batch_size=80, verbose=1, shuffle=False)

# Iterate over the DataFrame index
for i in range(len(df.index)):
    # Skip the initial time steps since they don't have sufficient look-back data
    if i <= look_back:
        continue
    
    # Extract the input sequence for prediction
    input_sequence = target_series.loc[target_series.index[i-look_back:i]].values
    
    # Reshape the input sequence to match the model's expected input shape
    reshaped_input = input_sequence.reshape(1, look_back, 1)
    
    # Use the trained LSTM model to make predictions on the input sequence
    predicted_output = model.predict(reshaped_input)
