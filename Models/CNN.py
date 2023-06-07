# Import necessary libraries
import os
import time
import warnings
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer

# Set a random seed for reproducibility
np.random.seed(7)

# Set environment variable to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Ignore warning messages
warnings.filterwarnings("ignore") 

# Read data from HDF5 file into a pandas DataFrame
data = pd.read_hdf('DLdata.h5', 'GoldData')

# Calculate returns for each column
for column in data.columns:
    data[column + '_returns'] = data[column].pct_change().fillna(0)

# Define function to create dataset with specified look-back and columns
def create_dataset(dataset, look_back=1, columns=['Gold']):
    dataX, dataY = [], []
    for i in range(len(dataset.index)):
        if i <= look_back:
            continue
        a = None
        for col in columns:
            b = dataset.loc[dataset.index[i - look_back:i], col].to_numpy()
            if a is None:
                a = b
            else:
                a = np.append(a, b)
        dataX.append(a)
        dataY.append(dataset.loc[dataset.index[i], columns].to_numpy())
    return np.array(dataX), np.array(dataY)

# Define the number of previous time steps to consider
look_back = 12

# Create a scaler for standardization
scaler = StandardScaler()

# Normalize the 'Gold' column
data.loc[:, 'Gold'] = scaler.fit_transform(data.loc[:, 'Gold'])

# Create an instance of the sequential model
model = Sequential()

# Add CNN layers to the model
model.add(Convolution1D(input_shape=(look_back, 1),
                        nb_filter=64,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))

model.add(Convolution1D(input_shape=(look_back, 1),
                        nb_filter=64,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(250))
model.add(Dropout(0.25))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('linear'))

# Compile the model by specifying the loss function and optimizer
model.compile(loss="mse", optimizer="rmsprop")

# Train the model using the input data and target variable
model.fit(X,
          y,
          epochs=1000,
          batch_size=80,
          verbose=1,
          shuffle=False)

# Perform predictions using the trained model
data['Pred'] = data.loc[data.index[0], 'Gold']
for i in range(len(data.index)):
    if i <= look_back:
        continue
    a = None
    for col in predictors:
        b = data.loc[data.index[i - look_back:i], col].to_numpy()
        if a is None:
            a = b
        else:
            a = np.append(a, b)
        a = a
    y = model.predict(a.reshape(1, look_back * len(predictors), 1))
    data.loc[data.index[i], 'Pred'] = y[0][0]

# Inverse transform the 'Gold' and 'Pred' columns
data.loc[:, 'Gold'] = scaler.inverse_transform(data.loc[:, 'Gold'])
data.loc[:, 'Pred'] = scaler.inverse_transform(data.loc[:, 'Pred'])

# Save the predicted data to HDF5 file
data.to_hdf('DLdata.h5', 'Pred_CNN')

# Plot the actual and predicted data
plt.plot(data.Gold, 'y')
plt.plot(data.Pred, 'g')
plt.show()
