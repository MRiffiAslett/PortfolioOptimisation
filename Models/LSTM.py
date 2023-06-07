import os
import time
import warnings
import numpy as np
import pandas as pd
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer

np.random.seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore") 

data = pd.read_hdf('DLdata.h5', 'GoldData')

for column in data.columns:
    data[column + '_returns'] = data[column].pct_change().fillna(0)

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

look_back = 12
scaler = StandardScaler()
data.loc[:, 'Gold'] = scaler.fit_transform(data.loc[:, 'Gold'])
scaler1 = StandardScaler()
data.loc[:, 'Inflation'] = scaler1.fit_transform(data.loc[:, 'Inflation'])
scaler2 = StandardScaler()
data.loc[:, 'InterestRate'] = scaler2.fit_transform(data.loc[:, 'InterestRate'])
scaler3 = StandardScaler()
data.loc[:, 'DJI'] = scaler3.fit_transform(data.loc[:, 'DJI'])

train_data = data.loc[data.index < pd.to_datetime('2016-01-01')]

timeseries = np.asarray(data.Gold)
timeseries = np.atleast_2d(timeseries)
if timeseries.shape[0] == 1:
    timeseries = timeseries.T

X = np.atleast_3d(np.array([timeseries[start:start + look_back] for start in range(0, timeseries.shape[0] - look_back)]))
y = timeseries[look_back:]

predictors = ['Gold']

model = Sequential()
model.add(LSTM(input_shape=(1,), input_dim=1, output_dim=6, return_sequences=True))
model.add(LSTM(input_shape=(1,), input_dim=1, output_dim=6, return_sequences=False))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss="mse", optimizer="rmsprop")
model.fit(X,
          y,
          epochs=1000,
          batch_size=80,
          verbose=1,
          shuffle=False)

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
data.to_hdf('DLdata.h5', 'Pred_LSTM')
data.loc[:, 'Gold'] = scaler.inverse_transform(data.loc[:, 'Gold'])
data.loc[:, 'Pred'] = scaler.inverse_transform(data.loc[:, 'Pred'])

plt.plot(data.Gold, 'y')
plt.plot(data.Pred, 'g')
