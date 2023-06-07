import os
import warnings
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore") 

data = pd.read_hdf('DLdata.h5', 'Data_Gold')

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
val_data = train_data.loc[train_data.index >= pd.to_datetime('2013-01-01')]
train_data = train_data.loc[train_data.index < pd.to_datetime('2013-01-01')]
test_data = data.loc[data.index >= pd.to_datetime('2016-01-01')]

predictors = ['Gold', 'DJI', 'Inflation']
train_x, train_y = create_dataset(train_data, look_back=look_back, columns=predictors)
val_x, val_y = create_dataset(val_data, look_back=look_back, columns=predictors)
test_x, test_y = create_dataset(test_data, look_back=look_back, columns=predictors)

model = Sequential()
model.add(Dense(10, activation='tanh', kernel_initializer='normal', input_dim=look_back * len(predictors)))
model.add(Dropout(0.2))
model.add(Dense(5, activation='tanh', kernel_initializer='normal'))
model.add(Dropout(0.2))
model.add(Dense(len(predictors)))

model.compile(loss="mse", optimizer="adam")
model.fit(train_x, train_y, epochs=10000, batch_size=50, verbose=2, shuffle=False, validation_data=(val_x, val_y))

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
    y = model.predict(a.reshape(1, look_back * len(predictors)))
    data.loc[data.index[i], 'Pred'] = y[0][0]

data.loc[:, 'Gold'] = scaler.inverse_transform(data.loc[:, 'Gold'])
data.loc[:, 'Pred'] = scaler.inverse_transform(data.loc[:, 'Pred'])
data.to_hdf('DLdata.h5', 'Pred_DeepRegression')

plt.plot(data.Gold, 'y')
plt.plot(data.Pred, 'g')
