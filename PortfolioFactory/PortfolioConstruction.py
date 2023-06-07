# Import necessary libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import Normalizer, StandardScaler
import matplotlib.pyplot as plt

# Read data from CSV file into a pandas DataFrame
df = pd.read_csv('dji.csv', index_col='Date')
df.index = pd.to_datetime(df.index)
df = df.resample('W-MON', how='last')
stox = list(df.columns)
stox.remove('^DJI')
ind = '^DJI'

# Select a specific time range for the data
df = df[df.index < pd.to_datetime('17-4-2014')]
df = df[df.index > pd.to_datetime('1-07-2011')]


# Calculate returns and create a modified return column based on a threshold
df['ret'] = df['^DJI'].pct_change().fillna(0.1)
df.loc[:, 'new_ret'] = df.apply(lambda r: 0.1 if r['ret'] < -0.08 else r['ret'], axis=1)

# Create a new index column based on the modified returns
df['new_index'] = df.loc[df.index[0], '^DJI']

for i in range(len(df.index)):
    if i > 0:
        df.loc[df.index[i], 'new_index'] = df.loc[df.index[i-1], 'new_index'] * (1.0 + df.loc[df.index[i], 'new_ret'])

# Standardize the features using StandardScaler
en = StandardScaler()
df['^DJI'] = en.fit_transform(df['^DJI'])

for s in stox:
    en1 = StandardScaler()
    df[s] = en1.fit_transform(df[s])

df['new_index'] = en.transform(df['new_index'])

test = df[(df.index >= pd.to_datetime('17-4-2013')) & (df.index < pd.to_datetime('17-4-2014'))]
train = df[df.index < pd.to_datetime('17-04-2014')]

# Function to create the dataset for training
def create_dataset(df):
    dataX = []
    for c in df.columns:
        x = df.loc[:, c].values
        dataX.append(x)
    return np.array(dataX), np.array(dataX)

train_x, train_y = create_dataset(train)

# Create a sequential model and add layers
model = Sequential()
model.add(Dense(output_dim=20, input_dim=30))
model.add(Activation('tanh'))
model.add(Dropout(0.1))
model.add(Dense(output_dim=10, input_dim=20))
model.add(Activation('tanh'))
model.add(Dropout(0.1))
model.add(Dense(output_dim=5, input_dim=10))
model.add(Activation('tanh'))
model.add(Dropout(0.1))
model.add(Dense(output_dim=1, input_dim=5))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

# Train the model
model.fit(train_x, train_y, nb_epoch=2000, batch_size=50, verbose=1)

# Function to create the dataset for the new index
def create_dataset_new_index(df):
    dataX = []
    dataY = []
    for i in df.index:
        x = df.loc[i, stox].values
        dataX.append(x)
        y = df.loc[i, 'new_index']
        dataY.append(y)
    return np.array(dataX), np.array(dataY)

train_x, train_y = create_dataset_new_index(train)

# Create a new model for predicting the new index
new_model = Sequential()
new_model.add(Dense(output_dim=20, input_dim=30))
new_model.add(Activation('tanh'))
new_model.add(Dropout(0.1))
new_model.add(Dense(output_dim=10, input_dim=20))
new_model.add(Activation('tanh'))
new_model.add(Dropout(0.1))
new_model.add(Dense(output_dim=5, input_dim=10))
new_model.add(Activation('tanh'))
new_model.add(Dropout(0.1))
new_model.add(Dense(output_dim=1, input_dim=5))
new_model.compile(loss='mean_squared_error', optimizer='rmsprop')

# Train the new model
new_model.fit(train_x, train_y, nb_epoch=2000, batch_size=50, verbose=1)

# Perform predictions and inverse transform to obtain the original values
df['pred'] = 0.0
df = df[df.index < pd.to_datetime('17-4-2014')]
for i in df.index:
    x = df.loc[i, stox].values
    df.loc[i, 'pred'] = model.predict(x.reshape(1, 30))

df['new_pred'] = 0.0
for i in df.index:
    x = df.loc[i, stox].values
    df.loc[i, 'new_pred'] = new_model.predict(x.reshape(1, 30))

df['pred'] = en.inverse_transform(df['pred'])
df['new_pred'] = en.inverse_transform(df['new_pred'])
df['^DJI'] = en.inverse_transform(df['^DJI'])

# Save the DataFrame to an HDF5 file
df.to_hdf('DeepLearning.h5', 'Deep_Portfolio')

# Plot the predicted values
plt.plot(df['pred'], 'r', label='pred')
plt.plot(df['new_pred'], 'g', label='new_pred')
plt.plot(df['^DJI'], 'k', label='index')
plt.legend()
plt.show()
