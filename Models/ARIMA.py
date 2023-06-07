import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARIMA

np.random.seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore") 

# Read data from HDF5 file into a pandas DataFrame
data_frame = pd.read_hdf('DLdata.h5', 'GoldData')

# Calculate returns for each column
for col in data_frame.columns:
    data_frame[col+'_returns'] = data_frame[col].pct_change(2).fillna(0)

# Create a scaler for standardization
scaler = StandardScaler()

# Set the step size for processing the data
step_size = 30

# Process the data in steps
for i in range(0, len(data_frame), step_size):
    # Select a subset of the data
    df = data_frame.loc[data_frame.index[i:i+step_size], :]
    x = df['Gold_returns']
    x_min = min(x)
    x_max = max(x)

    # Find the best ARIMA model using AIC criterion
    min_aic = np.inf
    best_params = (0, 0, 0)
    
    for p in range(5):
        for d in range(5):
            for q in range(5):
                try:
                    arima_model = ARIMA(x, order=(p, d, q)).fit()
                    if arima_model.aic < min_aic:
                        min_aic = arima_model.aic
                        best_params = (p, d, q)
                except:
                    pass
                
    # Fit the ARIMA model with the best parameters
    arima_model = ARIMA(x, order=best_params).fit()
    print('AIC of ARIMA model:', arima_model.aic)
    print('Params of ARIMA model:', best_params)
    
    # Generate predicted returns using the ARIMA model
    x_pred = arima_model.fittedvalues
    data_frame.loc[df.index, 'Predicted_returns'] = x_pred

# Perform predictions using the ARIMA model
df = data_frame
df['Predicted'] = df.Gold

for j in range(len(df.index)):
    if j < 2: 
        continue
    i = df.index[j]
    prev = df.index[j-2]
    df.loc[i, 'Predicted'] = df.loc[prev, 'Predicted'] * (1 + df.loc[i, 'Predicted_returns'])

# Save the predicted data to HDF5 file
df.to_hdf('DLdata.h5', 'Predicted_ARIMA')

# Plot the actual and predicted data
plt.plot(df['Gold'])
plt.plot(df['Predicted'], label='Prediction')
plt.legend()
plt.show()
