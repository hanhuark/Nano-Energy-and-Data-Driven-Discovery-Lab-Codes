import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy import array, hstack
import matplotlib.pyplot as plt
import os, math
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", type=str, default="", help="datafile in text format")
ap.add_argument("-n", "--num", type=int, default=3, help="number of variables")
ap.add_argument("-e", "--epochs", type=int, default=5, help="number of epochs")
ap.add_argument("-i", "--in", type=int, default=100, help="size of input vector")
ap.add_argument("-o", "--out", type=int, default=100, help="size of output vector")
ap.add_argument("-b", "--batch", type=int, default=32, help="batch_size")
ap.add_argument("-m", "--model", type=str, default="default.h5", help="name of model file")
ap.add_argument("-t", "--outfile", type=str, default="default.txt", help="name of output file")
args = vars(ap.parse_args())

F = args["file"]
N = args["num"]
E = args["epochs"]
I = args["in"]
O = args["out"]
B = args["batch"]
M = args["model"]
T = args["outfile"]

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # Finding the end of the pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        
        # Checking if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
            
        # Gather input and output parts of pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix: out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Fix random seed for reproducibility
np.random.seed(7)

# load data
print("loading data...")
usecols=[0,1,2,3]
dataset = pd.read_csv(F, sep='\t', header=None, usecols=usecols,
                          engine='python', index_col=0)
print("done!")

# manually specify column names
print("preparing data...") 
dataset.columns = ['pressure', 'temperature', 'mass flux']
dataset.index.name = 'time'
# mark all NA values with 0
dataset.fillna(0, inplace=True)

pressure_array = dataset['pressure'].values
temperature_array = dataset['temperature'].values
mass_array = dataset['mass flux'].values
scaler = MinMaxScaler(feature_range=(0,1))
pressure_scaled = scaler.fit_transform(pressure_array.reshape(-1,1))
temperature_scaled = scaler.fit_transform(temperature_array.reshape(-1,1))
mass_scaled = scaler.fit_transform(mass_array.reshape(-1,1))
pressure_scaled = pressure_scaled.reshape((len(pressure_scaled), 1))
temperature_scaled = temperature_scaled.reshape((len(temperature_scaled), 1))
mass_scaled = mass_scaled.reshape((len(mass_scaled), 1))

if N==2:
	full_dataset_scaled = hstack((pressure_scaled, temperature_scaled))
elif N==3:
	full_dataset_scaled = hstack((pressure_scaled, temperature_scaled, mass_scaled))
else:
	full_dataset_scaled = hstack((pressure_scaled, temperature_scaled, mass_scaled))
print("done!")

print(full_dataset_scaled.shape)
print(full_dataset_scaled[0:3])


print("preparing model...")
n_steps_in = I
n_steps_out = O
X, y = split_sequence(full_dataset_scaled, n_steps_in, n_steps_out)
n_features = N

print(X.shape, y.shape)

es = EarlyStopping(patience=4, monitor='loss')

# define model
model = Sequential()
model.add(LSTM(n_steps_in, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(n_steps_in, activation='relu', return_sequences=True))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
model.summary()
print("done!")

print("training model...")
# fit model
history = model.fit(X, y, epochs=E, verbose=1, callbacks=[es], batch_size=B)
model_name = M
model.save(model_name)
print("done!")

loss = history.history['loss']

outputfilename = T

output_file = open(outputfilename, "w")

print("Opened txt file")

output_file.write("loss\n")
for i in range(len(loss)):
    output_file.write(str(loss[i]))
    output_file.write("\n")

print("Closed txt file")