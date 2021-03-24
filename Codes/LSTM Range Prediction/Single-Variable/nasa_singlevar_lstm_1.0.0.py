import os
import pandas
import matplotlib.pyplot as plt
import numpy
from numpy import array
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", type=str, default="", help="datafile in text format")
ap.add_argument("-c", "--cols", type=int, default=0, help="column in datafile")
ap.add_argument("-e", "--epochs", type=int, default=5, help="number of epochs")
ap.add_argument("-i", "--in", type=int, default=100, help="size of input vector")
ap.add_argument("-o", "--out", type=int, default=100, help="size of output vector")
ap.add_argument("-b", "--batch", type=int, default=32, help="batch_size")
ap.add_argument("-m", "--model", type=str, default="default.h5", help="name of model file")
ap.add_argument("-t", "--outfile", type=str, default="default.txt", help="name of output file")
args = vars(ap.parse_args())

F = args["file"]
C = args["cols"]
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

def load_col(txtfile, col, process):
    dataset = pandas.read_csv(txtfile, sep='\t',
                               header=None,usecols=[col])
    if process==True:
        #dataset = dataset.values[2:]
        dataset = dataset.astype('float32')
        scaler = MinMaxScaler(feature_range=(0,1))
        dataset = scaler.fit_transform(dataset)
        return dataset, scaler
    elif process==False:
        return dataset

# Load data from text file with selected columns, scaled to (0,1)
data_array, data_scaler = load_col(F, C, True)
print(data_array[0:10])

# Fix random seed for reproducibility
numpy.random.seed(7)

n_steps_in = I
n_steps_out = O
dataset = data_array
n_features = 1

print("preparing data...")
X_dataset, y_dataset = split_sequence(data_array, n_steps_in, n_steps_out)
y_dataset = y_dataset.reshape((y_dataset.shape[0], y_dataset.shape[1]))
print("done!")
    
print("preparing model...")
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu', return_sequences=False))
model.add(Dense(n_steps_out))
model.compile(loss='mse', optimizer='adam')
print("done!")

print("training model...")
history = model.fit(X_dataset, y_dataset, epochs=E, batch_size=B, verbose=1)
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