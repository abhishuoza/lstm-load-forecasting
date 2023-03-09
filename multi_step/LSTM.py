# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks
import keras
from math import sqrt
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, SimpleRNN

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

plt.style.use('fivethirtyeight')


# To plot predictions with actual values for comparision
def plot_predictions(test, predicted):
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(5)
    plt.plot(test[0:250], color='red', label='True load Value')
    plt.plot(predicted[0:250], color='blue', label='Predicted Load Value')
    plt.title('Comparision of Forecast and Actual load ')
    plt.xlabel('Time Indices')
    plt.ylabel('KWH')
    plt.legend()
    plt.show()


def split_series(series, n_past, n_future):
    #
    # n_past ==> no of past observations
    #
    # n_future ==> no of future observations
    #
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)


# First, we get the data
dataset = pd.read_csv('C:/Users/abhis/Desktop/AU/UGRP/pythonProject1/OH_SGSC.csv')
dataset = dataset.drop(['READING_DATETIME'], axis=1)
dataset = dataset[(dataset["CUSTOMER_ID"] == 10509861)]
dataset = dataset.drop(['CUSTOMER_ID'], axis=1)

# Series to supervised
n_past = 12
n_future = 6
n_features = 56
X_values, Y_values = split_series(dataset.values, n_past, n_future)
print(X_values)
print(Y_values)

# n_predict is the test length used to split Train, Test
n_predict = 4353 - 3169
# Split Train, Test
X_train, Y_train = X_values[:(X_values.shape[0] - n_predict), :, :], Y_values[:(X_values.shape[0] - n_predict), :, 0]
X_test, Y_test = X_values[(X_values.shape[0] - n_predict):, :, :], Y_values[(X_values.shape[0] - n_predict):, :, 0]
print(X_train)
print(Y_test)

# reshaping train and test to feed to LSTM

# Model

"""
E1D1
n_features ==> no of features at each timestep in the data.
"""

encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(50, return_sequences=True, return_state=True, dropout=0.25)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.LSTM(50, return_state=True, dropout=0.25)
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]
#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
#
decoder_l1 = tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.25)(decoder_inputs,
                                                                           initial_state=encoder_states1)
decoder_l2 = tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.25)(decoder_l1, initial_state=encoder_states2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(80, activation='relu'))(decoder_l2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='linear'))(decoder_outputs2)
#
seqtoseq_LSTM = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
seqtoseq_LSTM.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae', metrics=["mape"])
seqtoseq_LSTM.summary()

"""
E2D2
n_features ==> no of features at each timestep in the data.
"""

encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.GRU(50, return_sequences=True, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.GRU(50, return_state=True)
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]
#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
#
decoder_l1 = tf.keras.layers.GRU(50, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
decoder_l2 = tf.keras.layers.GRU(50, return_sequences=True)(decoder_l1, initial_state=encoder_states2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(80, activation='relu'))(decoder_l2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='linear'))(decoder_outputs2)
#
seqtoseq_GRU = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
seqtoseq_GRU.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae', metrics=["mape"])
seqtoseq_GRU.summary()

oneshot_GRU = Sequential()
oneshot_GRU.add(GRU(units=100, return_sequences=True, input_shape=(n_past, n_features)))
oneshot_GRU.add(Dropout(0.25))
oneshot_GRU.add(GRU(units=100))
oneshot_GRU.add(Dropout(0.20))
oneshot_GRU.add(Dense(units=80, activation='relu'))
oneshot_GRU.add(Dropout(0.20))
oneshot_GRU.add(Dense(n_future, activation='linear'))
# oneshot.add(tf.keras.layers.Reshape([n_future, n_features]))
oneshot_GRU.compile(loss='mae', optimizer='Adam', metrics=["mape"])
oneshot_GRU.summary()

oneshot_LSTM = Sequential()
oneshot_LSTM.add(GRU(units=100, return_sequences=True, input_shape=(n_past, n_features)))
oneshot_LSTM.add(Dropout(0.25))
oneshot_LSTM.add(GRU(units=100))
oneshot_LSTM.add(Dropout(0.20))
oneshot_LSTM.add(Dense(units=80, activation='relu'))
oneshot_LSTM.add(Dropout(0.20))
oneshot_LSTM.add(Dense(n_future, activation='linear'))
# oneshot.add(tf.keras.layers.Reshape([n_future, n_features]))
oneshot_LSTM.compile(loss='mae', optimizer='Adam', metrics=["mape"])
oneshot_LSTM.summary()

# Fit and predict
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
callbacks = [callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

history_seqtoseq_LSTM = seqtoseq_LSTM.fit(X_train, Y_train, epochs=200, validation_data=(X_test, Y_test), batch_size=32,
                                          callbacks=callbacks)
history_seqtoseq_GRU = seqtoseq_GRU.fit(X_train, Y_train, epochs=200, validation_data=(X_test, Y_test), batch_size=32,
                                        callbacks=callbacks)
history_oneshot_LSTM = oneshot_LSTM.fit(X_train, Y_train, epochs=200, batch_size=32, validation_data=(X_test, Y_test),
                                        callbacks=callbacks)
history_oneshot_GRU = oneshot_GRU.fit(X_train, Y_train, epochs=200, batch_size=32, validation_data=(X_test, Y_test),
                                      callbacks=callbacks)

pred_seqtoseq_LSTM = seqtoseq_LSTM.predict(X_test)
pred_seqtoseq_LSTM = pred_seqtoseq_LSTM.reshape(pred_seqtoseq_LSTM.shape[0], n_future)
pred_seqtoseq_GRU = seqtoseq_GRU.predict(X_test)
pred_seqtoseq_GRU = pred_seqtoseq_GRU.reshape(pred_seqtoseq_GRU.shape[0], n_future)
pred_oneshot_LSTM = oneshot_LSTM.predict(X_test)
pred_oneshot_GRU = oneshot_GRU.predict(X_test)

# # Inverse Normalise
# predictions = sc.inverse_transform(pred_e1d1)
# final_test_Y = sc.inverse_transform(Y_test)

# Check Error

for j in range(1, n_future + 1):
    print("Time step ", j, ":")
    print("MAPE-seqtoseq_LSTM : ", mean_absolute_percentage_error(Y_test[:, j - 1], pred_seqtoseq_LSTM[:, j - 1]),
          end=", ")
    print("MAPE-seqtoseq_GRU : ", mean_absolute_percentage_error(Y_test[:, j - 1], pred_seqtoseq_GRU[:, j - 1]),
          end=", ")
    print("MAPE-oneshot_LSTM : ", mean_absolute_percentage_error(Y_test[:, j - 1], pred_oneshot_LSTM[:, j - 1]),
          end=", ")
    print("MAPE-oneshot_GRU : ", mean_absolute_percentage_error(Y_test[:, j - 1], pred_oneshot_GRU[:, j - 1]))

print("\nseqtoseq_LSTM mean_absolute_percentage_error : ", mean_absolute_percentage_error(Y_test, pred_seqtoseq_LSTM))
print("seqtoseq_LSTM root_mean_squared_error : ", sqrt(mean_squared_error(Y_test, pred_seqtoseq_LSTM)))
print("\nseqtoseq_GRU mean_absolute_percentage_error : ", mean_absolute_percentage_error(Y_test, pred_seqtoseq_GRU))
print("seqtoseq_GRU root_mean_squared_error : ", sqrt(mean_squared_error(Y_test, pred_seqtoseq_GRU)))
print("\noneshot_LSTM mean_absolute_percentage_error : ", mean_absolute_percentage_error(Y_test, pred_oneshot_LSTM))
print("oneshot_LSTM root_mean_squared_error : ", sqrt(mean_squared_error(Y_test, pred_oneshot_LSTM)))
print("\noneshot_GRU mean_absolute_percentage_error : ", mean_absolute_percentage_error(Y_test, pred_oneshot_GRU))
print("oneshot_GRU root_mean_squared_error : ", sqrt(mean_squared_error(Y_test, pred_oneshot_GRU)))


# Plot

def plot_predictions(test, predicted):
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(5)
    plt.plot(test[0:250], color='red', label='True load Value')
    plt.plot(predicted[0:250], color='blue', label='Predicted Load Value')
    plt.title('Comparision of Forecast and Actual load ')
    plt.xlabel('Time Indices')
    plt.ylabel('KWH')
    plt.legend()
    plt.show()


plot_predictions(Y_test[0], pred_seqtoseq_LSTM[0])
plot_predictions(Y_test[0], pred_seqtoseq_GRU[0])
plot_predictions(Y_test[0], pred_oneshot_LSTM[0])
plot_predictions(Y_test[0], pred_oneshot_GRU[0])

mean = 0
for j in range(0,48):
    for i in range(1,21):
        mean = mean + mean_absolute_percentage_error(Y_test[(48*i)+j], Y_test[(48*(i-1))+j])
mean = mean / (48*20)
print(mean)

for i in range(1,21):
    print(mean_absolute_percentage_error(Y_test[48*i], Y_test[48*(i-1)]))
