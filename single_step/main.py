# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizer_v1 import Adam
from keras import callbacks
from pandas import concat, DataFrame


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, RNN, SimpleRNN
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from tensorflow.keras.utils import plot_model

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
#
#
# def return_rmse(test, predicted):
#     rmse = math.sqrt(mean_squared_error(test, predicted))
#     print("The root mean squared error is {}.".format(rmse))


# convert series to supervised learning
def series_to_supervised(data, n_lag=1, n_lead=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, … t-1)
    for i in range(n_lag, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, … t+n)
    for i in range(0, n_lead):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# First, we get the data
dataset = pd.read_csv('C:/Users/abhis/Desktop/AU/UGRP/pythonProject1/OH_SGSC.csv')
dataset = dataset.drop(['READING_DATETIME'], axis=1)
dataset = dataset[(dataset["CUSTOMER_ID"] == 10509861)]
dataset = dataset.drop(['CUSTOMER_ID'], axis=1)

dataset_scaled = dataset
# Normalistion
# dataset_scaled = dataset
# sc = MinMaxScaler(feature_range=(0, 1))
# dataset_scaled.loc[:, ['GENERAL_SUPPLY_KWH']] = sc.fit_transform(dataset_scaled.loc[:, ['GENERAL_SUPPLY_KWH']])
# print(dataset_scaled)

# Series to supervised
reframed = series_to_supervised(dataset_scaled, 12)
Y = reframed['var1(t)']
for i in range(56):
        reframed.pop(reframed.columns.values[-1])
X = reframed


print(X)
print(Y)

X_values = X.values
Y_values = Y.values

# n_predict is the test length
n_predict = 4369 - 3169
# Split Train, Test
train_X, train_Y = X_values[:(X_values.shape[0] - n_predict), :], Y_values[:(X_values.shape[0] - n_predict)]
test_X, test_Y = X_values[(X_values.shape[0] - n_predict):, :], Y_values[(X_values.shape[0] - n_predict):]
print(train_X)

# reshaping train and test to feed to LSTM
print ("----")
timestamps = 96
features = int(train_X.shape[1] / timestamps)
print(features)


train_X = train_X.reshape((train_X.shape[0], timestamps, features))
test_X = test_X.reshape((test_X.shape[0], timestamps, features))



print(train_X)

model = Sequential()
model.add(GRU(units=100, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.15))
model.add(GRU(units=100))
model.add(Dropout(0.10))
model.add(Dense(units=80, activation='relu'))
model.add(Dropout(0.10))
model.add(Dense(1, activation='linear'))
model.compile(loss='mae', optimizer='Adam', metrics=["mape"])
print(model.summary())

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#
#
# # The LSTM architecture
# regressor = Sequential()
# # First LSTM layer with Dropout regularisation
# regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
# regressor.add(Dropout(0.2))
# # Second LSTM layer
# regressor.add(LSTM(units=50, return_sequences=True))
# regressor.add(Dropout(0.2))
# # Third LSTM layer
# regressor.add(LSTM(units=50, return_sequences=True))
# regressor.add(Dropout(0.2))
# # Fourth LSTM layer
# regressor.add(LSTM(units=50))
# regressor.add(Dropout(0.2))
# # The output layer
# regressor.add(Dense(units=1))
#
# # Compiling the RNN
# regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
# # Fitting to the training set
# regressor.fit(X_train,y_train,epochs=50,batch_size=32)
callbacks = [callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

history = model.fit(train_X, train_Y, epochs=200, batch_size=32, validation_data=(test_X, test_Y), callbacks = callbacks)

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 0.4])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MAPE]')
  plt.legend()
  plt.grid(True)

plot_loss(history)

yhat = model.predict(test_X)
print(yhat)
yhat = np.reshape(yhat, (1200, 1))
final_test_Y = np.reshape(test_Y, (1200, 1))
yhat_df = pd.DataFrame(yhat)
# yhat_df.to_csv('C:/Users/abhis/Desktop/AU/UGRP/pythonProject1/single_step/predictions_normalised_10006414.csv')
# nsamples, nx, ny = dataset.shape
# d2_train_dataset = dataset.reshape((nsamples,nx*ny))

predictions = yhat
# Normalisation
# predictions = sc.inverse_transform(yhat)
# final_test_Y = sc.inverse_transform(final_test_Y)
print(predictions)
plot_predictions(final_test_Y, predictions)
mape = mean_absolute_percentage_error(final_test_Y, predictions)
rmse = sqrt(mean_squared_error(final_test_Y, predictions))
print("The mean absolute percentage error is {}.".format(mape))
print("The root mean square error is {}.".format(rmse))
