# Data Preprocessing
# Mengimport  librari
import numpy as np #numpy untuk komputasi
import matplotlib.pyplot as plt #matplotlib untuk visualisasi
import pandas as pd #pandas untuk manipulasi dataset
# Mengimport data untuk dijadikan training set
dataset_train = pd.read_csv('dataset-training.csv')
training_set = dataset_train.iloc[:, 1:2].values #menggunakan kolom open & high
dataset_train.head()
# melakukan scalling data diantara 0 hingga 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# membuat 3D array dengan x_train, 60 timestep, satu fitur
X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN
# Imporr keras
from keras.models import Sequential # menggunakan model neural network berupa sequential network
from keras.layers import Dense # menjalankan full connection nn kita
from keras.layers import LSTM # menambahkan layer lstm
from keras.layers import Dropout #mencegah overfitting
# Initialising the RNN
regressor = Sequential()
# menambah layer lstm dengan 50 hidden unit dengan mengembalikan hidden output pada input time step, input shape sebagai training dataset kita
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) #melakukan drop pada data input sebanyak 20%
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
# Adding the output layer
regressor.add(Dense(units = 1)) #sigmoid
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') #menggunakan optimasi adam
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Part 3 - Making the predictions and visualising the results
# import data real untuk digunakan prediksi
dataset_test = pd.read_csv('dataset-test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
# membuat prediksi
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# menampilkan hasil menggunakan matplotlib
plt.plot(real_stock_price, color = 'green', label = 'Real Harga Saham')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted Harga Saham')
plt.title('Prediksi Harga Saham')
plt.xlabel('Time')
plt.ylabel('Harga Saham')
plt.legend()
plt.show()
