import pandas as pd
import numpy as np
import pandas_datareader as web
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import math
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, LSTM 


data = web.DataReader('SBIN.NS',data_source = 'yahoo', start = '2018-01-01' , end = '2021-01-01')

df = data.filter(['Close'])
dataset = df.values
training_data_len = math.ceil ( len(dataset) * 0.8)

scaler = MinMaxScaler( feature_range= (0,1))

scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[0:training_data_len, :]



x_train = []      
y_train = []      

for i in range(60,len(training_data)):
    x_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train ,(x_train.shape[0],x_train.shape[1],1))

#Building of LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences= True , input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences= False)) #its false because we are not using any more lstm model
model.add(Dense(25))   
model.add(Dense(1)) 

model.compile(optimizer= 'adam', loss='mean_squared_error')

model.fit(x_train ,y_train ,batch_size=1,epochs=1)

#Creating the test dataset
#Creating a new array containing scaled values from index 532 to 739(this len of orignal data set)
test_data = scaled_data[ training_data_len-60: , :]
#Spliting the data set into two parts
x_test = [] #list
y_test = dataset[training_data_len: , :] #this the data excluded intially also these are not scaled

for i in range(60 ,len(test_data)):
  x_test.append(test_data[i-60:i , 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test , (x_test.shape[0],x_test.shape[1],1))

#Geting the predicted values from the model 
predictions = model.predict(x_test)                   #will be in 0 to 1 form
predictions = scaler.inverse_transform(predictions)   #descaled to original


#Plot the data
train = df[:training_data_len]
valid = df[training_data_len:]
valid['Predictions'] = predictions      #adding a column to prediction


import pickle 
pickle_out = open("stpredict.pkl", mode = "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()







