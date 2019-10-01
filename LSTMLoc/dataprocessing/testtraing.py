#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 17:35:34 2018

@author: weixijia
"""

import numpy as np
import numpy as numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, concatenate, LSTM, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model

# fix random seed for reproducibility
numpy.random.seed(7)
time_step=10

filenub=10


test=1
epoch=5
batch_size=100
LR=0.005

if time_step==10:
    if test==1:
        filepath=str(str(filenub)+'_10ms.csv')
    elif test==2:
        filepath=str('2_10ms.csv')
    elif test==3:
        filepath=str('3_10ms.csv')
    elif test==4:
        filepath=str('4_10ms.csv')
    elif test==5:
        filepath=str('5_10ms.csv')
    elif test==6:
        filepath=str('6_10ms.csv')
    elif test==7:
        filepath=str('7_10ms.csv')
    elif test==8:
        filepath=str('8_10ms.csv')
    elif test==9:
        filepath=str('9_10ms.csv')
    elif test==10:
        filepath=str('10_10ms.csv')
    elif test==11:
        filepath=str('11_10ms.csv')
    elif test==12:
        filepath=str('12_10ms.csv')
elif time_step==100:
    if test==1:
        filepath=str('1_100ms.csv')
    elif test==2:
        filepath=str('2_100ms.csv')
    elif test==3:
        filepath=str('3_100ms.csv')
    elif test==4:
        filepath=str('4_100ms.csv')
    elif test==5:
        filepath=str('5_100ms.csv')
    elif test==6:
        filepath=str('6_100ms.csv')
    elif test==7:
        filepath=str('7_100ms.csv')
    elif test==8:
        filepath=str('8_100ms.csv')
    elif test==9:
        filepath=str('9_100ms.csv')
    elif test==10:
        filepath=str('10_100ms.csv')
    elif test==11:
        filepath=str('11_10ms.csv')
    elif test==12:
        filepath=str('12_100ms.csv')
# load the dataset
dataframe = pandas.read_csv(filepath, usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15], engine='python',skipfooter=0)
skipfooter = len(dataframe)-((len(dataframe)//time_step)*time_step)
dataframe = pandas.read_csv(filepath, usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15], engine='python',skipfooter=skipfooter)
dataset = dataframe.values
dataset = dataset.astype('float64')
sample_num=dataframe.shape[0]//time_step
if time_step==1:
    lat=np.array(dataframe['lat']).reshape(-1, 1)
    lng=np.array(dataframe['lng']).reshape(-1, 1)
else:
    lat=(dataframe.lat.unique())[:(dataframe.shape[0]//time_step)].reshape(-1, 1)
    lng=(dataframe.lng.unique())[:(dataframe.shape[0]//time_step)].reshape(-1, 1)
location=numpy.column_stack((lat,lng))
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
lat = scaler.fit_transform(lat)
lng = scaler.fit_transform(lng)
location=scaler.fit_transform(location)
sensordata = dataset[:,0:(dataframe.shape[1]-2)]#get acc,gyr,mag
SensorTrain=numpy.reshape(sensordata, ((dataframe.shape[0]//time_step),time_step,(dataframe.shape[1]-2)))



#build 2d model
model_2d = Sequential()
model_2d.add(LSTM(128, input_shape=(SensorTrain.shape[1], SensorTrain.shape[2])))
model_2d.add(Dense(2))
model_2d.compile(optimizer=RMSprop(LR), loss='mse', metrics=['acc'])
model_2d.fit(SensorTrain, location, epochs=epoch, batch_size=batch_size, verbose=2)




locPrediction = model_2d.predict(SensorTrain,batch_size=batch_size)

#get the average prediction result
def moving_average(x, n, type='simple'):
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a

avelatPrediction=locPrediction[:,0]
avelngPrediction=locPrediction[:,1]
#ave2DlatPrediction=locPrediction[:,0]
#ave2DlngPrediction=locPrediction[:,1]
avelatPrediction = moving_average(avelatPrediction, 100, 'simple')
avelngPrediction = moving_average(avelngPrediction, 100, 'simple')


# Make an example plot with two subplots...
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.plot(dataframe['GyrTotal'])
#ax1.set_title('raw prediction')

ax2 = fig.add_subplot(2,2,3)
ax2.plot(location[:,0],location[:,1])
ax2.plot(avelatPrediction,avelngPrediction)
#ax2.set_title('ave_100 prediction')

ax3 = fig.add_subplot(2,2,2)
ax3.plot(dataframe['AccTotal'])

ax4 = fig.add_subplot(2,2,4)
ax4.plot(dataframe['MagTotal'])
# Save the full figure...
fig.savefig(str(filenub)+'_10ms.pdf')

