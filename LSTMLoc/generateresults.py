#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 19:41:32 2018

@author: weixijia
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import math
from math import sqrt
import tensorflow as tf
import functions
import json
from keras.models import Sequential
from keras.layers import Dense, concatenate, LSTM, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from keras.models import model_from_json
from keras.models import load_model
from functions import  inversescaler, PCA_compress, SVD_compress, CDF, SimpleDownsampling, overlapping, LossHistory, BatchTensorBoard, moving_average, load_file, normolization, dataprocessing, get_ave_prediction
# fix random seed for reproducibility
# fix random seed for reproducibility
np.random.seed(7)
time_step=1000
epoch=300
batch_size=100
LR=0.005
average_num=100
DownSample_num=100
compress_num=100

SensorTrain1, location1 = overlapping('1_timestep1000_overlap900.csv',3, time_step)
SensorTrain2, location2 = overlapping('2_timestep1000_overlap900.csv',3, time_step)
SensorTrain3, location3 = overlapping('3_timestep1000_overlap900.csv',3, time_step)
SensorTrain4, location4 = overlapping('4_timestep1000_overlap900.csv',3, time_step)
SensorTrain5, location5 = overlapping('5_timestep1000_overlap900.csv',3, time_step)
SensorTrain6, location6 = overlapping('6_timestep1000_overlap900.csv',3, time_step)
SensorTrain7, location7 = overlapping('7_timestep1000_overlap900.csv',3, time_step)
SensorTrain8, location8 = overlapping('8_timestep1000_overlap900.csv',3, time_step)
SensorTrain9, location9 = overlapping('9_timestep1000_overlap900.csv',3, time_step)
SensorTrain10, location10 = overlapping('10_timestep1000_overlap900.csv',3, time_step)
SensorTrain11, location11 = overlapping('11_timestep1000_overlap900.csv',3, time_step)
SensorTrain12, location12 = overlapping('12_timestep1000_overlap900.csv',3, time_step)
SensorTrain13, location13 = overlapping('13_timestep1000_overlap900.csv',3, time_step)
SensorTrain14, location14 = overlapping('14_timestep1000_overlap900.csv',3, time_step)

SensorTrain1=SimpleDownsampling(SensorTrain1,DownSample_num)
SensorTrain2=SimpleDownsampling(SensorTrain2,DownSample_num)
SensorTrain3=SimpleDownsampling(SensorTrain3,DownSample_num)
SensorTrain4=SimpleDownsampling(SensorTrain4,DownSample_num)
SensorTrain5=SimpleDownsampling(SensorTrain5,DownSample_num)
SensorTrain6=SimpleDownsampling(SensorTrain6,DownSample_num)
SensorTrain7=SimpleDownsampling(SensorTrain7,DownSample_num)
SensorTrain8=SimpleDownsampling(SensorTrain8,DownSample_num)
SensorTrain9=SimpleDownsampling(SensorTrain9,DownSample_num)
SensorTrain10=SimpleDownsampling(SensorTrain10,DownSample_num)
SensorTrain11=SimpleDownsampling(SensorTrain11,DownSample_num)
SensorTrain12=SimpleDownsampling(SensorTrain12,DownSample_num)
SensorTrain13=SimpleDownsampling(SensorTrain13,DownSample_num)
SensorTrain14=SimpleDownsampling(SensorTrain14,DownSample_num)

sen1=SensorTrain1[0:1841,:,:]
sen2=SensorTrain2[0:1831,:,:]
sen3=SensorTrain3[0:2202,:,:]
sen4=SensorTrain4[0:2301,:,:]
sen5=SensorTrain5[0:2161,:,:]
sen6=SensorTrain6[0:2171,:,:]
sen7=SensorTrain7[0:2091,:,:]
sen8=SensorTrain8[0:2181,:,:]
sen9=SensorTrain9[0:2482,:,:]
sen10=SensorTrain10[0:2371,:,:]
sen11=SensorTrain11[0:2051,:,:]
sen12=SensorTrain12[0:2071,:,:]
sen13=SensorTrain13[0:2141,:,:]
sen14=SensorTrain14[0:1731,:,:]

modeldownsampleOL900 = load_model('TS1000LR0.005overlappedmodel.h5')
pre1 = modeldownsampleOL900.predict(sen1,batch_size=batch_size)
pre2 = modeldownsampleOL900.predict(sen2,batch_size=batch_size)
pre3 = modeldownsampleOL900.predict(sen3,batch_size=batch_size)
pre4 = modeldownsampleOL900.predict(sen4,batch_size=batch_size)
pre5 = modeldownsampleOL900.predict(sen5,batch_size=batch_size)
pre6 = modeldownsampleOL900.predict(sen6,batch_size=batch_size)
pre7 = modeldownsampleOL900.predict(sen7,batch_size=batch_size)
pre8 = modeldownsampleOL900.predict(sen8,batch_size=batch_size)
pre9 = modeldownsampleOL900.predict(sen9,batch_size=batch_size)
pre10 = modeldownsampleOL900.predict(sen10,batch_size=batch_size)
pre11 = modeldownsampleOL900.predict(sen11,batch_size=batch_size)
pre12 = modeldownsampleOL900.predict(sen12,batch_size=batch_size)
pre13 = modeldownsampleOL900.predict(sen13,batch_size=batch_size)
pre14 = modeldownsampleOL900.predict(sen14,batch_size=batch_size)


realpre1=inversescaler('1_timestep1000_overlap900.csv', time_step, pre1)
realpre2=inversescaler('2_timestep1000_overlap900.csv', time_step, pre2)
realpre3=inversescaler('3_timestep1000_overlap900.csv', time_step, pre3)
realpre4=inversescaler('4_timestep1000_overlap900.csv', time_step, pre4)
realpre5=inversescaler('5_timestep1000_overlap900.csv', time_step, pre5)
realpre6=inversescaler('6_timestep1000_overlap900.csv', time_step, pre6)
realpre7=inversescaler('7_timestep1000_overlap900.csv', time_step, pre7)
realpre8=inversescaler('8_timestep1000_overlap900.csv', time_step, pre8)
realpre9=inversescaler('9_timestep1000_overlap900.csv', time_step, pre9)
realpre10=inversescaler('10_timestep1000_overlap900.csv', time_step, pre10)
realpre11=inversescaler('11_timestep1000_overlap900.csv', time_step, pre11)
realpre12=inversescaler('12_timestep1000_overlap900.csv', time_step, pre12)
realpre13=inversescaler('13_timestep1000_overlap900.csv', time_step, pre13)
realpre14=inversescaler('14_timestep1000_overlap900.csv', time_step, pre14)

pd.DataFrame(pre1).to_csv("pre1.csv")
pd.DataFrame(pre2).to_csv("pre2.csv")
pd.DataFrame(pre3).to_csv("pre3.csv")
pd.DataFrame(pre4).to_csv("pre4.csv")
pd.DataFrame(pre5).to_csv("pre5.csv")
pd.DataFrame(pre6).to_csv("pre6.csv")
pd.DataFrame(pre7).to_csv("pre7.csv")
pd.DataFrame(pre8).to_csv("pre8.csv")
pd.DataFrame(pre9).to_csv("pre9.csv")
pd.DataFrame(pre10).to_csv("pre10.csv")
pd.DataFrame(pre11).to_csv("pre11.csv")
pd.DataFrame(pre12).to_csv("pre12.csv")
pd.DataFrame(pre13).to_csv("pre13.csv")
pd.DataFrame(pre14).to_csv("pre14.csv")

pd.DataFrame(realpre1).to_csv("realpre1.csv")
pd.DataFrame(realpre2).to_csv("realpre2.csv")
pd.DataFrame(realpre3).to_csv("realpre3.csv")
pd.DataFrame(realpre4).to_csv("realpre4.csv")
pd.DataFrame(realpre5).to_csv("realpre5.csv")
pd.DataFrame(realpre6).to_csv("realpre6.csv")
pd.DataFrame(realpre7).to_csv("realpre7.csv")
pd.DataFrame(realpre8).to_csv("realpre8.csv")
pd.DataFrame(realpre9).to_csv("realpre9.csv")
pd.DataFrame(realpre10).to_csv("realpre10.csv")
pd.DataFrame(realpre11).to_csv("realpre11.csv")
pd.DataFrame(realpre12).to_csv("realpre12.csv")
pd.DataFrame(realpre13).to_csv("realpre13.csv")
pd.DataFrame(realpre14).to_csv("realpre14.csv")