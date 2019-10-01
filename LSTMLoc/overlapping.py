#import numpy as np
#import matplotlib.pyplot as plt
#import pandas
#import math
#import tensorflow as tf
#import functions
#from keras.models import Sequential
#from keras.layers import Dense, concatenate, LSTM, TimeDistributed
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
#from keras.optimizers import Adam, RMSprop
#from keras.utils import plot_model
#from keras.callbacks import EarlyStopping, Callback, TensorBoard
#from functions import SimpleDownsampling,  overlapping, LossHistory, BatchTensorBoard, moving_average, load_file, normolization,dataprocessing_overlap, dataprocessing, get_ave_prediction, dataprocessing_stateful
## fix random seed for reproducibility
#np.random.seed(7)
#time_step=1000
#epoch=200
#batch_size=1000
#LR=0.005
#average_num=100
#DownSample_num=100
#
#SensorTrain1=np.load('SensorTrain1.npy')
#SensorTrain2=np.load('SensorTrain2.npy')
#SensorTrain3=np.load('SensorTrain3.npy')
#SensorTrain4=np.load('SensorTrain4.npy')
#SensorTrain5=np.load('SensorTrain5.npy')
#SensorTrain6=np.load('SensorTrain6.npy')
##SensorTrain7=np.load('SensorTrain7.npy')
##SensorTrain8=np.load('SensorTrain8.npy')
##SensorTrain9=np.load('SensorTrain9.npy')
##SensorTrain10=np.load('SensorTrain10.npy')
#SensorTrain11=np.load('SensorTrain11.npy')
#SensorTrain12=np.load('SensorTrain12.npy')
#
#location1=np.load('location1.npy')
#location2=np.load('location2.npy')
#location3=np.load('location3.npy')
#location4=np.load('location4.npy')
#location5=np.load('location5.npy')
#location6=np.load('location6.npy')
##location7=np.load('location7.npy')
##location8=np.load('location8.npy')
##location9=np.load('location9.npy')
##location10=np.load('location10.npy')
#location11=np.load('location11.npy')
#location12=np.load('location12.npy')
#
#SensorTrain1=SimpleDownsampling(SensorTrain1,DownSample_num)
#SensorTrain2=SimpleDownsampling(SensorTrain2,DownSample_num)
#SensorTrain3=SimpleDownsampling(SensorTrain3,DownSample_num)
#SensorTrain4=SimpleDownsampling(SensorTrain4,DownSample_num)
#SensorTrain5=SimpleDownsampling(SensorTrain5,DownSample_num)
#SensorTrain6=SimpleDownsampling(SensorTrain6,DownSample_num)
#SensorTrain11=SimpleDownsampling(SensorTrain11,DownSample_num)
#SensorTrain12=SimpleDownsampling(SensorTrain12,DownSample_num)
##filepath=str('1_timestep1000_overlap900.csv')
##
##train1path=str('1_timestep1000_overlap900.csv')
##train2path=str('2_timestep1000_overlap900.csv')
##train3path=str('3_timestep1000_overlap900.csv')
##train4path=str('4_timestep1000_overlap900.csv')
##train5path=str('5_timestep1000_overlap900.csv')
##train6path=str('6_timestep1000_overlap900.csv')
##train7path=str('7_timestep1000_overlap900.csv')
##train8path=str('8_timestep1000_overlap900.csv')
##train9path=str('9_timestep1000_overlap900.csv')
##train10path=str('10_timestep1000_overlap900.csv')
##train11path=str('11_timestep1000_overlap900.csv')
##train12path=str('12_timestep1000_overlap900.csv')
##valpath=str('13_timestep1000_overlap900.csv')
##testpath=str('14_timestep1000_overlap900.csv')
##
##
##
##location1 = overlapping(train1path, time_step)
##location2 = overlapping(train2path, time_step)
##location3 = overlapping(train3path, time_step)
##location4 = overlapping(train4path, time_step)
##location5 = overlapping(train5path, time_step)
##SensorTrain6, location6 = overlapping(train6path, time_step)
##SensorTrain7, location7 = overlapping(train7path, time_step)
##SensorTrain8, location8 = overlapping(train8path, time_step)
##SensorTrain9, location9 = overlapping(train9path, time_step)
##SensorTrain10, location10 = overlapping(train10path, time_step)
##SensorTrain11, location11 = overlapping(train11path, time_step)
##SensorTrain12, location12 = overlapping(train12path, time_step)
##
##Sensor_val, location_val = overlapping(valpath, time_step)
##Sensor_test, location_test = overlapping(testpath, time_step)
#
#
##np.save('SensorTrain1',SensorTrain1)
##np.save('SensorTrain2',SensorTrain2)
##np.save('SensorTrain3',SensorTrain3)
##np.save('SensorTrain4',SensorTrain4)
##np.save('SensorTrain5',SensorTrain5)
##np.save('SensorTrain6',SensorTrain6)
##np.save('SensorTrain7',SensorTrain7)
##np.save('SensorTrain8',SensorTrain8)
##np.save('SensorTrain9',SensorTrain9)
##np.save('SensorTrain10',SensorTrain10)
##np.save('SensorTrain11',SensorTrain11)
##np.save('SensorTrain12',SensorTrain12)
##
##np.save('location1',location1)
##np.save('location2',location2)
##np.save('location3',location3)
##np.save('location4',location4)
##np.save('location5',location5)
##np.save('location6',location6)
##np.save('location7',location7)
##np.save('location8',location8)
##np.save('location9',location9)
##np.save('location10',location10)
##np.save('location11',location11)
##np.save('location12',location12)
##SensorTrain=np.concatenate((SensorTrain1,SensorTrain2,SensorTrain3,SensorTrain4,SensorTrain5,SensorTrain6,SensorTrain7,SensorTrain8,SensorTrain9,SensorTrain10,SensorTrain11,SensorTrain12),axis=0)
##location=np.concatenate((location1,location2,location3,location4,location5,location6,location7,location8,location9,location10,location11,location12),axis=0)
#
#
#
#    
#
#SensorTrain=np.concatenate((SensorTrain1,SensorTrain2,SensorTrain3,SensorTrain4, SensorTrain5,SensorTrain6),axis=0)
#location=np.concatenate((location1,location2,location3,location4,location5,location6),axis=0)
#
##location=np.reshape(location, ((location.shape[0]//time_step), time_step, location.shape[1]))
##lat=np.reshape(lat, ((lat.shape[0]//time_step), time_step, lat.shape[1]))
##lng=np.reshape(lng, ((lng.shape[0]//time_step), time_step, lng.shape[1]))
#
#model_2d = Sequential()
#model_2d.add(LSTM(256,
#    input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),
#    #return_sequences=True,      # True: output at all steps. False: output as last step.
#    #stateful=True  # True: the final state of batch1 is feed into the initial state of batch2
#))
#
##model_lat.add(LSTM(256, input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]), return_sequences=True))
##model_2d.add(TimeDistributed(Dense(2)))
#model_2d.add(Dense(2))
#
#model_2d.compile(optimizer=RMSprop(lr=LR),
#                 loss='mse',metrics=['acc'])
#
#history = model_2d.fit(SensorTrain, location,
#                       validation_data=(SensorTrain12,location12),
#                       epochs=epoch, batch_size=batch_size, verbose=1,
#                       #shuffle=False,
#                       callbacks=[#TensorBoard(log_dir='Tensorboard/epoch'),
#                                  EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')]
#                       )
#
#locPrediction = model_2d.predict(SensorTrain11,batch_size=batch_size)
#
#aveLocPrediction = get_ave_prediction(locPrediction, average_num)
#
#
##
##
## Make an example plot with two subplots...
#fig = plt.figure()
#
#ax1 = fig.add_subplot(2,2,1)
#ax1.plot(location[:,0],location[:,1])
#ax1.plot(locPrediction[:,0],locPrediction[:,1])
#ax1.set_title('raw prediction')
#
#ax2 = fig.add_subplot(2,2,2)
#ax2.plot(location[:,0],location[:,1])
#ax2.plot(aveLocPrediction[:,0],aveLocPrediction[:,1])
#ax2.set_title('ave_'+str(average_num)+'_prediction')
## Save the full figure...
#fig.savefig('downsample_overlap_time_step='+str(time_step)+'.pdf')
#aaa=SensorTrain1
#SensorTrain1=np.load('SensorTrain1.npy')
#aa=np.concatenate([aaa[0,:,9],aaa[1,:,9],aaa[2,:,9],aaa[3,:,9],aaa[4,:,9],aaa[5,:,9],aaa[6,:,9]])
#gg=np.concatenate([aaa[0,:,10],aaa[1,:,10],aaa[2,:,10],aaa[3,:,10],aaa[4,:,10],aaa[5,:,10],aaa[6,:,10]])
#mm=np.concatenate([aaa[0,:,11],aaa[1,:,11],aaa[2,:,11],aaa[3,:,11],aaa[4,:,11],aaa[5,:,11],aaa[6,:,11]])
#a=np.concatenate([SensorTrain1[0,:,9],SensorTrain1[1,:,9],SensorTrain1[2,:,9],SensorTrain1[3,:,9],SensorTrain1[4,:,9],SensorTrain1[5,:,9],SensorTrain1[6,:,9]])
#g=np.concatenate([SensorTrain1[0,:,10],SensorTrain1[1,:,10],SensorTrain1[2,:,10],SensorTrain1[3,:,10],SensorTrain1[4,:,10],SensorTrain1[5,:,10],SensorTrain1[6,:,10]])
#m=np.concatenate([SensorTrain1[0,:,11],SensorTrain1[1,:,11],SensorTrain1[2,:,11],SensorTrain1[3,:,11],SensorTrain1[4,:,11],SensorTrain1[5,:,11],SensorTrain1[6,:,11]])

import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import json
import tensorflow as tf
import functions
from keras.models import Sequential
from keras.layers import Dense, concatenate, LSTM, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from functions import PCA_compress, SVD_compress,SimpleDownsampling, overlapping, LossHistory, BatchTensorBoard, moving_average, load_file, normolization,dataprocessing_overlap, dataprocessing, get_ave_prediction, dataprocessing_stateful
# fix random seed for reproducibility
np.random.seed(7)
time_step=1000
epoch=100
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

#SensorTrain1=SVD_compress(SensorTrain1,compress_num)
#SensorTrain2=SVD_compress(SensorTrain2,compress_num)
#SensorTrain3=SVD_compress(SensorTrain3,compress_num)
#SensorTrain4=SVD_compress(SensorTrain4,compress_num)
#SensorTrain5=SVD_compress(SensorTrain5,compress_num)
#SensorTrain6=SVD_compress(SensorTrain6,compress_num)
#SensorTrain7=SVD_compress(SensorTrain7,compress_num)
#SensorTrain8=SVD_compress(SensorTrain8,compress_num)
#SensorTrain9=SVD_compress(SensorTrain9,compress_num)
#SensorTrain10=SVD_compress(SensorTrain10,compress_num)
#SensorTrain11=SVD_compress(SensorTrain11,compress_num)
#SensorTrain12=SVD_compress(SensorTrain12,compress_num)

#SensorTrain1=SVD_compress(SensorTrain1)
#SensorTrain2=SVD_compress(SensorTrain2)
#SensorTrain3=SVD_compress(SensorTrain3)
#SensorTrain4=SVD_compress(SensorTrain4)
#SensorTrain5=SVD_compress(SensorTrain5)
#SensorTrain6=SVD_compress(SensorTrain6)
#SensorTrain7=SVD_compress(SensorTrain7)
#SensorTrain8=SVD_compress(SensorTrain8)
#SensorTrain9=SVD_compress(SensorTrain9)
#SensorTrain10=SVD_compress(SensorTrain10)
#SensorTrain11=SVD_compress(SensorTrain11)
#SensorTrain12=SVD_compress(SensorTrain12)


#SensorTrain=np.concatenate((SensorTrain1,SensorTrain2,SensorTrain3,SensorTrain4, SensorTrain5),axis=0)
#location=np.concatenate((location1,location2,location3,location4,location5),axis=0)
#
#Sensor_val=np.concatenate((SensorTrain9,SensorTrain10,SensorTrain11),axis=0)
#loc_val=np.concatenate((location9,location10,location11),axis=0)

#SensorTrain=np.concatenate((SensorTrain1,SensorTrain2,SensorTrain3,SensorTrain4, SensorTrain5,SensorTrain6,SensorTrain7,SensorTrain8),axis=0)
#location=np.concatenate((location1,location2,location3,location4,location5,location6,location7,location8),axis=0)
#
#Sensor_val=np.concatenate((SensorTrain9,SensorTrain10,SensorTrain11),axis=0)
#loc_val=np.concatenate((location9,location10,location11),axis=0)
##
#
SensorTrain=np.concatenate((SensorTrain1,SensorTrain2,SensorTrain3,SensorTrain4, SensorTrain5,SensorTrain6,SensorTrain7,SensorTrain8,SensorTrain9),axis=0)
location=np.concatenate((location1,location2,location3,location4,location5,location6,location7,location8,location9),axis=0)


Sensor_val=np.concatenate((SensorTrain10,SensorTrain11,SensorTrain12,SensorTrain13),axis=0)
loc_val=np.concatenate((location10,location11,location12,location13),axis=0)

#location=np.reshape(location, ((location.shape[0]//time_step), time_step, location.shape[1]))
#lat=np.reshape(lat, ((lat.shape[0]//time_step), time_step, lat.shape[1]))
#lng=np.reshape(lng, ((lng.shape[0]//time_step), time_step, lng.shape[1]))

############################################stateless
#model_2d = Sequential()
#model_2d.add(LSTM(
#    input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),
#    units=128,
#    #return_sequences=True,      # True: output at all steps. False: output as last step.
#    #stateful=True  # True: the final state of batch1 is feed into the initial state of batch2
#))
#
##model_lat.add(LSTM(256, input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]), return_sequences=True))
##model_2d.add(TimeDistributed(Dense(2)))
#model_2d.add(Dense(2))
#
#model_2d.compile(optimizer=RMSprop(LR),
#                 loss='mse',metrics=['acc'])
#
#history = model_2d.fit(SensorTrain, location,
#                       validation_data=(Sensor_val,loc_val),
#                       epochs=epoch, batch_size=batch_size, verbose=1,
#                       #shuffle=False,
#                       callbacks=[TensorBoard(log_dir='Tensorboard/svd100'),
#                                  #EarlyStopping(monitor='val_loss', patience=40, verbose=1, mode='min')
#                                  ]
#                       )

############################################stateless


############################################stateful code for first 12 rounds of data,1-11 for training, 12 for validation
#SensorTrain=SensorTrain[0:23790,:,:]
#location=location[0:23790,:]
#Sensor_val=Sensor_val[0:2080,:,:]
#loc_val=loc_val[0:2080,:]
#
#SensorTrain12=SensorTrain12[0:2080,:,:]
#location12=location12[0:2080,:]
#
#stateful_batchsize=10
model_2d = Sequential()
model_2d.add(LSTM(
    input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),
    units=128,
    #return_sequences=True,      # True: output at all steps. False: output as last step.
    #stateful=True  # True: the final state of batch1 is feed into the initial state of batch2
))

#model_lat.add(LSTM(256, input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]), return_sequences=True))
#model_2d.add(TimeDistributed(Dense(2)))
model_2d.add(Dense(2))

model_2d.compile(optimizer=RMSprop(LR),
                 loss='mse',metrics=['acc'])

history = model_2d.fit(SensorTrain, location,
                       validation_data=(Sensor_val,loc_val),
                       epochs=epoch, batch_size=batch_size, verbose=1,
                       #shuffle=False,
                       callbacks=[TensorBoard(log_dir='Tensorboard/svd100'),
                                  #EarlyStopping(monitor='val_loss', patience=40, verbose=1, mode='min')
                                  ]
                       )
############################################stateful
locPrediction = model_2d.predict(SensorTrain14,batch_size=batch_size)

aveLocPrediction = get_ave_prediction(locPrediction, average_num)



#
#
# Make an example plot with two subplots...
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.plot(location[:,0],location[:,1])
ax1.plot(locPrediction[:,0],locPrediction[:,1])
ax1.set_title('raw prediction')

ax2 = fig.add_subplot(2,2,2)
ax2.plot(location[:,0],location[:,1])
ax2.plot(aveLocPrediction[:,0],aveLocPrediction[:,1])
ax2.set_title('ave_'+str(average_num)+'_prediction')
# Save the full figure...
fig.savefig('overlap_time_step='+str(time_step)+'overlappedmodel.pdf')

model_2d.save('TS'+str(time_step)+'LR'+str(LR)+'overlappedmodel.h5')
print("Saved model to disk")



with open('TS'+str(time_step)+'LR'+str(LR)+'overlappedmodel.json', 'w') as fp:
    json.dump(history.history, fp)
    print("Saved history to disk")




