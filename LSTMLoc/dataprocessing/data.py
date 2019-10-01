# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 23:26:55 2018

@author: weixijia
"""
from generatedata import generatedata

time_step=1000


for file_num in (3,4,5):
    fileaddress='Smartisan/'+str(file_num)+'.xml'
    pointaddress='Smartisan/'+str(file_num)+'.csv'
    savefile=str(file_num)+'_timestep'+str(time_step)+'.csv'
    generatedata(file_num,time_step,fileaddress,pointaddress,savefile)