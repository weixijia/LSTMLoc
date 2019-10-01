file_num=1
time_step=10

fileaddress='Smartisan/'+str(file_num)+'.xml'
pointaddress='Smartisan/'+str(file_num)+'.csv'
savefile='test_'+str(file_num)+'_timestep'+str(time_step)+'.csv'

import xml.etree.ElementTree as ET
import math
import numpy as numpy
import numpy as np
import pandas as pandas
import pandas as pd
tree=ET.parse(fileaddress)
root=tree.getroot()
read_limit = len(root)

data=[]
for i in range(read_limit):
    if root[i].tag=='a':
        data.append(root[i])  
    elif root[i].tag=='g':
        data.append(root[i])
    elif root[i].tag=='m':
        data.append(root[i])
root=data
read_limit=len(root)

if len(root) < 1:
    print('Xml file is empty.')
    exit(-1)
a_st = {}
g_st = {}
m_st = {}
max_st = 0

for i in range(read_limit):  
    st = int((float(root[i].attrib['st'])-float(root[0].attrib['st']))/1e6)
    if root[i].tag == 'a':
        a_st[st] = i
        if st > max_st:
            max_st = st
    elif root[i].tag == 'g':
        g_st[st] = i
        if st > max_st:
            max_st = st
    elif root[i].tag == 'm':
        m_st[st] = i
        if st > max_st:
            max_st = st
            
st=[]
ax=[]
ay=[]
az=[]
gx=[]
gy=[]
gz=[]
mx=[]
my=[]
mz=[]

for i in range(max_st+1):
    st = numpy.append(st, i, axis=None)
    if i in a_st:
        ax = numpy.append(ax, float(root[a_st[i]].attrib['x']), axis=None)
        ay = numpy.append(ay, float(root[a_st[i]].attrib['y']), axis=None)
        az = numpy.append(az, float(root[a_st[i]].attrib['z']), axis=None)
    else:
        ax = numpy.append(ax, numpy.NaN, axis=None)
        ay = numpy.append(ay, numpy.NaN, axis=None)
        az = numpy.append(az, numpy.NaN, axis=None)
        
    if i in g_st:
        gx = numpy.append(gx, float(root[g_st[i]].attrib['x']), axis=None)
        gy = numpy.append(gy, float(root[g_st[i]].attrib['y']), axis=None)
        gz = numpy.append(gz, float(root[g_st[i]].attrib['z']), axis=None)
    else:
        gx = numpy.append(gx, numpy.NaN, axis=None)
        gy = numpy.append(gy, numpy.NaN, axis=None)
        gz = numpy.append(gz, numpy.NaN, axis=None)
        
    if i in m_st:
        mx = numpy.append(mx, float(root[m_st[i]].attrib['x']), axis=None)
        my = numpy.append(my, float(root[m_st[i]].attrib['y']), axis=None)
        mz = numpy.append(mz, float(root[m_st[i]].attrib['z']), axis=None)
    else:
        mx = numpy.append(mx, numpy.NaN, axis=None)
        my = numpy.append(my, numpy.NaN, axis=None)
        mz = numpy.append(mz, numpy.NaN, axis=None)

df = pd.DataFrame(data=st,columns=['st'])
df['ax'] = ax
df['ay'] = ay
df['az'] = az
df['gx'] = gx
df['gy'] = gy
df['gz'] = gz
df['mx'] = mx
df['my'] = my
df['mz'] = mz

df=df.drop_duplicates(subset='st', keep='first', inplace=False)

df['AccTotal'] = numpy.sqrt(df['ax']**2+df['ay']**2+df['az']**2)
df['GyrTotal'] = numpy.sqrt(df['gx']**2+df['gy']**2+df['gz']**2)
df['MagTotal'] = numpy.sqrt(df['mx']**2+df['my']**2+df['mz']**2)

df=df.interpolate(method='linear', axis=0, limit=None, inplace=False, limit_direction='forward', limit_area=None, downcast=None)

for i in range(1000):
    if not math.isnan(df['gx'][i]):
        df['gx'][0]=df['gx'][i]
        df['gy'][0]=df['gy'][i]
        df['gz'][0]=df['gz'][i]
        df['GyrTotal'][0]=df['GyrTotal'][i]
        g_first_num=i
        break
for j in range(1000):
    if not math.isnan(df['ax'][j]):
        df['ax'][0]=df['ax'][j]
        df['ay'][0]=df['ay'][j]
        df['az'][0]=df['az'][j]
        df['AccTotal'][0]=df['AccTotal'][j]
        a_first_num=j
        break
for k in range(1000):
    if not math.isnan(df['mx'][k]):
        df['mx'][0]=df['mx'][k]
        df['my'][0]=df['my'][k]
        df['mz'][0]=df['mz'][k]
        df['MagTotal'][0]=df['MagTotal'][k]
        m_first_num=k
        break

df=df.interpolate(method='linear', axis=0, limit=None, inplace=False, limit_direction='forward', limit_area=None, downcast=None)

point = pd.read_csv(pointaddress)
point['Time']=point['Time']*1000

df['lat']=numpy.NaN
df['lng']=numpy.NaN

sequcnce=0
s=0.0
sl=0.0
l_s=0
for i in range (len(df)):
    if sequcnce >len(point)-1:
            break
    if df['st'][i]==point['Time'][sequcnce]:
        df['lat'][i]=point['lat'][sequcnce]
        df['lng'][i]=point['Lng'][sequcnce]
        diff=(point['lat'][sequcnce] - s)/(i-l_s)
        difflng=(point['Lng'][sequcnce] - sl)/(i-l_s)
        counter=1
        sum=s
        suml=sl
        for j in range (l_s+1,i):
            if counter%time_step==0:
                sum=sum+diff*time_step
                suml=suml+difflng*time_step
            df['lat'][j]=sum
            df['lng'][j]=suml
            counter=counter+1
        
        s=point['lat'][sequcnce]
        sl=point['Lng'][sequcnce]
        sequcnce=sequcnce+1
        l_s=i

df=df.drop(df[df.st < point['Time'][0]].index)
df=df.drop(df[df.st > point['Time'][len(point)-1]].index)

df.to_csv(savefile)