import matplotlib.pyplot as plt
import numpy as np
import os
working_dir = r'E:\\BUYUK_2021\\MASAUSTU\\resim1\\OT\UCGUL\\'
os.chdir(working_dir)
import pandas as pd
scale = 0.00611
data = pd.read_csv('trif5.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data['0']=data['0']*scale*scale
data['perimeter'] = data['perimeter']*scale
data['bboxarea'] = data['bboxarea']*scale*scale
data['convexarea'] = data['convexarea']*scale*scale
data['equivalent_diameter']=data['equivalent_diameter']*scale
data['filled_area'] = data['filled_area']*scale*scale
data['major']=data['major']*scale
data['minor']=data['minor']*scale
data.to_csv('triful5scale.csv')
datadesc,feat = [],['0','perimeter','bboxarea','convexarea','equivalent_diameter','filled_area','major','minor']
for i in range(0,5):
    for l in feat:
        df = data[data['labels']==i][l].describe()
        print( df)
        datadesc.append(df)
dat=pd.DataFrame(datadesc)
dat.to_csv(working_dir +'describe.csv')
#data = data[data['0']>25000]
#%%
#data= data[(data.labels == 0 )|( data.labels == 1)|( data.labels == 2)|( data.labels == 5) ]
#data=data.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
labels = data.labels

data['maovermi']=data['minor']/data['major']
data['area']=data['0']
#data['solidity']=data['area']/data['convexarea']
data['are2per']=(2*np.pi*data['perimeter'])/data['area']
data['fvsa']=data['area']/data['filled_area']
data['varea']=(2*np.pi*data['equivalent_diameter'])/data['area']
data = data.drop(columns=['0','shent','labels','fvsa','major','minor','bboxarea','area','perimeter','convexarea','euler_number','filled_area','equivalent_diameter'])

