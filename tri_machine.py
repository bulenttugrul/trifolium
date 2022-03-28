# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 12:53:24 2022

@author: HP
"""

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

#%%%%%%%%%%%%
labels = data.labels

data['maovermi']=data['minor']/data['major']
data['area']=data['0']
#data['solidity']=data['area']/data['convexarea']
data['are2per']=(2*np.pi*data['perimeter'])/data['area']
data['fvsa']=data['area']/data['filled_area']
data['varea']=(2*np.pi*data['equivalent_diameter'])/data['area']

data = data.drop(columns=['0','labels','shent','fvsa','major','minor','bboxarea','area','perimeter','convexarea','euler_number','filled_area','equivalent_diameter'])
data.to_csv(working_dir+ 'triful5scale_s.csv')
#%%
df  = pd.read_csv(working_dir+ 'triful5scale_s.csv')
df=df.drop(columns=['Unnamed: 0'])
#%%
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=0)


pipe = make_pipeline(RandomForestClassifier(max_depth=5, random_state=0,
                             n_estimators=100))

#max_depth=5, n_estimators=50, max_features=1

params={'randomforestclassifier__max_depth':[5,10,20],\
      'randomforestclassifier__n_estimators':[50,100],\
      'randomforestclassifier__max_features':[1,5],\
      'randomforestclassifier__criterion':['entropy','gini']
      }
      

clf=GridSearchCV(pipe,param_grid=params,cv=10)
clf.fit( x_train,y_train)

y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#%%
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(n_estimators=200,criterion='gini', random_state=3)
clf = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=200)

prediction = clf.fit(x_train, y_train).predict(x_train)
print(accuracy_score(y_train,prediction))
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



print('%0.3f'%accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, x_test, y_test, cv=2)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
#pip install scikit-plot
import scikitplot as skplt
skplt.estimators.plot_feature_importances(clf,feature_names=df.columns)

plt.show()


#%%


#%%
