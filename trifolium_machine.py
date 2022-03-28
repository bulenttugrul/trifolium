import numpy as np
import os
working_dir = r'E:\\BUYUK_2021\\MASAUSTU\\resim1\\OT\UCGUL\\'
os.chdir(working_dir)
import pandas as pd
data = pd.read_csv('trif5.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
#data = data[data['0']>25000]

#data= data[(data.labels == 0 )|( data.labels == 1)|( data.labels == 2)|( data.labels == 5) ]
#data=data.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
labels = data.labels

data['maovermi']=data['minor']/data['major']
data['area']=data['0']
data['solidity']=data['area']/data['convexarea']
data['are2per']=(2*np.pi*data['perimeter'])/data['area']
data['fvsa']=data['area']/data['filled_area']
data['varea']=(2*np.pi*data['equivalent_diameter'])/data['area']
data = data.drop(columns=['0','labels','fvsa','major','minor','bboxarea','area','perimeter','convexarea','euler_number','filled_area','equivalent_diameter'])


import numpy as np
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

#%%%
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
pipe = make_pipeline(RandomForestClassifier(max_depth=5, random_state=0,
                             n_estimators=100))

#max_depth=5, n_estimators=50, max_features=1

params={'randomforestclassifier__max_depth':[10],\
      'randomforestclassifier__n_estimators':[50],\
      'randomforestclassifier__max_features':[2],\
      'randomforestclassifier__criterion':['gini']
      }
      

clf=GridSearchCV(pipe,param_grid=params,cv=5)
clf.fit( x_train,y_train)

y_pred = clf.predict(x_test)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
#%%%

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier




clf = ExtraTreesClassifier(n_estimators=61,criterion='entropy', random_state=3)
clf = DecisionTreeClassifier()
clf = RandomForestClassifier(max_depth=5, n_estimators=50, max_features=1)
clf = KNeighborsClassifier()
clf = GaussianNB()
clf = MLPClassifier(random_state=1,max_iter=500,activation='relu',learning_rate_init=0.003)

from sklearn.metrics import accuracy_score
prediction = clf.fit(x_train, y_train).predict(x_train)

print(accuracy_score(y_train,prediction))

y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
#%%%
