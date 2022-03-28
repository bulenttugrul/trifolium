from skimage import  measure
from skimage.measure import regionprops
from skimage.measure import label
from skimage import  filters
import matplotlib.pyplot as plt
import numpy as np
import cv2
def calculate(img):
    im = filters.gaussian(img, sigma=15)
    blobs = im > im.mean()-0.2 #ot icin mean() yeterli
    #all_labels = measure.label(blobs)
    blobs_labels = measure.label(blobs, background=1)

   
    props = measure.regionprops(blobs_labels) 
    return props

def image_colorfulness(image):	
	(B, G, R) = cv2.split(image.astype("float"))
	rg = np.absolute(R - G)
	yb = np.absolute(0.5 * (R + G) - B)
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	return stdRoot + (0.3 * meanRoot)    
#%%

import os
import pandas as pd
from skimage.measure import shannon_entropy
#working_dir = 'C:\\Users\\pc\\Desktop\\resim1\\OT\\UCGUL\\'
working_dir = r'E:\\BUYUK_2021\\MASAUSTU\\resim1\\OT\UCGUL\\'
os.chdir(working_dir)
folders, labels,clrfulness = ['TrifoliumRepens'], [],[]
for folder in folders:
    print (folder)
    for path in os.listdir(os.getcwd()+'\\'+folder):
        img = cv2.imread(folder+'/'+path)
        clr = image_colorfulness(img)
        clrfulness.append(clr)
        labels.append(folders.index(folder))
        
clrdata =pd.DataFrame(clrfulness)  
clrdata.columns = ['clrful']
clrdata.to_csv('TrifoliumRepensclr.csv')     
#%%
import os
from skimage.measure import shannon_entropy
#working_dir = 'C:\\Users\\pc\\Desktop\\resim1\\OT\\UCGUL\\'
working_dir = r'E:\\BUYUK_2021\\MASAUSTU\\resim1\\OT\UCGUL\\'
os.chdir(working_dir)
folders, labels, props, shent = ['TrifoliumRepens','TrifoliumPratense'], [], [],[]
for folder in folders:
    print (folder)
    for path in os.listdir(os.getcwd()+'\\'+folder):
        img = cv2.imread(folder+'/'+path,0)
        prop = calculate(img)
        shent.append(shannon_entropy(img,base=10))
        #if prop[0]['area'] < 2500:
            #props.append(prop)
            
            #print('%d  %d'%(prop[0]['area'],prop[1]['area']))
        #else:
        props.append(prop)
            
            
        labels.append(folders.index(folder))

#%%
import pandas as pd
area, perimeter, bbox_area, convex_area, convex_image, eccentricity  = [],[],[],[],[],[]
equivalent_diameter, euler_number, extent, filled_area, filled_image  =[],[],[],[],[]
minor_axis_length,moments,moments_hu, solidity =[],[],[],[]
major,minor,labell =[],[],[]
for i in range(0,len(props)):
    if props[i][0]['area'] < 10000:
            area.append(props[i][1]['area'])
            perimeter.append(props[i][1]['perimeter'])
            bbox_area.append(props[i][1]['bbox_area'])
            convex_area.append(props[i][1]['convex_area'])
            eccentricity.append(props[i][1]['eccentricity'])
            equivalent_diameter.append(props[i][1]['equivalent_diameter'])
            euler_number.append(props[i][1]['euler_number'])
            extent.append(props[i][1]['extent'])
            filled_area.append(props[i][1]['filled_area'])
            major.append(props[i][1]['major_axis_length'])
            minor.append(props[i][1]['minor_axis_length'])
            solidity.append(props[i][1]['solidity'])
            #print('%d  %d'%(props[i][0]['area'],props[i][1]['area']))
    else:
            area.append(props[i][0]['area'])
            perimeter.append(props[i][0]['perimeter'])
            bbox_area.append(props[i][0]['bbox_area'])
            convex_area.append(props[i][0]['convex_area'])
            eccentricity.append(props[i][0]['eccentricity'])
            equivalent_diameter.append(props[i][0]['equivalent_diameter'])
            euler_number.append(props[i][0]['euler_number'])
            extent.append(props[i][0]['extent'])
            filled_area.append(props[i][0]['filled_area'])
            major.append(props[i][0]['major_axis_length'])
            minor.append(props[i][0]['minor_axis_length'])
            solidity.append(props[i][0]['solidity'])
            labell.append(labels[i])
            
data=pd.DataFrame(area)
data['perimeter']=perimeter
data['bboxarea']=bbox_area
data['convexarea']=convex_area
data['eccentiricty']=eccentricity
data['extent']=extent
data['equivalent_diameter']=equivalent_diameter
data['euler_number']=euler_number
data['filled_area']=filled_area
data['major']=major
data['minor']=minor
data['shent'] = shent
data['solidity']=solidity
data['labels']=labels
#data['labels']=4
data.to_csv('TrifoliumRepens_mart_1.csv')

#%%
import pandas as pd
c=pd.read_csv('TrifoliumAlexandrinum.csv')
os=pd.read_csv('TrifoliumFragiferum.csv')
y1=pd.read_csv('TrifoliumIncarnatum.csv')
pr=pd.read_csv('TrifoliumPratense.csv')
tr=pd.read_csv('TrifoliumRepens.csv')
data=pd.DataFrame(c)
data =data.append(os)
data =data.append(y1)
data =data.append(pr)
data =data.append(tr)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data.to_csv('trifs5.csv')
#df=pd.read_csv('perimeter.csv',usecols=[1])

#%%
df = pd.read_csv('TrifoliumRepens.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
ss = []
name = df
up = name['0'].mean() +  name['0'].std()
down = name['0'].mean()  - name['0'].std()

for x in range(0,len(name)):
    if (name['0'][x] >= down and name['0'][x] <= up ):
        ss.append(name.iloc[x,:])

sdata= pd.DataFrame(ss)
sdata.to_csv('TrifoliumRepenss_1.csv')

