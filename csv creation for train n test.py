#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:04:30 2020

@author: vivek 
"""
#%%
""""import essential packages"""

import pandas as pd
import numpy as np
from osgeo import gdal
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from rotation_forest import RotationForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle
import graphviz
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle

#%%

"""function to return pixel values"""
def value(ds):
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    band1 = ds.GetRasterBand(1).ReadAsArray()
    data = np.array(band1)
    values = []
    for row in range(0,rows):
        for col in range(0,cols):
            values.append(data[row,col])
    return values

"""function to return latitude and longitude"""
def latlong(ds):
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    xoff,a,b,yoff,d,e = ds.GetGeoTransform()
    lat =[]
    long = []
    def pixel2coord(x,y):
        xp = a*y+b*x+a*0.5+b*0.5+xoff
        yp = d*y+e*x+d*0.5+e*0.5+yoff
        lat.append(xp)
        long.append(yp)
    for row in range(0,rows):
        for col in range(0,cols):
            pixel2coord(row,col)
    return lat, long

#%%

"""load images and extract values for features""" 
ASPECT = value(gdal.Open("../extract/aspect/ASPECT.tif"))
BUFFER_FAULT2 = value(gdal.Open("../extract/buffer/distance_2/fault/buffer_fault2.tif"))
BUFFER_RIVER2 = value(gdal.Open("../extract/buffer/distance_2/river/buffer_river2.tif"))
BUFFER_ROAD2 = value(gdal.Open("../extract/buffer/distance_2/road/buffer_road2.tif"))
DEM = value(gdal.Open("../extract/dem/DEM.tif"))
LITHO = value(gdal.Open("../extract/litho/LITHO.tif"))
PLAN_CURVETURE = value(gdal.Open("../extract/plan_curveture/PLAN_CURVETURE.tif"))
PROFILE_CURVETURE = value(gdal.Open("../extract/profile_curveture/PROFILE_CURVETURE.tif"))
SLOPE_LENGTH = value(gdal.Open("../extract/slope_length/SLOPE_LENGTH.tif"))
SLOPEW = value(gdal.Open("../extract/slope/SLOPE.tif"))
SPI= value(gdal.Open("../extract/spi/SPI.tif"))
STI = value(gdal.Open("../extract/sti/STI.tif"))
TWI = value(gdal.Open("../extract/twi/twi_12.tif"))
LAND_COVER= value(gdal.Open("../extract/land_cover/LAND_COVER1.tif"))
LANDSLIDE = value(gdal.Open("../extract/landslide/landslide_2.tif"))

"""calculate latitude, longitude and round to integer"""

lat, long = latlong(gdal.Open("../extract/land_cover/LAND_COVER1.tif"))
lat = [round(x) for x in lat]
long = [round(x) for x in long]

#%%
"""check if total no of data points are equal in all"""
features_len = []
features_len.append(len(ASPECT))
features_len.append(len(BUFFER_FAULT2))
features_len.append(len(BUFFER_RIVER2))
features_len.append(len(BUFFER_ROAD2))
features_len.append(len(DEM))
features_len.append(len(LITHO))
features_len.append(len(PLAN_CURVETURE))
features_len.append(len(PROFILE_CURVETURE))
features_len.append(len(SLOPE_LENGTH))
features_len.append(len(SLOPEW))
features_len.append(len(SPI))
features_len.append(len(STI))
features_len.append(len(TWI))
features_len.append(len(LAND_COVER))
features_len.append(len(LANDSLIDE))
features_len.append(len(lat))
features_len.append(len(long))

print(features_len)

"""if length of features are not same, recheck the data"""

#%%
"""make dataframe from features and create csv"""


d = {'Latitude' :lat, 'Longitude' :long, 'Fault' :BUFFER_FAULT2,
     'River' :BUFFER_RIVER2, 'Road' :BUFFER_ROAD2, 'Lithology' :LITHO, 'Land_Cover' :LAND_COVER,
     'Dem' :DEM, 'Aspect' :ASPECT, 'Plan_Curvature' :PLAN_CURVETURE, 'Profile_Curvature' :PROFILE_CURVETURE, 
     'Slope_Length' :SLOPE_LENGTH, 'Slope' :SLOPEW, 'SPI' :SPI, 'STI' :STI, 'TWI' :TWI, 
      'Landslide' :LANDSLIDE}

df = pd.DataFrame(d, columns =['Latitude', 'Longitude', 'Fault' , 'River' , 'Road' ,'Lithology', 'Land_Cover',
                               'Dem', 'Aspect', 'Plan_Curvature','Profile_Curvature',
                               'Slope_Length', 'Slope', 'SPI', 'STI', 'TWI','Landslide'] )

df.to_csv('../csv/complete_data.csv',index=False)

#%%
df = pd.read_csv('../csv/complete_data.csv')
"""
Now we need to calculate the number of values outside study region( no data values).
They should be equal for all the features
We can either save them in 2D list for each feature or print them out in the 
console


value_counts() function gives the count of each unique entry in a series in 
descending order
"""

#printing value_counts in the console
print(df['Aspect'].value_counts())
print(df['Fault'].value_counts())
print(df['River'].value_counts())
print(df['Road'].value_counts())
print(df['Dem'].value_counts())
print(df['Lithology'].value_counts())
print(df['Plan_Curvature'].value_counts()) 
print(df['Profile_Curvature'].value_counts())
print(df['Slope_Length'].value_counts())
print(df['Slope'].value_counts()) 
print(df['SPI'].value_counts()) 
print(df['STI'].value_counts())
print(df['TWI'].value_counts())
print(df['Land_Cover'].value_counts())
print(df['Landslide'].value_counts() )


#saving them in a list
mode_values = []

for x in df:
    mode_values.append(df[x].value_counts())

"""
analyse value_count for all the features, 
Since number of no data values is more, first value of 
value_counts() function is same for all features, if not check the original 
images"""

"""
assign a variable drop_value as one of the no data value,
here value of buffer fault is being used"""

drop_value = 65536

"""save the index of all the pixels with value as drop_value in a list"""

indexNames = df[df['Fault'] == drop_value].index


"""from the original dataframe, drop all the rows of index in indexNames"""
df.drop(indexNames , inplace = True)


#write the dataframe into a csv file
df.to_csv('../csv/row_truncated_data.csv', index = False)

#%% read truncated data
df = pd.read_csv('../csv/row_truncated_data.csv')
""""multi collinearity check"""

"""vif values between 1-5 have low collinearity
                5-10 , moderate
                >10, high collinearity
                """
df_new = df.iloc[:,2:16]
vif = pd.DataFrame()
vif['vif factor'] = [variance_inflation_factor(df_new.values, i) for i in range(df_new.shape[1])]
vif['features'] = df_new.columns

corr = pd.DataFrame()
corr = df_new.iloc[:,5:]
corr_matrix=corr.corr()
#%%
"""preprocessing - basic layout for on  one hot encoding the data

X = df.iloc[:,2:16]
Y = df.iloc[:,16]


landc = LabelBinarizer().fit_transform(X.LandCover)
litho = LabelBinarizer().fit_transform(X.Lithology)

X = X.drop(['LandCover','Lithology'], axis = 1)
landc = pd.DataFrame(landc)
litho = pd.DataFrame(litho)

X = pd.concat([X,landc,litho],axis=1)
"""
#%% preprocessing 0 - handling profile and plan curvature aspect, all buffer variables"""

i = df['Plan_Curvature']
Plan_Curvature = []
for x in i:
    if(x<0):
        Plan_Curvature.append(0)
    elif(x>0):
        Plan_Curvature.append(1)
    else:
        Plan_Curvature.append(2)

i = df['Profile_Curvature']
Profile_Curvature = []
for x in i:
    if(x<0):
        Profile_Curvature.append(0)
    elif(x>0):
        Profile_Curvature.append(1)
    else:
        Profile_Curvature.append(2)

df = df.drop(['Plan_Curvature','Profile_Curvature'], axis = 1)
Plan_Curvature = pd.Series(Plan_Curvature)
Profile_Curvature = pd.Series(Profile_Curvature)
df = pd.concat([df,Plan_Curvature,Profile_Curvature], axis=1)
df = df.rename(columns={0:'Plan_Curvature',1:'Profile_Curvature'})

i = df['Aspect']
j = []
for x in i:
    if(x<0):
        j.append(0)
    elif(x>=0 and x<22.5):
        j.append(1)
    elif(x>=22.5 and x<67.5):
        j.append(2)
    elif(x>=67.5 and x<112.5):
        j.append(3)
    elif(x>=112.5 and x<157.5):
        j.append(4)
    elif(x>=157.5 and x<202.5):
        j.append(5)
    elif(x>=202.5 and x<247.5):
        j.append(6)
    elif(x>=247.5 and x<292.5):
        j.append(7)
    elif(x>=292.5 and x<337.5):
        j.append(8)
    else:
        j.append(1)
df = df.drop(['Aspect'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'Aspect'})

i = df['Fault']
Fault = []
for x in i:
    if(x==50):
        Fault.append(0)
    elif(x==100):
        Fault.append(1)
    elif(x==150):
        Fault.append(2)
    elif(x==200):
        Fault.append(3)
    elif(x==250):
        Fault.append(4)
    elif(x==300):
        Fault.append(5)
df = df.drop(['Fault'],axis=1)
Fault = pd.Series(Fault)
df = pd.concat([df,Fault], axis=1)
df = df.rename(columns={0:'Fault'})

i = df['River']
j = []
for x in i:
    if(x==40):
        j.append(0)
    elif(x==80):
        j.append(1)
    elif(x==120):
        j.append(2)
    elif(x==160):
        j.append(3)
    elif(x==200):
        j.append(4)
    elif(x==240):
        j.append(5)
df = df.drop(['River'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'River'})

i = df['Road']
j = []
for x in i:
    if(x==40):
        j.append(0)
    elif(x==80):
        j.append(1)
    elif(x==120):
        j.append(2)
    elif(x==160):
        j.append(3)
    elif(x==200):
        j.append(4)
    elif(x==240):
        j.append(5)
df = df.drop(['Road'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'Road'})

#moving landslide in the last
landslide = df['Landslide']
df = df.drop(['Landslide'],axis=1)
df = pd.concat([df,landslide], axis=1)

#saving csv file
df.to_csv('../csv/processed0_truncated_data.csv')

#%% preprocessing 1 - handling DEM derivatives, all buffer variables"""
"""Class (as per Jenks natural break)"""

i = df['Plan_Curvature']
Plan_Curvature = []
for x in i:
    if(x<=-2.543456388):
        Plan_Curvature.append(0)
    elif(x>=-2.543456387 and x<=-0.957240183):
        Plan_Curvature.append(1)
    elif(x>=-0.957240182 and x<=-0.092031344):
        Plan_Curvature.append(2)
    elif(x>=-0.092031343 and x<=0.773177495):
        Plan_Curvature.append(3)
    elif(x>=0.773177496 and x<=2.215192226):
        Plan_Curvature.append(4)
    else:
        Plan_Curvature.append(5)

i = df['Profile_Curvature']
Profile_Curvature = []
for x in i:
    if(x<=-3.46856041):
        Profile_Curvature.append(0)
    elif(x>=-3.46856040 and x<=-1.346149886):
        Profile_Curvature.append(1)
    elif(x>=-1.346149885 and x<=-0.133343871):
        Profile_Curvature.append(2)
    elif(x>=-0.133343870 and x<=1.079462141):
        Profile_Curvature.append(3)
    elif(x>=1.079462142 and x<=3.050271913):
        Profile_Curvature.append(4)
    else:
        Profile_Curvature.append(5)

df = df.drop(['Plan_Curvature','Profile_Curvature'], axis = 1)
Plan_Curvature = pd.Series(Plan_Curvature)
Profile_Curvature = pd.Series(Profile_Curvature)
df = pd.concat([df,Plan_Curvature,Profile_Curvature], axis=1)
df = df.rename(columns={0:'Plan_Curvature',1:'Profile_Curvature'})

i = df['Aspect']
j = []
for x in i:
    if(x<0):
        j.append(0)
    elif(x>=0 and x<22.5):
        j.append(1)
    elif(x>=22.5 and x<67.5):
        j.append(2)
    elif(x>=67.5 and x<112.5):
        j.append(3)
    elif(x>=112.5 and x<157.5):
        j.append(4)
    elif(x>=157.5 and x<202.5):
        j.append(5)
    elif(x>=202.5 and x<247.5):
        j.append(6)
    elif(x>=247.5 and x<292.5):
        j.append(7)
    elif(x>=292.5 and x<337.5):
        j.append(8)
    else:
        j.append(1)
df = df.drop(['Aspect'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'Aspect'})

i = df['Fault']
Fault = []
for x in i:
    if(x==50):
        Fault.append(0)
    elif(x==100):
        Fault.append(1)
    elif(x==150):
        Fault.append(2)
    elif(x==200):
        Fault.append(3)
    elif(x==250):
        Fault.append(4)
    elif(x==300):
        Fault.append(5)
df = df.drop(['Fault'],axis=1)
Fault = pd.Series(Fault)
df = pd.concat([df,Fault], axis=1)
df = df.rename(columns={0:'Fault'})

i = df['River']
j = []
for x in i:
    if(x==40):
        j.append(0)
    elif(x==80):
        j.append(1)
    elif(x==120):
        j.append(2)
    elif(x==160):
        j.append(3)
    elif(x==200):
        j.append(4)
    elif(x==240):
        j.append(5)
df = df.drop(['River'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'River'})

i = df['Road']
j = []
for x in i:
    if(x==40):
        j.append(0)
    elif(x==80):
        j.append(1)
    elif(x==120):
        j.append(2)
    elif(x==160):
        j.append(3)
    elif(x==200):
        j.append(4)
    elif(x==240):
        j.append(5)
df = df.drop(['Road'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'Road'})


i = df['Dem']
j = []
for x in i:
    if(x<=887):
        j.append(0)
    elif(x>=888 and x<=1437):
        j.append(1)
    elif(x>=1438 and x<=2057):
        j.append(2)
    elif(x>=2058 and x<=2748):
        j.append(3)
    elif(x>=2749 and x<=3457):
        j.append(4)
    else:
        j.append(5)
df = df.drop(['Dem'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'Dem'})

i = df['Slope']
Slope = []
for x in i:
    if(x<=10.87785184):
        Slope.append(0)
    elif(x>=10.87785185 and x<=21.16771168):
        Slope.append(1)
    elif(x>=21.16771169 and x<=29.69359556):
        Slope.append(2)
    elif(x>=29.69359557 and x<=37.92548343):
        Slope.append(3)
    elif(x>=37.92548344 and x<=47.33335529):
        Slope.append(4)
    else:
        Slope.append(5)
df = df.drop(['Slope'],axis=1)
Slope = pd.Series(Slope)
df = pd.concat([df,Slope], axis=1)
df = df.rename(columns={0:'Slope'})

i = df['TWI']
j = []
for x in i:
    if(x<=3.375663230):
        j.append(0)
    elif(x>=3.375663231 and x<=5.585415369):
        j.append(1)
    elif(x>=5.585415370 and x<=8.679068363):
        j.append(2)
    elif(x>=8.679068364 and x<=11.44125854):
        j.append(3)
    elif(x>=11.44125855 and x<=13.65101068):
        j.append(4)
    else:
        j.append(5)

df = df.drop(['TWI'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'TWI'})

i = df['STI']
j = []
for x in i:
    if(x<=1.129179711):
        j.append(0)
    elif(x>=1.129179712 and x<=4.516718846):
        j.append(1)
    elif(x>=4.516718847 and x<=10.72720726):
        j.append(2)
    elif(x>=10.72720727 and x<=20.88982466):
        j.append(3)
    elif(x>=20.88982467 and x<=40.65046961):
        j.append(4)
    else:
        j.append(5)
df = df.drop(['STI'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'STI'})

i = df['SPI']
j = []
for x in i:
    if(x<=7.843282782):
        j.append(0)
    elif(x>=7.843282783 and x<=39.21641391):
        j.append(1)
    elif(x>=39.21641392 and x<=113.7276003):
        j.append(2)
    elif(x>=113.7276004 and x<=282.3581801):
        j.append(3)
    elif(x>=282.3581802 and x<=615.6976984):
        j.append(4)
    else:
        j.append(5)
df = df.drop(['SPI'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'SPI'})

i = df['Slope_Length']
j = []
for x in i:
    if(x<=108.0233130):
        j.append(0)
    elif(x>=108.0233131 and x<=324.0699391):
        j.append(1)
    elif(x>=324.0699392 and x<=669.7445408):
        j.append(2)
    elif(x>=669.7445409 and x<=1188.256443):
        j.append(3)
    elif(x>=1188.256444 and x<=2052.442948):
        j.append(4)
    else:
        j.append(5)
df = df.drop(['Slope_Length'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'Slope_Length'})


#moving landslide in the last
landslide = df['Landslide']
df = df.drop(['Landslide'],axis=1)
df = pd.concat([df,landslide], axis=1)

#saving csv file
df.to_csv('../csv/processed1_truncated_data.csv')

 
#%% """preprocessing 2 - handling DEM derivatives, all buffer variables"""
"""Class (equal interval)"""

i = df['Plan_Curvature']
Plan_Curvature = []
for x in i:
    if(x<=-13.86327203):
        Plan_Curvature.append(0)
    elif(x>=-13.86327202 and x<=-7.734709422):
        Plan_Curvature.append(1)
    elif(x>=-7.734709421 and x<=-1.606146812):
        Plan_Curvature.append(2)
    elif(x>=-1.606146811 and x<=4.522415797):
        Plan_Curvature.append(3)
    elif(x>=4.522415798 and x<=10.65097841):
        Plan_Curvature.append(4)
    else:
        Plan_Curvature.append(5)
        
i = df['Profile_Curvature']
Profile_Curvature = []
for x in i:
    if(x<=-12.51407193):
        Profile_Curvature.append(5)
    elif(x>=-12.51407192 and x<=-6.096306772):
        Profile_Curvature.append(4)
    elif(x>=-6.096306771 and x<= 0.321458383):
        Profile_Curvature.append(3)
    elif(x>=0.321458384 and x<=6.739223538):
        Profile_Curvature.append(2)
    elif(x>=6.739223539 and x<=13.15698869):
        Profile_Curvature.append(1)
    else:
        Profile_Curvature.append(0)

df = df.drop(['Plan_Curvature','Profile_Curvature'], axis = 1)
Plan_Curvature = pd.Series(Plan_Curvature)
Profile_Curvature = pd.Series(Profile_Curvature)
df = pd.concat([df,Plan_Curvature,Profile_Curvature], axis=1)
df = df.rename(columns={0:'Plan_Curvature',1:'Profile_Curvature'})

i = df['Aspect']
j = []
for x in i:
    if(x<0):
        j.append(0)
    elif(x>=0 and x<22.5):
        j.append(1)
    elif(x>=22.5 and x<67.5):
        j.append(2)
    elif(x>=67.5 and x<112.5):
        j.append(3)
    elif(x>=112.5 and x<157.5):
        j.append(4)
    elif(x>=157.5 and x<202.5):
        j.append(5)
    elif(x>=202.5 and x<247.5):
        j.append(6)
    elif(x>=247.5 and x<292.5):
        j.append(7)
    elif(x>=292.5 and x<337.5):
        j.append(8)
    else:
        j.append(1)
df = df.drop(['Aspect'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'Aspect'})

i = df['Fault']
Fault = []
for x in i:
    if(x==50):
        Fault.append(0)
    elif(x==100):
        Fault.append(1)
    elif(x==150):
        Fault.append(2)
    elif(x==200):
        Fault.append(3)
    elif(x==250):
        Fault.append(4)
    elif(x==300):
        Fault.append(5)
df = df.drop(['Fault'],axis=1)
Fault = pd.Series(Fault)
df = pd.concat([df,Fault], axis=1)
df = df.rename(columns={0:'Fault'})

i = df['River']
j = []
for x in i:
    if(x==40):
        j.append(0)
    elif(x==80):
        j.append(1)
    elif(x==120):
        j.append(2)
    elif(x==160):
        j.append(3)
    elif(x==200):
        j.append(4)
    elif(x==240):
        j.append(5)
df = df.drop(['River'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'River'})

i = df['Road']
j = []
for x in i:
    if(x==40):
        j.append(0)
    elif(x==80):
        j.append(1)
    elif(x==120):
        j.append(2)
    elif(x==160):
        j.append(3)
    elif(x==200):
        j.append(4)
    elif(x==240):
        j.append(5)
df = df.drop(['Road'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'Road'})


i = df['Dem']
j = []
for x in i:
    if(x<=1073):
        j.append(0)
    elif(x>=1074 and x<=1827):
        j.append(1)
    elif(x>=1828 and x<=2580):
        j.append(2)
    elif(x>=2581 and x<=3333):
        j.append(3)
    elif(x>=3334 and x<=4087):
        j.append(4)
    else:
        j.append(5)
df = df.drop(['Dem'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'Dem'})

i = df['Slope']
Slope = []
for x in i:
    if(x<=12.34783182):
        Slope.append(0)
    elif(x>=12.34783183 and x<=24.98965963):
        Slope.append(1)
    elif(x>=24.98965964 and x<=37.63148744):
        Slope.append(2)
    elif(x>=37.63148745 and x<=49.97931925):
        Slope.append(3)
    elif(x>=49.97931926 and x<=62.62114707):
        Slope.append(4)
    else:
        Slope.append(5)
df = df.drop(['Slope'],axis=1)
Slope = pd.Series(Slope)
df = pd.concat([df,Slope], axis=1)
df = df.rename(columns={0:'Slope'})

i = df['TWI']
j = []
for x in i:
    if(x<=-3.087861776):
        j.append(0)
    elif(x>=-3.087861775 and x<=1.607861519):
        j.append(1)
    elif(x>=1.60786152 and x<=6.303584814):
        j.append(2)
    elif(x>=6.303584815 and x<=10.99930811):
        j.append(3)
    elif(x>=10.99930812 and x<=15.6950314):
        j.append(4)
    else:
        j.append(5)

df = df.drop(['TWI'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'TWI'})

i = df['STI']
j = []
for x in i:
    if(x<=23.99506887):
        j.append(0)
    elif(x>=23.99506888 and x<=47.99013774):
        j.append(1)
    elif(x>=47.99013775 and x<=71.9852066):
        j.append(2)
    elif(x>=71.98520661 and x<=95.98027547):
        j.append(3)
    elif(x>=95.98027548 and x<=119.9753443):
        j.append(4)
    else:
        j.append(5)
df = df.drop(['STI'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'STI'})

i = df['SPI']
j = []
for x in i:
    if(x<=166.66975910):
        j.append(0)
    elif(x>=166.6697592 and x<=333.3395182):
        j.append(1)
    elif(x>=333.3395183 and x<=500.0092773):
        j.append(2)
    elif(x>=500.0092774 and x<=666.6790365):
        j.append(3)
    elif(x>=666.6790366 and x<=833.3487956):
        j.append(4)
    else:
        j.append(5)
df = df.drop(['SPI'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'SPI'})

i = df['Slope_Length']
j = []
for x in i:
    if(x<=918.1981608):
        j.append(0)
    elif(x>=918.1981609 and x<=1836.396322):
        j.append(1)
    elif(x>=1836.396323 and x<=2754.594482):
        j.append(2)
    elif(x>=2754.594483 and x<=3672.792643):
        j.append(3)
    elif(x>=3672.792644 and x<=4590.990804):
        j.append(4)
    else:
        j.append(5)
df = df.drop(['Slope_Length'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'Slope_Length'})

#moving landslide in the last
landslide = df['Landslide']
df = df.drop(['Landslide'],axis=1)
df = pd.concat([df,landslide], axis=1)

#saving csv file
df.to_csv('../csv/processed2_truncated_data.csv')

#%%
"""preprocessing 3 - handling DEM derivatives, all buffer variables """
"""Class (quantile) 5 equal and one unequal"""

import pandas as pd
import numpy as np
from osgeo import gdal
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from rotation_forest import RotationForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle
import graphviz
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
df = pd.read_csv('../csv/row_truncated_data.csv')

df=df.sort_values(by=['Slope_Length'])#Sorting Slope Length vise 
df['Slope_Length1'] = -1# creating null column
index0 = df['Slope_Length'][df['Slope_Length'] == 0].index
SLlenid0=len(index0)
print(SLlenid0)
for i in range(0,SLlenid0):#for all zeros I'm assinging 0 into newly created bin column
    df.iat[i,-1] = 0
#df
t=0
length=len(df)
bins5width=(length-SLlenid0)//5
#print(bins5width)
for i in range(SLlenid0,length): # for rest of the values I'm dividing into 5 equal sized parts( (length-SLlenid0)//5=bins5width is the size of each part)
    g=i-SLlenid0
    if g % bins5width ==0:# chaning part value by one....t=1 will be assigned to the 1st part among 5 as 0 is assigned above.
        t=t+1
    df.iat[i,-1] = t
#df1=df[['Slope_Length1','Slope_Length']]
# df.iloc[SLlenid0-3:SLlenid0+4]
# df.iloc[SLlenid0+bins5width*1-5:SLlenid0+bins5width*1+5]
# df.iloc[SLlenid0+bins5width*2-5:SLlenid0+bins5width*2+5]
# df.iloc[SLlenid0+bins5width*3-5:SLlenid0+bins5width*3+5]
# df.iloc[SLlenid0+bins5width*4-5:SLlenid0+bins5width*4+5]
# df.iloc[SLlenid0+bins5width*5-5:]
# df['Slope_Length1'].value_counts()
# df = df.drop(['STI1'], axis = 1)

df=df.sort_values(by=['STI'])#Sorting Slope Length vise 
df['STI1'] = -1# creating null column
index0 = df['STI'][df['STI'] == 0].index
STIlenid0=len(index0)
print(STIlenid0)
for i in range(0,STIlenid0):#for all zeros I'm assinging 0 into newly created bin column
    df.iat[i,-1] = 0

t=0
bins5width1=(length-STIlenid0)//5
#print(bins5width1,STIlenid0)
for i in range(STIlenid0,length): # for rest of the values I'm dividing into 5 equal sized parts( (length-STIlenid0)//5=bins5width1 is the size of each part)
    g=i-STIlenid0
    if g % bins5width1 ==0:# chaning part value by one....t=1 will be assigned to the 1st part among 5 as 0 is assigned above.
        t=t+1
    df.iat[i,-1] = t


# df.iloc[STIlenid0-3:STIlenid0+4]
# df.iloc[STIlenid0+bins5width1*1-5:STIlenid0+bins5width1*1+5]
# df.iloc[STIlenid0+bins5width1*2-5:STIlenid0+bins5width1*2+5]
# df.iloc[STIlenid0+bins5width1*3-5:STIlenid0+bins5width1*3+5]
# df.iloc[STIlenid0+bins5width1*4-5:STIlenid0+bins5width1*4+5]
# df.iloc[STIlenid0+bins5width1*5-4:]
# df
# df['STI1'].value_counts()
df.iloc[STIlenid0+bins5width1*5-5:]['STI1']=5 # for few values which is being assigned to 6 so converting them to 5

df=df.sort_values(by=['SPI'])#Sorting Slope Length vise 
df['SPI1'] = -1# creating null column
index0 = df['SPI'][df['SPI'] == 0].index
SPIlenid0=len(index0)
print(SPIlenid0)
for i in range(0,SPIlenid0):#for all zeros I'm assinging 0 into newly created bin column
    df.iat[i,-1] = 0
t=0
bins5width2=(length-SPIlenid0)//5
#print(bins5width)
for i in range(SPIlenid0,length): # for rest of the values I'm dividing into 5 equal sized parts( (length-SPIlenid0)//5=bins5width2 is the size of each part)
    g=i-SPIlenid0
    if g % bins5width2 ==0:# chaning part value by one....t=1 will be assigned to the 1st part among 5 as 0 is assigned above.
        t=t+1
    df.iat[i,-1] = t

df.iloc[STIlenid0+bins5width1*5-5:]['SPI1']=5 # for few values which is being assigned to 6 so converting them to 5
# df.iloc[SPIlenid0-3:SPIlenid0+4]
# df.iloc[SPIlenid0+bins5width2*1-5:SPIlenid0+bins5width2*1+5]
# df.iloc[SPIlenid0+bins5width2*2-5:SPIlenid0+bins5width2*2+5]
# df.iloc[SPIlenid0+bins5width2*3-5:SPIlenid0+bins5width2*3+5]
# df.iloc[SPIlenid0+bins5width2*4-5:SPIlenid0+bins5width2*4+5]
# df.iloc[SPIlenid0+bins5width2*5-5:]
# df['SPI1'].value_counts()
#Functions which Performs binning on provided column name
binswidth=length//6
def bins(df,cname):
    df=df.sort_values(by=[cname]) #sorting in asc order
    df[cname+'1'] = 0 #creating null column
    t=-1
    for  i in range(0,length):
        if(i%binswidth==0): #binning value vhange condition
            t=t+1
        df.iat[i,-1] = t
    return df


df=bins(df,'Slope')
df=bins(df,'Dem')
df=bins(df,'Aspect')
df=bins(df,'Plan_Curvature')
df=bins(df,'Profile_Curvature')
df=bins(df,'TWI')
df = df.drop(['Plan_Curvature','Profile_Curvature','Slope','Dem','Slope_Length','Aspect','STI','SPI','TWI'], axis = 1)
df = df.rename(columns={'Slope_Length':'Slope_Length1'})
dict = {'Slope_Length1': 'Slope_Length','Aspect1': 'Aspect','Slope1': 'Slope','STI1': 'STI','SPI1': 'SPI','Dem1': 'Dem','Plan_Curvature1': 'Plan_Curvature','Profile_Curvature1': 'Profile_Curvature','TWI1':'TWI'}
df.rename(columns=dict,inplace=True)

df = df.rename(columns={0:'Fault'})
df=df.sort_index(axis = 0) # sorting indexwise as original data.
#df
i = df['Fault']
Fault = []
for x in i:
    if(x==50):
        Fault.append(0)
    elif(x==100):
        Fault.append(1)
    elif(x==150):
        Fault.append(2)
    elif(x==200):
        Fault.append(3)
    elif(x==250):
        Fault.append(4)
    elif(x==300):
        Fault.append(5)
df = df.drop(['Fault'],axis=1)
Fault = pd.Series(Fault)
df = pd.concat([df,Fault], axis=1)
df = df.rename(columns={0:'Fault'})


i = df['River']
j = []
for x in i:
    if(x==40):
        j.append(0)
    elif(x==80):
        j.append(1)
    elif(x==120):
        j.append(2)
    elif(x==160):
        j.append(3)
    elif(x==200):
        j.append(4)
    elif(x==240):
        j.append(5)
df = df.drop(['River'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'River'})



i = df['Road']
j = []
for x in i:
    if(x==40):
        j.append(0)
    elif(x==80):
        j.append(1)
    elif(x==120):
        j.append(2)
    elif(x==160):
        j.append(3)
    elif(x==200):
        j.append(4)
    elif(x==240):
        j.append(5)
df = df.drop(['Road'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'Road'})
# putting original order
df = df[["Latitude", "Longitude", "Lithology","Land_Cover","Plan_Curvature","Profile_Curvature","Aspect","Fault","River","Road","Dem","Slope","TWI","STI","SPI","Slope_Length","Landslide"]]
df.to_csv('../csv/processed3_truncated_data.csv', index=False)# saving it into csv file.

#%%
"""preprocessing 4 - handling DEM derivatives, all buffer variables """
"""Class (quantile) all equal histogram"""
import pandas as pd
import numpy as np
from osgeo import gdal
from statsmodels.stats.outliers_influence import variance_inflation_factor
df = pd.read_csv('../csv/row_truncated_data.csv')

A=[]
length=len(df)
for i in range(length):
    A.append(7+i)
binswidth=length//6
index0 = df['Slope_Length'][df['Slope_Length'] == 0].index
index0 = pd.Series(index0)
index0 = np.array(index0)
index1 = np.random.choice(index0,binswidth, replace=False)
index2 = []
for  k in index0:    
    if k not in index1:
        index2.append(k)

SLlenid2=len(index2)
for k in index1: #random 203092 Zeros
    A[k]=0
for k in index2: #remaining Zeros 
    A[k]=1

print(binswidth+SLlenid2)
A = pd.Series(A)
df = pd.concat([df,A], axis=1)
df = df.rename(columns={0:'Slope_Length1'})
df=df.sort_values(by=['Slope_Length','Slope_Length1'],ignore_index=False)
#df.iloc[:binswidth+1]
#print("#######################################################################")
#df.iloc[binswidth+SLlenid2-2:binswidth+SLlenid2+3]
t=1
for  i in range(binswidth+SLlenid2,length):
    if(i%binswidth==0): #binning value change condition(binswidth + SLlenid2 to 2*binswidth we will assign 1 to it. & later on we will increment t as given)
        t=t+1
    df.iat[i,-1] = t
#df.iloc[binswidth+SLlenid2-3:binswidth+SLlenid2+4]
#df.iloc[binswidth*2-5:binswidth*2+5]
#df.iloc[binswidth*3-5 : binswidth*3+5]
#df.iloc[binswidth*4-5:binswidth*4+5]
#df.iloc[binswidth*5-5:binswidth*5+5]
#df.iloc[binswidth*6-5:]
#df['Slope_Length1'].value_counts()
B=[]
for i in range(length):
    B.append(7+i)
index0 = df['STI'][df['STI'] == 0].index
index0 = pd.Series(index0)
index0 = np.array(index0)
index1 = np.random.choice(index0,binswidth, replace=False)
index2 = []
for  k in index0:    
    if k not in index1:
        index2.append(k)
STIlenid2=len(index2)
for k in index1: #random 203092 Zeros
    B[k]=0
for k in index2: #remaining 51520 Zeros 
    B[k]=1
print(STIlenid2,len(index1)+STIlenid2)
B = pd.Series(B)
df = pd.concat([df,B], axis=1)
df = df.rename(columns={0:'STI1'})
df=df.sort_values(by=['STI','STI1'],ignore_index=False)
#df.iloc[:binswidth+1]
#print("#######################################################################")
#df.iloc[binswidth+STIlenid2-2:binswidth+STIlenid2+3]
t=1
for  i in range(binswidth+STIlenid2,length):
    if(i%binswidth==0): #binning value change condition(binswidth + STIlenid2 to 2*binswidth we will assign 1 to it. & later on we will increment t as given)
        t=t+1
    df.iat[i,-1] = t
#df.iloc[binswidth+STIlenid2-3:binswidth+STIlenid2+4]
#df.iloc[binswidth*2-5:binswidth*2+5]
#df.iloc[binswidth*3-5 : binswidth*3+5]
#df.iloc[binswidth*4-5:binswidth*4+5]
#df.iloc[binswidth*5-5:binswidth*5+5]
#df.iloc[binswidth*6-5:]
#df['STI1'].value_counts()
C=[]
for i in range(length):
    C.append(7+i)
index0 = df['SPI'][df['SPI'] == 0].index
index0 = pd.Series(index0)
index0 = np.array(index0)
index1 = np.random.choice(index0,binswidth, replace=False)
index2 = []
for  k in index0:    
    if k not in index1:
        index2.append(k)
SPIlenid2=len(index2)
for k in index1: #random 203092 Zeros
    C[k]=0
for k in index2: #remaining Zeros 
    C[k]=1
print(SPIlenid2,len(index1)+SPIlenid2)
C = pd.Series(C)
df = pd.concat([df,C], axis=1)
df = df.rename(columns={0:'SPI1'})
df=df.sort_values(by=['SPI','SPI1'],ignore_index=False)
#df.iloc[:binswidth+1]
#print("#######################################################################")
#df.iloc[binswidth+SPIlenid2-2:binswidth+SPIlenid2+3]
t=1
for  i in range(binswidth+SPIlenid2,length):
    if(i%binswidth==0): #binning value change condition(binswidth + SPIlenid2 to 2*binswidth we will assign 1 to it. & later on we will increment t as given)
        t=t+1
    df.iat[i,-1] = t
#df.iloc[binswidth+SPIlenid2-3:binswidth+SPIlenid2+4]
# df.iloc[binswidth*2-5:binswidth*2+5]
# df.iloc[binswidth*3-5 : binswidth*3+5]
# df.iloc[binswidth*4-5:binswidth*4+5]
# df.iloc[binswidth*5-5:binswidth*5+5]
# df.iloc[binswidth*6-5:]
# df['SPI1'].value_counts()
#Functions which Performs binning on provided column name
def bins(df,cname):
    df=df.sort_values(by=[cname]) #sorting in asc order
    df[cname+'1'] = 0 #creating null column
    t=-1
    for  i in range(0,length):
        if(i%binswidth==0): #binning value change condition
            t=t+1
        df.iat[i,-1] = t
    return df
df=bins(df,'Slope')
df=bins(df,'Dem')
df=bins(df,'Aspect')
df=bins(df,'Plan_Curvature')
df=bins(df,'Profile_Curvature')
df=bins(df,'TWI')
df = df.drop(['Plan_Curvature','Profile_Curvature','Slope','Dem','Slope_Length','Aspect','STI','SPI','TWI'], axis = 1)
df=df.sort_index(axis = 0)# sorting indexwise as original data.
i = df['Fault']
Fault = []
for x in i:
    if(x==50):
        Fault.append(0)
    elif(x==100):
        Fault.append(1)
    elif(x==150):
        Fault.append(2)
    elif(x==200):
        Fault.append(3)
    elif(x==250):
        Fault.append(4)
    elif(x==300):
        Fault.append(5)
df = df.drop(['Fault'],axis=1)
Fault = pd.Series(Fault)
df = pd.concat([df,Fault], axis=1)
df = df.rename(columns={0:'Fault'})
i = df['River']
j = []
for x in i:
    if(x==40):
        j.append(0)
    elif(x==80):
        j.append(1)
    elif(x==120):
        j.append(2)
    elif(x==160):
        j.append(3)
    elif(x==200):
        j.append(4)
    elif(x==240):
        j.append(5)
df = df.drop(['River'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'River'})
i = df['Road']
j = []
for x in i:
    if(x==40):
        j.append(0)
    elif(x==80):
        j.append(1)
    elif(x==120):
        j.append(2)
    elif(x==160):
        j.append(3)
    elif(x==200):
        j.append(4)
    elif(x==240):
        j.append(5)
df = df.drop(['Road'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'Road'})

dict = {'Slope_Length1': 'Slope_Length','Aspect1': 'Aspect','Slope1': 'Slope','STI1': 'STI','SPI1': 'SPI','Dem1': 'Dem','Plan_Curvature1': 'Plan_Curvature','Profile_Curvature1': 'Profile_Curvature','TWI1':'TWI'}
df.rename(columns=dict,inplace=True)


df = df[["Latitude", "Longitude", "Lithology","Land_Cover","Plan_Curvature","Profile_Curvature","Aspect","Fault","River","Road","Dem","Slope","TWI","STI","SPI","Slope_Length","Landslide"]]
df.to_csv('../csv/processed4_truncated_data.csv', index=False)# saving it into csv file.

#%% stratification and train test split(by lable) by undersampling
import pandas as pd
import numpy as np
from osgeo import gdal
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from rotation_forest import RotationForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle
import graphviz
from sklearn import tree
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv("../csv/processed1_truncated_data.csv")
df=df.drop(df.columns[0],axis=1) # one extra column of index created in processed and hence removed for first 3 classification code.

""""since data is unbalanced, stratify the data without replacement"""

df1 = df.Landslide[df.Landslide.eq(2)].index
df2 = df.Landslide[df.Landslide.eq(1)].sample(len(df1),replace=False).index

df_new = df1.union(df2)
df_final= df.loc[df_new]
df_final.to_csv('../csv/stratified_data.csv', index = False)

""""creating train test split and saving them for future use after reading startified data"""

#df_final = pd.read_csv("../csv/stratified_data0.csv")
X = df_final.iloc[:,:16]
Y = df_final.iloc[:,16]

X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=1,test_size = 0.3,stratify = Y)

#saving latitude and longtitude along with independent variable
y_train = pd.concat([y_train,X_train['Latitude'],X_train['Longitude']],axis=1)
y_test = pd.concat([y_test,X_test['Latitude'],X_test['Longitude']],axis=1)

X_train = X_train.iloc[:,2:]  # remove 0 and 1 index of lat long
X_test = X_test.iloc[:,2:]    # remove 0 and 1 index of lat long

#precautionary check if there are any nan values in the data
print(X_train.isnull().values.any())
print(y_train.isnull().values.any())
print(X_test.isnull().values.any())
print(y_test.isnull().values.any())

#writing train test splits to the system for future use
X_train.to_csv('../csv/X_train.csv', index = False)
X_test.to_csv('../csv/X_test.csv', index = False)
y_train.to_csv('../csv/Y_train.csv', index = False)
y_test.to_csv('../csv/Y_test.csv', index = False)

#obtain labels without lat long 
Y_train = y_train.iloc[:,0]
Y_test = y_test.iloc[:,0]


#%% stratification and train test split (by label) for whole population
import pandas as pd
import numpy as np
from osgeo import gdal
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from rotation_forest import RotationForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle
import graphviz
from sklearn import tree
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv("../csv/processed4_truncated_data.csv")
#df=df.drop(df.columns[0],axis=1) # one extra column of index created in processed and hence removed for first 3 classification code.


X = df.iloc[:,:16]
Y = df.iloc[:,16]

X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=1,test_size = 0.3, stratify = Y )

#saving latitude and longtitude along with independent variable
y_train = pd.concat([y_train,X_train['Latitude'],X_train['Longitude']],axis=1)
y_test = pd.concat([y_test,X_test['Latitude'],X_test['Longitude']],axis=1)

X_train = X_train.iloc[:,2:]  # remove 0 and 1 index of lat long
X_test = X_test.iloc[:,2:]    # remove 0 and 1 index of lat long

#precautionary check if there are any nan values in the data
print(X_train.isnull().values.any())
print(y_train.isnull().values.any())
print(X_test.isnull().values.any())
print(y_test.isnull().values.any())

#writing train test splits to the system for future use
X_train.to_csv('../csv/X_train.csv', index = False)
X_test.to_csv('../csv/X_test.csv', index = False)
y_train.to_csv('../csv/Y_train.csv', index = False)
y_test.to_csv('../csv/Y_test.csv', index = False)

#obtain labels without lat long 
Y_train = y_train.iloc[:,0]
Y_test = y_test.iloc[:,0]

