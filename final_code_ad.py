#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:04:30 2020

@author: aggarwal
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
#import graphviz
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

#%%
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
#%%
"""preprocessing 0 - handling profile and plan curvature aspect, all buffer variables"""

i = df['Plan_Curvature']
Plan_Curvature = []
for x in i:
    if(x<0):
        Plan_Curvature.append(2)
    elif(x>0):
        Plan_Curvature.append(1)
    else:
        Plan_Curvature.append(0)

i = df['Profile_Curvature']
Profile_Curvature = []
for x in i:
    if(x<0):
        Profile_Curvature.append(2)
    elif(x>0):
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

#moving landslide in the last
landslide = df['Landslide']
df = df.drop(['Landslide'],axis=1)
df = pd.concat([df,landslide], axis=1)

#saving csv file
df.to_csv('../csv/processed0_truncated_data.csv')

#%%
"""preprocessing 1 - handling DEM derivatives, all buffer variables"""
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
        Profile_Curvature.append(5)
    elif(x>=-3.46856040 and x<=-1.346149886):
        Profile_Curvature.append(4)
    elif(x>=-3.46856040 and x<=-1.346149886):
        Profile_Curvature.append(3)
    elif(x>=-0.133343871 and x<=1.079462141):
        Profile_Curvature.append(2)
    elif(x>=1.079462142 and x<=3.050271913):
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


#%%
"""preprocessing 2 - handling DEM derivatives, all buffer variables"""
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
    elif(x>=0.321458383 and x<=6.739223538):
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
"""preprocessing 3 - handling DEM derivatives, all buffer variables"""
"""Class (quantile)"""

i = df['Plan_Curvature']
Plan_Curvature = []
for x in i: 
    if(x>=-19.99183464 and x<=-2.543456383):
        Plan_Curvature.append(0)
    elif(x>=-2.543456382 and x<=-0.957240191):
        Plan_Curvature.append(1)
    elif(x>=-0.957240190 and x<=-0.092031344):
        Plan_Curvature.append(2)
    elif(x>=-0.092031343 and x<=0.340573075):
        Plan_Curvature.append(3)
    elif(x>=0.340573076 and x<=0.917378968):
        Plan_Curvature.append(4)
    elif(x>=0.917378969 and x<=16.777954102):
        Plan_Curvature.append(5)

i = df['Profile_Curvature']
Profile_Curvature = []
for x in i:
    if(x>=-18.93183708 and x<=-1.194549134):
        Profile_Curvature.append(5)
    elif(x>=-1.194549133 and x<=-0.436545376):
        Profile_Curvature.append(4)
    elif(x>=-0.436545375 and x<=0.018256879):
        Profile_Curvature.append(3)
    elif(x>=0.018256880 and x<=0.473059134):
        Profile_Curvature.append(2)
    elif(x>=0.473059135 and x<=1.382663645):
        Profile_Curvature.append(1)
    elif(x>=1.382663646 and x<=19.57475385):
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
    if(x>=320 and x<=958):
        j.append(0)
    elif(x>=959 and x<=1259):
        j.append(1)
    elif(x>=1260 and x<=1525):
        j.append(2)
    elif(x>=1526 and x<=1968):
        j.append(3)
    elif(x>=1969 and x<=2890):
        j.append(4)
    elif(x>=2891 and x<=4840):
        j.append(5)
df = df.drop(['Dem'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'Dem'})

i = df['Slope']
Slope = []
for x in i:
    if(x>=0 and x<=14.4058079):
        Slope.append(0)
    elif(x>=14.4058080 and x<=22.34369567):
        Slope.append(1)
    elif(x>=22.34369568 and x<=28.51761157):
        Slope.append(2)
    elif(x>=28.51761158 and x<=34.39753149):
        Slope.append(3)
    elif(x>=34.39753150 and x<=41.45343538):
        Slope.append(4)
    elif(x>=41.45343539 and x<=74.96897888):
        Slope.append(5)
df = df.drop(['Slope'],axis=1)
Slope = pd.Series(Slope)
df = pd.concat([df,Slope], axis=1)
df = df.rename(columns={0:'Slope'})

i = df['TWI']
j = []
for x in i:
    if(x>=-7.783585072 and x<=4.038588872):
        j.append(0)
    elif(x>=4.038588873 and x<=5.806390583):
        j.append(1)
    elif(x>=5.806390584 and x<=10.22589486):
        j.append(2)
    elif(x>=10.22589487 and x<=11.66223375):
        j.append(3)
    elif(x>=11.66223376 and x<=12.76710982):
        j.append(4)
    elif(x>=12.76710983 and x<=20.3907547):
        j.append(5)

df = df.drop(['TWI'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'TWI'})

i = df['STI']
j = []
new = []
for x in i:
    if(x<0.000001):
        new.append(0) 
new = shuffle(new)
l1 = new[:203092] 
l2 = new[203092:]
j = l1.copy()
for x in l2:
    j.append(1)
    
for x in i:
    if(x>0 and x<=0.564589856):
        j.append(1)
    elif(x>=0.564589857 and x<=1.129179711):
        j.append(2)
    elif(x>=1.129179712 and x<=2.258359423):
        j.append(3)
    elif(x>=2.258359424 and x<=4.516718846):
        j.append(4)
    elif(x>=4.516718847 and x<=143.9704132):
        j.append(5)
df = df.drop(['STI'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'STI'})

i = df['SPI']
j = []
new = []
for x in i:
    if(x<0.000001):
        new.append(0) 
new = shuffle(new)
l1 = new[:203092] 
l2 = new[203092:]
j = l1.copy()
for x in l2:
    j.append(1)
    
for x in i:
    if(x>0 and x<=3.921641391):
        j.append(1)
    elif(x>=3.921641392 and x<=7.843282782):
        j.append(2)
    elif(x>=7.843282783 and x<=15.68656556):
        j.append(3)
    elif(x>=15.68656557 and x<=631.37313113):
        j.append(4)
    elif(x>=31.37313114 and x<=1000.018555):
        j.append(5)
df = df.drop(['SPI'],axis=1)
j = pd.Series(j)
df = pd.concat([df,j], axis=1)
df = df.rename(columns={0:'SPI'})

i = df['Slope_Length']
j = []
new = []
for x in i:
    if(x<0.000001):
        new.append(0) 
new = shuffle(new)
l1 = new[:203092] 
l2 = new[203092:]
j = l1.copy()
for x in l2:
    j.append(1)

for x in i:
    if(x>0 and x<=21.60466261):
        j.append(1)
    elif(x>=21.60466262 and x<=86.41865043):
        j.append(2)
    elif(x>=86.41865044 and x<=172.8373009):
        j.append(3)
    elif(x>=172.8373010 and x<=345.6746017):
        j.append(4)
    elif(x>=345.6746018 and x<=5509.188965):
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
df.to_csv('../csv/processed3_truncated_data.csv')
#%%
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
#import graphviz
from sklearn import tree
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv("../csv/processed3_truncated_data.csv")
df=df.drop(df.columns[0],axis=1) # one extra column of index created in processed and hence removed.

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

X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=1,test_size = 0.3)

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

#%%
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
#import graphviz
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
"""to import already saved train test splits"""
X_train = pd.read_csv('../csv/0/X_train.csv')
y_train = pd.read_csv('../csv/0/Y_train.csv')
X_test = pd.read_csv('../csv/0/X_test.csv')
y_test = pd.read_csv('../csv/0/Y_test.csv')

Y_train = y_train.iloc[:,0]
Y_test = y_test.iloc[:,0]


#%%
"""DECISION TREE OPTIMISATION"""

params = {
        'criterion': ['gini','entropy'],
        'max_depth': [None,2,3, 4, 5],
        'min_samples_leaf': [1,2,3, 4, 5,6],
        'min_samples_split': [2,3, 4, 5,6]
        }

decision_model = DecisionTreeClassifier(random_state = 1)
grid_model = GridSearchCV(decision_model,params,scoring='roc_auc',verbose=1,n_jobs=-1)
grid_model.fit(X_train,Y_train)

#print("dt auc")
#print(grid_model.best_params_)
#Print out scores on validation set
#print(random_model.score(X_test,Y_test))
#print(grid_model.score(X_test,Y_test))
#print out scores on training data
#print(random_model.best_score_)
#print("train_score_auc:"+str(grid_model.best_score_))

""""OPTIMISED DECISION TREE"""


dt_model = DecisionTreeClassifier(random_state=1,min_samples_leaf=6,max_depth=5)
dt_model.fit(X_train, Y_train)
dt_score = dt_model.score(X_test, Y_test)
dt_score_train = dt_model.score(X_train, Y_train)

cm_dt = confusion_matrix(Y_test, dt_model.predict(X_test))

dt_predict  = dt_model.predict(X_test)
dt_probs = dt_model.predict_proba(X_test)
dtfpr = dict()
dttpr = dict()
dtroc_auc = dict()
dtfpr, dttpr, _ = roc_curve(Y_test, dt_probs[:,1], drop_intermediate = False,pos_label=2)
dtroc_auc = auc(dtfpr,dttpr)

print(dt_model.feature_importances_)
#%%
"""save dt model"""

pickle.dump(dt_model, open('decison_model.sav', 'wb'))

"""load saved model"""

load_dt_model = pickle.load(open('decison_model.sav', 'rb'))

#%% """Render decision tree"""

import os
os.environ['PATH'] += os.pathsep + 'C:\graphviz\bin'
features = []
for i in X_train:
    features.append(i)
output = ['0','1']
dot_data = tree.export_graphviz(dt_model, out_file = None, filled = True, 
                                rounded = True, special_characters = True, 
                                class_names= output, feature_names=features)
graph = graphviz.Source(dot_data)
graph.render("landslide_dt")

"""will save the graph in pwd"""

#%%
"""RANDOM FOREST OPTIMISATION"""

params = {
        'max_depth': [None,2,3, 4, 5],
        'n_estimators': [100,250,500],
        'min_samples_leaf': [1,2,3, 4, 5,6],
        'min_samples_split': [2,3, 4, 5,6]
        }

random_model = RandomForestClassifier(random_state = 1)
grid_model = GridSearchCV(random_model,params,scoring='roc_auc',verbose=1,n_jobs=-1)
grid_model.fit(X_train,Y_train)

#%%
#Print out best parameters
print("Random auc")
print(grid_model.best_params_)
#Print out scores on validation set

print(grid_model.score(X_test,Y_test))
#print out scores on training data

print(grid_model.best_score_)


#%%
"""OPTIMISED RANDOM FOREST"""

rf_model = RandomForestClassifier(random_state=1,n_estimators=500, max_depth=5)
rf_model.fit(X_train, Y_train)
rf_score = rf_model.score(X_test, Y_test)
rf_score_train = rf_model.score(X_train, Y_train)

cm_rf = confusion_matrix(Y_test, rf_model.predict(X_test))
rf_predict  = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)
rffpr = dict()
rftpr = dict()
rfroc_auc = dict()
rffpr, rftpr, _ = roc_curve(Y_test, rf_probs[:,1], drop_intermediate = False,pos_label=2)
rfroc_auc = auc(rffpr,rftpr)

#%%
"""save rf model"""

pickle.dump(rf_model, open('random_model.sav', 'wb'))

"""load saved model"""

load_rf_model = pickle.load(open('random_model.sav', 'rb'))

#%%
"""ROTATION FOREST OPTIMISATION"""

params = {
        'max_depth': [None,2,3, 4, 5],
        'n_estimators': [100,250,500],
        'criterion': ['gini','entropy'],
        'min_samples_leaf': [1,2,3, 4, 5],
        'min_samples_split': [2,3, 4, 5]
        }

rotation_model = RotationForestClassifier(random_state = 1)
grid_model = GridSearchCV(rotation_model,params,scoring='roc_auc',verbose=1,n_jobs=-1)
grid_model.fit(X_train,Y_train)

#%%
#Print out best parameters
print("rotation auc")
print(grid_model.best_params_)
#Print out scores on validation set

print(grid_model.score(X_test,Y_test))
#print out scores on training data

print(grid_model.best_score_)
#%%
"""OPTIMISED ROTATION FOREST"""

rt_model = RotationForestClassifier(random_state=1, max_depth = 5)
rt_model.fit(X_train,Y_train)
rt_score = rt_model.score(X_test, Y_test)
rt_score_train = rt_model.score(X_train, Y_train)

cm_rt = confusion_matrix(Y_test, rt_model.predict(X_test))
rt_predict  = rt_model.predict(X_test)
rt_probs = rt_model.predict_proba(X_test)
rtfpr = dict()
rttpr = dict()
rtroc_auc = dict()
rtfpr, rttpr, _ = roc_curve(Y_test, rt_probs[:,1], drop_intermediate = False,pos_label=2)
rtroc_auc = auc(rtfpr,rttpr)

#%%
"""save rt model"""

pickle.dump(rt_model, open('rotation_model.sav', 'wb'))

"""load saved model"""

load_rt_model = pickle.load(open('rotation_model.sav', 'rb'))


#%%
"""ADABOOST OPTIMISATION """

learning_rate = list(range(100,300,10))
rate = []
for i in learning_rate:
    rate.append(float(i/100))    
params = {
        'n_estimators': [50,100,250,500,800,1000],
        'learning_rate': rate,
        'algorithm':['SAMME','SAMME.R']
        }

ada_model = AdaBoostClassifier(random_state = 1)
adagrid_model = GridSearchCV(ada_model,params,scoring='roc_auc',verbose=1,n_jobs=-1)
adagrid_model.fit(X_train,Y_train)

#%%
#Print out best parameters
print("ada roc")
print(adagrid_model.best_params_)
#Print out scores on validation set

print(adagrid_model.score(X_test,Y_test))
#print out scores on training data

print(adagrid_model.best_score_)
#%%
"""OPTIMISED AdaBoost Classifier"""

adb_model = AdaBoostClassifier(random_state=1, learning_rate=1.6,n_estimators=1000,algorithm='SAMME')
adb_model.fit(X_train, Y_train)
adb_score = adb_model.score(X_test,Y_test)
adb_score_train = adb_model.score(X_train, Y_train)

cm_ad = confusion_matrix(Y_test, adb_model.predict(X_test))
adb_predict  = adb_model.predict(X_test)
ad_probs = adb_model.predict_proba(X_test)
adfpr = dict()
adtpr = dict()
adroc_auc = dict()
adfpr, adtpr, _ = roc_curve(Y_test, ad_probs[:,1], drop_intermediate = False,pos_label=2)
adroc_auc = auc(adfpr,adtpr)

#%%
"""save adb model"""

pickle.dump(adb_model, open('adaboost_model.sav', 'wb'))

"""load saved model"""

load_adb_model = pickle.load(open('adaboost_model.sav', 'rb'))

#%%
"""EXTRA TREE OPTIMISATION"""

learning_rate = list(range(100,300,10))
rate = []
for i in learning_rate:
    rate.append(float(i/100))    
params = {
        'n_estimators': [100,200,500],
        'max_depth': [None,2,3,4,5],
        'criterion': ['gini','entropy'],
        'min_samples_leaf': [1,2,3, 4, 5,6],
        'min_samples_split': [2,3, 4, 5,6]
        }

extra_model = ExtraTreesClassifier(random_state = 1)
extragrid_model = GridSearchCV(extra_model,params,scoring='roc_auc',verbose=1,n_jobs=-1)
extragrid_model.fit(X_train,Y_train)

#%%
#Print out best parameters
print("extra auc")
print(extragrid_model.best_params_)
#Print out scores on validation set

print(extragrid_model.score(X_test,Y_test))
#print out scores on training data

print(extragrid_model.best_score_)

#%%
"""OPTIMISED Extra Tree Classifier"""

ex_model = ExtraTreesClassifier(random_state=1,max_depth=5)
ex_model.fit(X_train,Y_train)
ex_score = ex_model.score(X_test, Y_test)
ex_score_train = ex_model.score(X_train, Y_train)

cm_ex = confusion_matrix(Y_test, ex_model.predict(X_test))
ex_predict  = ex_model.predict(X_test)
ex_probs = ex_model.predict_proba(X_test)
exfpr = dict()
extpr = dict()
exroc_auc = dict()
exfpr, extpr, _ = roc_curve(Y_test, ex_probs[:,1], drop_intermediate = False,pos_label=2)
exroc_auc = auc(exfpr,extpr)

#%%
"""save extra model"""

pickle.dump(ex_model, open('extra_model.sav', 'wb'))

"""load saved model"""

load_ex_model = pickle.load(open('extra_model.sav', 'rb'))


#%%
"""OPTIMISING XGBOOST"""

learning_rate = list(range(100,260,10))
rate = []
for i in learning_rate:
    rate.append(float(i/100))    
params = {
        'learning_rate': rate,
        'max_depth': [3,4,5,6],
        'gamma':[0.5,1,1.5,2,5],
        'colsample_bytree':[0.8,0.8,1.0],
        'min_child_weight':[1,5,10],
        'subsample': [0.6,0.8,1.0]
        }

xg_model = XGBClassifier(random_state = 1)
xggrid_model = GridSearchCV(xg_model,params,scoring='roc_auc',verbose=1,n_jobs=-1)
xggrid_model.fit(X_train,Y_train)

#%%
#Print out best parameters
print("xg auc")
print(xggrid_model.best_params_)
#Print out scores on validation set

print(xggrid_model.score(X_test,Y_test))
#print out scores on training data

print(xggrid_model.best_score_)

#%%
"""OPTIMISED implement XGBoost"""

xg_model = XGBClassifier(random_state=1, colsample_bytree=0.8, gamma = 5, 
                         max_depth=6)
xg_model.fit(X_train, Y_train)
xg_score = xg_model.score(X_test, Y_test)
xg_score_train = xg_model.score(X_train, Y_train)

cm_xg = confusion_matrix(Y_test, xg_model.predict(X_test))
xg_predict  = xg_model.predict(X_test)
xg_probs = xg_model.predict_proba(X_test)
xgfpr = dict()
xgtpr = dict()
xgroc_auc = dict()
xgfpr, xgtpr, _ = roc_curve(Y_test, xg_probs[:,1], drop_intermediate = False,pos_label=2)
xgroc_auc = auc(xgfpr,xgtpr)

    #%%
"""save xgboost model"""

pickle.dump(xg_model, open('xgboost_model.sav', 'wb'))

"""load saved model"""

load_xg_model = pickle.load(open('xgboost_model.sav', 'rb'))



#%%
"""optimizing logistic regression
penalty =['l2','none']
test_accuracy = []
train_accuracy = []
for i in penalty:
    model = LogisticRegression(random_state=1, penalty=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train, Y_train))

solver = ['newton-cg','lbfgs','liblinear','sag','saga']
test_accuracy = []
train_accuracy = []
for i in solver:
    model = LogisticRegression(random_state=1, solver=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train, Y_train))
#same in all

test_accuracy = []
train_accuracy = []
for i in range(1,101):
    model = LogisticRegression(random_state=1, C=i,max_iter=1000)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train, Y_train))

"""
#%%
"""implementing logistic regression"""


lt_model = LogisticRegression(random_state=1)
lt_model.fit(X_train,Y_train)
lt_score = lt_model.score(X_test,Y_test)
lt_score_train = lt_model.score(X_train,Y_train)

cm_lt = confusion_matrix(Y_test, lt_model.predict(X_test))
lt_predict  = lt_model.predict(X_test)
lt_probs = lt_model.predict_proba(X_test)
ltfpr = dict()
lttpr = dict()
ltroc_auc = dict()
ltfpr, lttpr, _ = roc_curve(Y_test, lt_probs[:,1], drop_intermediate = False,pos_label=2)
ltroc_auc = auc(ltfpr,lttpr)

#%%
"""save logistic model"""

pickle.dump(lt_model, open('logistic_model.sav', 'wb'))

"""load saved model"""

load_lt_model = pickle.load(open('logistic_model.sav', 'rb'))


#%%
"""plotting individual roc curves for all the algos

plt.figure()
plt.plot(dtfpr,dttpr, color = 'darkorange')
plt.plot([0,1],[0,1], color = 'navy',linestyle = '--')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Decision_Tree")
plt.text(0.8,0.2,"auc = "+str(round(dtroc_auc,4)))
plt.show()

plt.figure()
plt.plot(rffpr,rftpr, color = 'darkorange')
plt.plot([0,1],[0,1], color = 'navy',linestyle = '--')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.text(0.8,0.2,"auc = "+str(round(rfroc_auc,4)))
plt.title("Random_Forest")
plt.show()

plt.figure()
plt.plot(rtfpr,rttpr, color = 'darkorange')
plt.plot([0,1],[0,1], color = 'navy',linestyle = '--')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.text(0.8,0.2,"auc = "+str(round(rtroc_auc,4)))
plt.title("Rotation_Forest")
plt.show()

plt.figure()
plt.plot(adfpr,adtpr, color = 'darkorange')
plt.plot([0,1],[0,1], color = 'navy',linestyle = '--')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.text(0.8,0.2,"auc = "+str(round(adroc_auc,4)))
plt.title("AdaBoost")
plt.show()

plt.figure()
plt.plot(exfpr,extpr, color = 'darkorange')
plt.plot([0,1],[0,1], color = 'navy',linestyle = '--')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.text(0.8,0.2,"auc = "+str(round(exroc_auc,4)))
plt.title("Extra_Tree")
plt.show()

plt.figure()
plt.plot(xgfpr,xgtpr, color = 'darkorange')
plt.plot([0,1],[0,1], color = 'navy',linestyle = '--')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("XGBoost")
plt.text(0.8,0.2,"auc = "+str(round(xgroc_auc,4)))
plt.show()

plt.figure()
plt.plot(ltfpr,lttpr, color = 'darkorange')
plt.plot([0,1],[0,1], color = 'navy',linestyle = '--')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Logistic")
plt.text(0.8,0.2,"auc = "+str(round(ltroc_auc,4)))
plt.show()
"""
#%%
"""graph for all roc scores combined"""

data = [['red','Decision', round((100*dtroc_auc),2)],
        ['blue','Rotation', round((100*rtroc_auc),2)],
        ['black','AdaBoost', round((100*adroc_auc),2)],
        ['darkorange','XGBoost', round((100*xgroc_auc),2)],
        ['green','Random', round((100*rfroc_auc),2)],
        ['magenta','Extra Tree', round((100*exroc_auc),2)]]

def sort_data(data):
    data.sort(key = lambda x:x[2])
    return data

data = sort_data(data)
colors=[]

for x in range(6):
    colors.append(data[x].pop(0))

plt.figure()
plt.plot(dtfpr,dttpr, color = 'red')
plt.plot(rffpr,rftpr, color = 'green')
plt.plot(rtfpr,rttpr, color = 'blue')
plt.plot(adfpr,adtpr, color = 'black')
plt.plot(exfpr,extpr, color = 'magenta')
plt.plot(xgfpr,xgtpr, color = 'darkorange')
plt.plot([0,1],[0,1], color = 'navy',linestyle = '--')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.table(cellText=data,cellLoc='left',
          rowColours=colors, colWidths=[0.2,0.15],
          loc=4).auto_set_font_size(False)
plt.text(0.65,0.38,"algorithm")
plt.text(0.9,0.38,"auc")
plt.title("Area Under Curve")
plt.show()

#%%
"""taking majority vote"""
labels = pd.DataFrame()
labels['dt'] = dt_predict
labels['rf'] = rf_predict
labels['rt'] = rt_predict
labels['adb'] = adb_predict
labels['ex'] = ex_predict
labels['xg'] = xg_predict
labels['lt'] = lt_predict

majority_vote = labels.mode(axis = 1)

cm_majority_vote = confusion_matrix(Y_test,majority_vote)

#%%
"""calculating weighted output"""
score_list = list([dt_score,rt_score,rf_score,adb_score,ex_score,xg_score,lt_score])
total = sum(score_list)

weights = []
for i in score_list:
    weights.append(i/total)

weights = pd.Series(weights)
weighted_output = np.dot(labels,weights)

weighted_output = np.round(weighted_output)
cm_weights = confusion_matrix(Y_test,weighted_output)

#%%
data = pd.DataFrame(X_train)
data['Landslide'] = Y_train
data2 = pd.DataFrame(X_test)
data2['Landslide'] = Y_test

data.to_csv("../csv_files/train.csv", index= False)
data2.to_csv("../csv_files/test.csv", index= False)
