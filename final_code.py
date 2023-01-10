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
from sklearn.preprocessing import MinMaxScaler
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
import pickle
import graphviz
from sklearn import tree

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


d = {'Latitude' :lat, 'Longitude' :long, 'Aspect' :ASPECT, 'Buffer Fault2' :BUFFER_FAULT2,
     'Buffer River2' :BUFFER_RIVER2, 'Buffer Road2' :BUFFER_ROAD2, 'Dem' :DEM, 'Lithology' :LITHO,
     'Plan Curvature' :PLAN_CURVETURE, 'Profile Curvature' :PROFILE_CURVETURE, 
     'Slope Length' :SLOPE_LENGTH, 'Slope' :SLOPEW, 'SPI' :SPI, 'STI' :STI, 'TWI' :TWI, 
     'Land Cover' :LAND_COVER, 'Landslide' :LANDSLIDE}

df = pd.DataFrame(d, columns =['Latitude', 'Longitude', 'Aspect', 'Buffer Fault2' , 'Buffer River2' , 'Buffer Road2' ,
                               'Dem', 'Lithology', 'Plan Curvature','Profile Curvature',
                               'Slope Length', 'Slope', 'SPI', 'STI', 'TWI', 'Land Cover','Landslide'] )

df.to_csv('../csv/final_complete_data.csv',index=False)

#%%

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
print(df['Buffer Fault2'].value_counts())
print(df['Buffer River2'].value_counts())
print(df['Buffer Road2'].value_counts())
print(df['Dem'].value_counts())
print(df['Lithology'].value_counts())
print(df['Plan Curvature'].value_counts()) 
print(df['Profile Curvature'].value_counts())
print(df['Slope Length'].value_counts())
print(df['Slope'].value_counts()) 
print(df['SPI'].value_counts()) 
print(df['STI'].value_counts())
print(df['TWI'].value_counts())
print(df['Land Cover'].value_counts())
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

indexNames = df[df['Buffer Fault2'] == drop_value].index


"""from the original dataframe, drop all the rows of index in indexNames"""
df.drop(indexNames , inplace = True)


#write the dataframe into a csv file
df.to_csv('../csv/truncated_data.csv', index = False)

#%%
""""multi collinearity check"""

"""vif values between 1-5 have low collinearity
                5-10 , moderate
                >10, high collinearity
                """
df_new = df.iloc[:,2:16]
vif = pd.DataFrame()
vif['vif factor'] = [variance_inflation_factor(df_new.values, i) for i in range(df_new.shape[1])]
vif['features'] = df_new.columns

#%%

""""since data is unbalanced, stratify the data without replacement"""

df1 = df.Landslide[df.Landslide.eq(2)].index
df2 = df.Landslide[df.Landslide.eq(1)].sample(df1.shape[0],replace=False).index

df_new = df1.union(df2)
df_final= df.loc[df_new]
df_final.to_csv('../csv/startified_data.csv', index = False)

#%%
"""to read stratified csv file directly"""

df_final = pd.read_csv("../csv/stratified_data.csv")


#%%

""""preprocessing for classification"""

X = df_final.iloc[:,2:16]
Y = df_final.iloc[:,16]

features = []
for i in X:
    features.append(i)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=1,test_size = 0.3)

#%%
"""DECISION TREE OPTIMISATION

model = DecisionTreeClassifier(random_state = 1,criterion='entropy')
model.fit(X_train,Y_train)
print(model.score(X_test, Y_test))
print(model.score(X_train,Y_train))
#criterion:max at default, gini

model = DecisionTreeClassifier(random_state = 1,splitter='random')
model.fit(X_train,Y_train)
print(model.score(X_test, Y_test))
print(model.score(X_train,Y_train))
#splitter: max at default, best

test_accuracy = []
train_accuracy = []
for i in range(2,20):
    model = DecisionTreeClassifier(random_state=1, max_depth=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#max_depth: max at 11
    
    
min_samples_split = list(range(2,101))
test_accuracy = []
train_accuracy = []
for i in min_samples_split:
    model = DecisionTreeClassifier(random_state=1, max_depth=11, min_samples_split=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#min_samples_split: max at default,2
    
min_samples_leaf = list(range(1,101))
test_accuracy = []
train_accuracy = []
for i in min_samples_leaf:
    model = DecisionTreeClassifier(random_state=1, max_depth=11,min_samples_leaf=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#min_samples_leaf: max at deafult,1
    
test_accuracy = []
train_accuracy = []
for i in range(0,6):
    model = DecisionTreeClassifier(random_state=1, max_depth=11,
                                   min_weight_fraction_leaf=float(i/10))
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#max at default, 0
    
max_features=list(range(1,X_train.shape[1]+1))
max_features.append('sqrt')
max_features.append('log2')
max_features.append(None)

test_accuracy = []
train_accuracy = []
for i in max_features:
    model = DecisionTreeClassifier(random_state=1,  max_depth=11,max_features=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#max_features: max at default:None
    


model = DecisionTreeClassifier(random_state = 1,  max_depth=11,
                               class_weight='balanced')
model.fit(X_train,Y_train)
print(model.score(X_test, Y_test))
print(model.score(X_train,Y_train))
#class_weight: max at default, None
"""


#%%
""""OPTIMISED DECISION TREE"""


dt_model = DecisionTreeClassifier(random_state=1)
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

#%%
"""save dt model"""

pickle.dump(dt_model, open('decison_model.sav', 'wb'))

"""load saved model"""

load_dt_model = pickle.load(open('decision_model.sav', 'rb'))

#%%
"""Render decision tree"""
output = ['0','1']
dot_data = tree.export_graphviz(dt_model, out_file = None, filled = True, 
                                rounded = True, special_characters = True, 
                                class_names= output, feature_names=features)
graph = graphviz.Source(dot_data)
graph.render("landslide_dt")

"""will save the graph in pwd"""

#%%
"""RANDOM FOREST OPTIMISATION

test_accuracy = []
train_accuracy = []
for i in range(100,1001,100):
    model = RandomForestClassifier(random_state=1, n_estimators=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#n_estimators: max at 100 default
    

test_accuracy = []
train_accuracy = []
for i in range(2,21):
    model = RandomForestClassifier(random_state=1, max_depth=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#max_depth: max at default,None
    
min_samples_split = list(range(2,25))
test_accuracy = []
train_accuracy = []
for i in min_samples_split:
    model = RandomForestClassifier(random_state=1,min_samples_split=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#min_samples_split: max at 10
    
min_samples_leaf = list(range(1,25))
test_accuracy = []
train_accuracy = []
for i in min_samples_leaf:
    model = RandomForestClassifier(random_state=1, min_samples_split=10,
                                   min_samples_leaf=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#min_samples_leaf: max at default,1
    
test_accuracy = []
train_accuracy = []
for i in range(0,6):
    model = RandomForestClassifier(random_state=1, 
                                   min_weight_fraction_leaf=float(i/10))
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#min_weight_fraction_leaf: max at default, 0
    
max_features=list(range(1,X_train.shape[1]+1))
max_features.append('sqrt')
max_features.append('log2')
max_features.append(None)

test_accuracy = []
train_accuracy = []
for i in max_features:
    model = RandomForestClassifier(random_state=1,min_samples_split=10,
                                   max_features=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#max_features: max at default,None
    
model = RandomForestClassifier(random_state = 1,min_samples_split=10, bootstrap= False)
model.fit(X_train,Y_train)
print(model.score(X_test, Y_test))
print(model.score(X_train,Y_train))
#for bootstrap, max at default, True

model = RandomForestClassifier(random_state = 1, min_samples_split=10,class_weight='balanced')
model.fit(X_train,Y_train)
print(model.score(X_test, Y_test))
print(model.score(X_train,Y_train))
#class_weight: max at default,None

test_accuracy =[]
train_accuracy =[]
for i in range(1,X_train.shape[0]+1):
    model = RandomForestClassifier(random_state = 1,max_samples=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test, Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#mat at default, None 
"""


#%%
"""OPTIMISED RANDOM FOREST"""

rf_model = RandomForestClassifier(random_state=1,min_samples_split=10)
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
"""ROTATION FOREST OPTIMISATION

test_accuracy = []
train_accuracy = []
for i in range(10,201,10):
    model = RotationForestClassifier(random_state=1,n_estimators=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#n_estimators: max at default, 10

model = RotationForestClassifier(random_state = 1,criterion='entropy')
model.fit(X_train,Y_train)
print(model.score(X_test, Y_test))
print(model.score(X_train,Y_train))
#criterion: max at default, gini

test_accuracy = []
train_accuracy = []
for i in range(1,X_train.shape[1]+1):
    model = RotationForestClassifier(random_state=1,n_features_per_subset=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#n_features_per_subset: max at default,3
    

test_accuracy = []
train_accuracy = []
for i in range(2,11):
    model = RotationForestClassifier(random_state=1, max_depth=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#max_depth: max at default, None
    
min_samples_split = list(range(2,25))
test_accuracy = []
train_accuracy = []
for i in min_samples_split:
    model = RotationForestClassifier(random_state=1, min_samples_split=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#min_samples_split: max at default,2
    
min_samples_leaf = list(range(1,25))
test_accuracy = []
train_accuracy = []
for i in min_samples_leaf:
    model = RotationForestClassifier(random_state=1, min_samples_leaf=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#min_samples_leaf: max at 10
    
test_accuracy = []
train_accuracy = []
for i in range(0,6):
    model = RotationForestClassifier(random_state=1, 
                                   min_weight_fraction_leaf=float(i/10))
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#max at default, 0
    
max_features=list(range(1,X_train.shape[1]+1))
max_features.append('sqrt')
max_features.append('log2')
max_features.append(None)
max_features.append(1.0)

test_accuracy = []
train_accuracy = []
for i in max_features:
    model = RotationForestClassifier(random_state=1,min_samples_leaf=10,
                                     max_features=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#max_features: max at 9
    
model = RotationForestClassifier(random_state=1, bootstrap= True)
model.fit(X_train,Y_train)
print(model.score(X_test, Y_test))
print(model.score(X_train,Y_train))
#for bootstrap, max at default, False

model = RotationForestClassifier(random_state=1, class_weight='balanced')
model.fit(X_train,Y_train)
print(model.score(X_test, Y_test))
print(model.score(X_train,Y_train))
#class_weight: max at default, None 
"""
#%%
"""OPTIMISED ROTATION FOREST"""

rt_model = RotationForestClassifier(random_state=1,min_samples_leaf=10,
                                     max_features=9)
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
"""ADABOOST OPTIMISATION 
test_accuracy=[]
train_accuracy=[]
for i in range(1,30,1):
    model = AdaBoostClassifier(random_state=1,
                                learning_rate=float(i/10))
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#learning rate: max at default,1

test_accuracy=[]
train_accuracy=[]
for i in range(50,301,10):
    model = AdaBoostClassifier(random_state=1,n_estimators=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#n_estimators: max at default,50

model = AdaBoostClassifier(random_state=1, algorithm='SAMME')
model.fit(X_train,Y_train)
print(model.score(X_test,Y_test))
print(model.score(X_train,Y_train))
#algorithm: max at default, SAMME.R

"""
#%%
"""OPTIMISED AdaBoost Classifier"""

adb_model = AdaBoostClassifier(random_state=1)
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
"""EXTRA TREE OPTIMISATION
test_accuracy = []
train_accuracy = []
for i in range(10,150,10):
    model = ExtraTreesClassifier(random_state=1, n_estimators =i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#n_estimators: max at default,10

model = ExtraTreesClassifier(random_state=1, criterion='entropy')
model.fit(X_train,Y_train)
print(model.score(X_test,Y_test))
print(model.score(X_train,Y_train))
#criterion: max at default, gini
    
test_accuracy = []
train_accuracy = []
for i in range(2,11):
    model = ExtraTreesClassifier(random_state=1, max_depth=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#max_depth: max at default, None
    
    
min_samples_split = list(range(2,25))
test_accuracy = []
train_accuracy = []
for i in min_samples_split:
    model = ExtraTreesClassifier(random_state=1,min_samples_split=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#min_samples_split: max at default,2
    
min_samples_leaf = list(range(1,25))
test_accuracy = []
train_accuracy = []
for i in min_samples_leaf:
    model = ExtraTreesClassifier(random_state=1, min_samples_leaf=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#min_samples_leaf: max at default,1
    
test_accuracy = []
train_accuracy = []
for i in range(0,6):
    model = ExtraTreesClassifier(random_state=1,
                                   min_weight_fraction_leaf=float(i/10))
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#max at default, 0
    
max_features=list(range(1,X_train.shape[1]+1))
max_features.append('sqrt')
max_features.append('log2')
max_features.append(None)

test_accuracy = []
train_accuracy = []
for i in max_features:
    model = ExtraTreesClassifier(random_state=1, max_features=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#max_features: max at default,auto(sqrt)    
    
model = ExtraTreesClassifier(random_state=1, bootstrap= True)
model.fit(X_train,Y_train)
print(model.score(X_test, Y_test))
print(model.score(X_train,Y_train))
#for bootstrap, max at default, False

model = ExtraTreesClassifier(random_state=1,
                             class_weight='balanced_subsample')
model.fit(X_train,Y_train)
print(model.score(X_test, Y_test))
print(model.score(X_train,Y_train))
#class_weigth: max at default, None


test_accuracy =[]
train_accuracy =[]
for i in range(1,X_train.shape[0]+1):
    model = ExtraTreesClassifier(random_state=1,max_samples=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test, Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#mat at default, None 
"""

#%%
"""OPTIMISED Extra Tree Classifier"""

ex_model = ExtraTreesClassifier(random_state=1)
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
"""OPTIMISING XGBOOST
booster = ['gbtree','gblinear','dart']
test_accuracy=[]
train_accuracy=[]
for i in booster:
    model = XGBClassifier(random_state=1,booster = i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#booster: max at default, gbtree
    
learning_rate = list(range(1,101))
learning_rate=np.divide(learning_rate,100)
test_accuracy=[]
train_accuracy=[]
for i in learning_rate:
    model = XGBClassifier(random_state=1, learning_rate=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#learning_rate: max at default, 0.3

test_accuracy=[]
train_accuracy=[]    
for i in range(100,1001,100):
    model = XGBClassifier(random_state=1, n_estimators=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#n_estimators: max at default, 100

test_accuracy=[]
train_accuracy=[]    
for i in range(30):
    model = XGBClassifier(random_state=1, min_split_loss=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#min_split_loss: max at default, 0

test_accuracy=[]
train_accuracy=[]    
for i in range(1,10):
    model = XGBClassifier(random_state=1, max_depth =i+1)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#max_depth: max at default,6
    
test_accuracy=[]
train_accuracy=[]    
for i in range(25):
    model = XGBClassifier(random_state=1, min_child_weigth=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#min_child_weigth max at default,1

sub_sample = list(range(1,100,5))
sub_sample=np.divide(sub_sample,100)
test_accuracy=[]
train_accuracy=[]
for i in sub_sample:
    model = XGBClassifier(random_state=1,subsample=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#subsample: max at default,1
    
colsample_by = list(range(1,101))
colsample_by=np.divide(colsample_by,100)
test_accuracy=[]
train_accuracy=[]
for i in colsample_by:
    model = XGBClassifier(random_state=1,colsample_bytree=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#colsample_bytree: max at default, 1    

colsample_by = list(range(1,101))
colsample_by=np.divide(colsample_by,100)
test_accuracy=[]
train_accuracy=[]
for i in colsample_by:
    model = XGBClassifier(random_state=1, colsample_bylevel=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#colsample_bylevel: max at default, 1    

colsample_by = list(range(1,101))
colsample_by=np.divide(colsample_by,100)
test_accuracy=[]
train_accuracy=[]
for i in colsample_by:
    model = XGBClassifier(random_state=1,colsample_bynode=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#colsample_bynode: max at default, 1    

reg_alpha = list(range(0,101))
reg_alpha=np.divide(reg_alpha,100)
test_accuracy=[]
train_accuracy=[]
for i in reg_alpha:
    model = XGBClassifier(random_state=1,reg_alpha=i)
    model.fit(X_train,Y_train)
    test_accuracy.append(model.score(X_test,Y_test))
    train_accuracy.append(model.score(X_train,Y_train))
#reg_alpha: max at default, 0   

"""

#%%
"""OPTIMISED implement XGBoost"""

xg_model = XGBClassifier(random_state=1)
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

data = [['red','DT', round((100*dtroc_auc),2)],
        ['blue','Rot Forest', round((100*rtroc_auc),2)],
        ['black','AdaBoost', round((100*adroc_auc),2)],
        ['darkorange','XGBoost', round((100*xgroc_auc),2)],
        ['green','RF', round((100*rfroc_auc),2)],
        ['magenta','Extra Tree', round((100*exroc_auc),2)]
        ['cyan','Logit', round((100*ltroc_auc),2)]]

def sort_data(data):
    data.sort(key = lambda x:x[2])
    return data

data = sort_data(data)
colors=[]

for x in range(7):
    colors.append(data[x].pop(0))

plt.figure()
plt.plot(dtfpr,dttpr, color = 'red')
plt.plot(rffpr,rftpr, color = 'green')
plt.plot(rtfpr,rttpr, color = 'blue')
plt.plot(adfpr,adtpr, color = 'black')
plt.plot(exfpr,extpr, color = 'magenta')
plt.plot(xgfpr,xgtpr, color = 'darkorange')
plt.plot(ltfpr,lttpr, color = 'cyan')
plt.plot([0,1],[0,1], color = 'navy',linestyle = '--')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.table(cellText=data,cellLoc='left',
          rowColours=colors, colWidths=[0.2,0.15],
          loc=4).auto_set_font_size(False)
plt.text(0.65,0.38,"algorithm")
plt.text(0.9,0.38,"auc")
plt.title("Mushroom")
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

