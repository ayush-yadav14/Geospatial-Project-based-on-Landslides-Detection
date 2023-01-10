# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:59:57 2021
@author: vivek
"""
#%%
import pandas as pd
import numpy as np
#from osgeo import gdal
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelBinarizer

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
from sklearn.naive_bayes import GaussianNB
"""to import already saved train test splits"""
X_train = pd.read_csv('../csv/undersampling/0/X_train.csv')
y_train = pd.read_csv('../csv/undersampling/0/Y_train.csv')
X_test = pd.read_csv('../csv/undersampling/0/X_test.csv')
y_test = pd.read_csv('../csv/undersampling/0/Y_test.csv')

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

dt_score = grid_model.score(X_test, Y_test)
dt_score_train = grid_model.score(X_train, Y_train)
cm_dt = confusion_matrix(Y_test, grid_model.predict(X_test))
dt_predict  = grid_model.predict(X_test)
dt_probs = grid_model.predict_proba(X_test)
dtfpr = dict()
dttpr = dict()
dtroc_auc = dict()
dtfpr, dttpr, _ = roc_curve(Y_test, dt_probs[:,1], drop_intermediate = False,pos_label=2)
dtroc_auc = auc(dtfpr,dttpr)
print("DT auc")
print(grid_model.best_params_)
#Print out scores on validation set

print(grid_model.score(X_test,Y_test))
#print out scores on training data

print(grid_model.best_score_)

#%%
""""OPTIMISED DECISION TREE"""


#dt_model = DecisionTreeClassifier(random_state=1,min_samples_leaf=5,max_depth=5,min_samples_split=2)
dt_model = DecisionTreeClassifier(random_state=1,criterion = 'entropy', min_samples_leaf=6,min_samples_split=2) #for cl1 cl3
#dt_model = DecisionTreeClassifier(random_state=1, min_samples_leaf=6,min_samples_split=2) #for cl4
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
#import graphviz
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

rf_score = grid_model.score(X_test, Y_test)
rf_score_train = grid_model.score(X_train, Y_train)
cm_rf = confusion_matrix(Y_test, grid_model.predict(X_test))
rf_predict  = grid_model.predict(X_test)
rf_probs = grid_model.predict_proba(X_test)
rffpr = dict()
rftpr = dict()
rfroc_auc = dict()
rffpr, rftpr, _ = roc_curve(Y_test, rf_probs[:,1], drop_intermediate = False,pos_label=2)
rfroc_auc = auc(rffpr,rftpr)

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

#rf_model = RandomForestClassifier(random_state=1,n_estimators=500, max_depth=5,min_samples_split=5) #class1

#rf_model = RandomForestClassifier(random_state=1,n_estimators=500,min_samples_split=5,min_samples_leaf=1) #class2

#rf_model = RandomForestClassifier(random_state=1,n_estimators=500,min_samples_split=5,min_samples_leaf=2) #class3
rf_model = RandomForestClassifier(random_state=1,n_estimators=500,min_samples_split=6,min_samples_leaf=2) #class4

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

rt_score = grid_model.score(X_test, Y_test)
rt_score_train = grid_model.score(X_train, Y_train)
cm_rt = confusion_matrix(Y_test, grid_model.predict(X_test))
rt_predict  = grid_model.predict(X_test)
rt_probs = grid_model.predict_proba(X_test)
rtfpr = dict()
rttpr = dict()
rtroc_auc = dict()
rtfpr, rttpr, _ = roc_curve(Y_test, rt_probs[:,1], drop_intermediate = False,pos_label=2)
rtroc_auc = auc(rtfpr,rttpr)
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

#rt_model = RotationForestClassifier(random_state=1, max_depth = 5)#for cl1
#rt_model = RotationForestClassifier(random_state=1, n_estimators = 500, criterion = "gini" , max_depth = 5, min_samples_leaf=4,min_samples_split = 2) #cl2
#rt_model = RotationForestClassifier(random_state=1, n_estimators = 100, criterion = "gini" , max_depth = 5, min_samples_leaf=3,min_samples_split = 2) #cl3
rt_model = RotationForestClassifier(random_state=1, n_estimators = 500, criterion = "gini" , max_depth = 5, min_samples_leaf=4,min_samples_split = 2) #cl4
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

adb_score = adagrid_model.score(X_test, Y_test)
adb_score_train = adagrid_model.score(X_train, Y_train)
cm_ad = confusion_matrix(Y_test, grid_model.predict(X_test))
adb_predict  = adagrid_model.predict(X_test)
ad_probs = adagrid_model.predict_proba(X_test)
adfpr = dict()
adtpr = dict()
adroc_auc = dict()
adfpr, adtpr, _ = roc_curve(Y_test, ad_probs[:,1], drop_intermediate = False,pos_label=2)
adroc_auc = auc(adfpr,adtpr)

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

adb_model = AdaBoostClassifier(random_state=1, learning_rate=1,n_estimators=500,algorithm='SAMME')
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

ex_score = extragrid_model.score(X_test, Y_test)
ex_score_train = extragrid_model.score(X_train, Y_train)

cm_ex = confusion_matrix(Y_test, extragrid_model.predict(X_test))
ex_predict  = extragrid_model.predict(X_test)
ex_probs = extragrid_model.predict_proba(X_test)
exfpr = dict()
extpr = dict()
exroc_auc = dict()
exfpr, extpr, _ = roc_curve(Y_test, ex_probs[:,1], drop_intermediate = False,pos_label=2)
exroc_auc = auc(exfpr,extpr)

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

#ex_model = ExtraTreesClassifier(random_state=1,max_depth=5)#Cl1
ex_model = ExtraTreesClassifier(random_state=1,criterion='entropy',n_estimators=500,min_samples_split = 2, min_samples_leaf =2)
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
        'gamma':[0,1,2,3,4],
        'colsample_bytree':[0.8,0.9,1.0],
        'min_child_weight':[1,2,5],
        'subsample': [0.6,0.8,1.0]
        }

xg_model = XGBClassifier(random_state = 1)
xggrid_model = GridSearchCV(xg_model,params,scoring='roc_auc',verbose=1,n_jobs=-1)
xggrid_model.fit(X_train,Y_train)

xg_score = xggrid_model.score(X_test, Y_test)
xg_score_train = xggrid_model.score(X_train, Y_train)

cm_xg = confusion_matrix(Y_test, xggrid_model.predict(X_test))
xg_predict  = xggrid_model.predict(X_test)
xg_probs = xggrid_model.predict_proba(X_test)
xgfpr = dict()
xgtpr = dict()
xgroc_auc = dict()
xgfpr, xgtpr, _ = roc_curve(Y_test, xg_probs[:,1], drop_intermediate = False,pos_label=2)
xgroc_auc = auc(xgfpr,xgtpr)

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

xg_model = XGBClassifier(random_state=1, colsample_bytree=0.8, gamma = 4, max_depth=4, learning_rate=1,subsample = 1, min_child_weight = 1)
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
#optimizing logistic regression
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

#%% NB

nb_model = GaussianNB()
nb_model.fit(X_train, Y_train)

nb_score = nb_model.score(X_test,Y_test)
nb_score_train = nb_model.score(X_train,Y_train)

cm_nb = confusion_matrix(Y_test, nb_model.predict(X_test))
nb_predict  = nb_model.predict(X_test)
nb_probs = nb_model.predict_proba(X_test)
nbfpr = dict()
nbtpr = dict()
nbroc_auc = dict()
nbfpr, nbtpr, _ = roc_curve(Y_test, nb_probs[:,1], drop_intermediate = False,pos_label=2)
nbroc_auc = auc(nbfpr,nbtpr)

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
