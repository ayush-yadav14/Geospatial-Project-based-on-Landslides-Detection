# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:59:03 2021

@author: vivek
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sb

#%%
plt.figure()
a=[i/100 for i in range(100)]
i=0
base = [i+2 for i in range(9)]
color_arr = ['r','g','b','c','m','y','k','orange','grey']
for j in range(9):
    c=[]
    c.append(0)
    for i in range(99):
        k = (1-a[i+1])*math.log((1-a[i+1]),j+2)
        b = a[i+1]*math.log(a[i+1],2)
        b = (-1)*(b+k)
        c.append(b)
    i=['red',]
    #vv=color_arr[j]
    plt.plot(a,c,color = color_arr[j])
