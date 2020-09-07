# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:52:36 2020

@author: Ahsen Yavas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Salary_Data.csv') 
x = dataset.iloc[:, 0].values  
y = dataset.iloc[:, 1].values 

 
plt.scatter(x,y,color='red')
x_bias = np.ones((30,1))

x = np.reshape(x,(30,1))

x = np.append(x_bias,x,axis=1)

x_transpose = np.transpose(x)
x_transpose_dot_x = x_transpose.dot(x)

temp_1 = np.linalg.inv(x_transpose_dot_x)
temp_2=x_transpose.dot(y)
theta =temp_1.dot(temp_2)

print(theta)

         
y = 25792.2 +  9449.96*x  
plt.plot(x,y,color='blue')
plt.show()