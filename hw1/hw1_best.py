import csv
import numpy as np
import math
import pandas as pd
import sys
feature=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
square=[3,4,7,8,9,12]
lamda=0
mean = np.load('mean.npy')
std=np.load('std.npy')
weight = np.load('model.npy')
test_x = [[]]
n_row=0

test_raw_data = np.genfromtxt(sys.argv[1], delimiter=',')  
test_data = test_raw_data[:, 2: ]
where_are_NaNs = np.isnan(test_data)
test_data[where_are_NaNs] = 0 

test_x = np.empty(shape = (240, (len(feature)+len(square)) * 9),dtype = float)



for month in range(12):
    for day in range(20):
    	temp=np.empty(shape=(0,len(feature)+len(square) * 9),dtype = float)
    	i=0
    	for k in feature:
    		for l in range(len(test_data[0])):
    			temp=np.append(temp,test_data[k + (month * 360 + 18*day),l])
    			i=i+1    	
    	for s in square:
    		for l in range(len(test_data[0])):
    			temp=np.append(temp,test_data[s + (month * 360 + 18*day),l]**2)
    			i=i+1
    	test_x[month*20+day,:] = temp
    


for i in range(test_x.shape[0]):       
    for j in range(test_x.shape[1]):
        if not std[j] == 0 :
            test_x[i][j] = (test_x[i][j]- mean[j]) / std[j]

test_x = np.concatenate((np.ones(shape = (test_x.shape[0],1)),test_x),axis = 1).astype(float)
answer = test_x.dot(weight)



f = open(sys.argv[2],"w")
w = csv.writer(f)
title = ['id','value']
w.writerow(title) 
for i in range(240):
    content = ['id_'+str(i),answer[i][0]]
    w.writerow(content) 
