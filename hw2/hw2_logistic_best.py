import numpy as np
import csv
import math
import sys
import pandas as pd
from numpy.linalg import inv
from sklearn import tree


def readfile(n):
	rawData=[]
	rawData=pd.read_csv(sys.argv[n])
	rawData=np.array(rawData).astype(float)
	return rawData
	
def feature(x):
	x1 = x[:,[0]]**2
	x2 = x[:,[10]]**2
	x3 = x[:,[78]]**2
	x4 = x[:,[79]]**2
	x5 = x[:,[80]]**2
	x6 = x[:,[0]]**3
	x7 = x[:,[80]]**3
	x8 = x[:,[10]]**3
	x9= x[:,[78]]**3
	x10 = x[:,[79]]**3
	x11 = x[:,[0]]**4
	x12 = x[:,[80]]**4
	x13 = x[:,[78]]**5
	x14 = x[:,[79]]**5
	x = np.concatenate((x,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14),axis = 1)
	return x
	
def sigmoid(z):
	seg = 1 / (1.0 + np.exp(-z))
	return np.clip(seg,1e-14,1-(1e-14))
	

def calaccuracy(w,x,y):
	correct = 0.0
	for i in range(len(x)):
		z = np.dot(x[i],w)
		predict = 1.0 if z >= 0.0 else 0.0
		correct += 1 if predict==y[i] else 0
	return correct




def normalize(trainX,x):
	for i in range(len(x[0])):
		mean = np.mean(trainX[:,i],axis = 0)
		std = np.std(trainX[:,i],axis = 0)
		x[:,i] = (x[:,i] - mean)/std
	return x

def outputfile(n, ans):
	f = open(sys.argv[n],"w+")
	w = csv.writer(f)
	title = ['id','label']
	w.writerow(title) 
	for i in range(len(ans)):
		content = [i+1,ans[i]]
		w.writerow(content) 
	f.close()


def main():
	
	w=np.load('model_logistic.npy')
	ans = []	
	trainX = readfile(3)
	trainX = feature(trainX)
	testX = readfile(5)
	testX = feature(testX)
	testX = normalize(trainX,testX)
	testX = np.concatenate((np.ones((testX.shape[0],1)),testX), axis=1)

	for i in range(len(testX)):
		predict = np.dot(testX[i],w) 	
		if sigmoid(predict) >= 0.5:
			ans.append(1)
		else:
			ans.append(0) 
	outputfile(6,ans)


if __name__=="__main__":
	main()