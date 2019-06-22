import numpy as np
import csv
import pandas as pd
import math
import sys


def readfile(n):
	rawData=[]
	rawData=pd.read_csv(sys.argv[n])
	rawData=np.array(rawData).astype(float)
	return rawData

def normalize(trainX,x):
	for i in range(len(x[0])):
		mean = np.mean(trainX[:,i],axis = 0)
		std = np.std(trainX[:,i],axis = 0)
		x[:,i] = (x[:,i] - mean)/std
	return x


def sigmoid(z):
	seg = 1 / (1.0 + np.exp(-z))
	return np.clip(seg,1e-14,1-(1e-14))

def gaussian(x, cov_inv, n1, n2,mean1, mean2):
	w = np.dot((mean1-mean2),cov_inv)
	b = (-0.5)*np.dot(np.dot(mean1.transpose(),cov_inv),mean1) \
		+ (0.5)*np.dot(np.dot(mean2.transpose(),cov_inv),mean2) \
		+np.log(float(n1/n2))
	z = np.dot(w,x)+b
	y =sigmoid(z)
	return y	

def outputfile(n, ans):
	f = open(sys.argv[n],"w+")
	w = csv.writer(f)
	title = ['id','label']
	w.writerow(title) 
	for i in range(len(ans)):
		content = [i+1,int(ans[i])]
		w.writerow(content) 
	f.close()



def main():
	trainY = readfile(4)
	trainX = readfile(3)
	trainX = normalize(trainX,trainX)
		
	mean1 = np.zeros((len(trainX[0]),)) 
	mean2 = np.zeros((len(trainX[0]),)) 
	cov1 = np.zeros((len(trainX[0]),len(trainX[0]))) 
	cov2 = np.zeros((len(trainX[0]),len(trainX[0])))
	length1 = 0
	length2 = 0

	for i in range(len(trainX)):
		if trainY[i] == 1:
			mean1 += trainX[i]
			length1+=1
		else:
			mean2 += trainX[i]
			length2+=1

	mean1=mean1/float(length1)
	mean2=mean2/float(length2)

	for i in range(len(trainX)):
		if trainY[i] == 1:
			cov1 += np.dot(np.transpose([trainX[i]-mean1]),[(trainX[i]-mean1)])
		else:
			cov2 += np.dot(np.transpose([trainX[i]-mean2]),[(trainX[i]-mean2)])

	cov1=cov1/float(length1)
	cov2=cov2/float(length2)

	
	sharedCov = (float(length1)/len(trainX))*cov1+(float(length2)/len(trainX)*cov2)
	sharedCov_inv =  np.linalg.inv(sharedCov)
	
	correct = 0

	for i in range(len(trainX)):
		predict = gaussian(trainX[i], sharedCov_inv, length1, length2, mean1, mean2)
		if predict >= 0.5:
			temp=1
		else:
		 	temp=0
		if temp == trainY[i]:
			correct += 1
			accuracy=float(correct/len(trainX))
			print("Accuracy: %f" % accuracy)
	
	trainX = readfile(3)
	testX = readfile(5)
	testX =normalize(trainX,testX)
	result = []
	for i in range(len(testX)):
		predict = gaussian(testX[i], sharedCov_inv, length1, length2, mean1, mean2)
		if predict >= 0.5:
			result.append(1)
		else:
			result.append(0)
	outputfile(6,result)			


if __name__=="__main__":
	main()