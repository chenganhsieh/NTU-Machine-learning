import numpy as np
import csv
import math
import sys
import pandas as pd

from sklearn import tree

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def testfileX(n):
	rawData=[]
	rawData=pd.read_csv(sys.argv[n])
	rawData = rawData.drop(['fnlwgt','native_country'], axis=1)
	rawData = pd.get_dummies(rawData, columns=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex'])
	rawData=np.array(rawData).astype(float)
	return rawData
def outputfile(n, ans):
	f = open(sys.argv[n],"w+")
	w = csv.writer(f)
	title = ['id','label']
	w.writerow(title) 
	for i in range(len(ans)):
		content = [i+1,int(ans[i])]
		w.writerow(content) 
	f.close()
def randomdecisionTree(trainX,trainY,testX):
	print("start!")
	param_grid = {}
	
	clf= GridSearchCV(RandomForestClassifier(n_estimators=90,max_depth=13,min_samples_split=50,oob_score = True, random_state = 42,max_features='auto'), param_grid,cv=3)
	#clf=RandomForestClassifier(oob_score=True,random_state=10)
	#max_depth=13,min_samples_split=50
	clf.fit(trainX,trainY.ravel())	
	
	print(clf.best_params_)
	predictTrain=clf.predict(trainX)
	correct=0
	for i in range(len(trainY)):
		if trainY[i]==predictTrain[i]:
			correct=correct+1
	print(correct/len(trainY))
	joblib.dump(clf,'clf.pkl', compress=1)
	predict=clf.predict(testX)
	print(predict)
	return predict 

def main():
	testX=testfileX(2)
	clf=joblib.load("clf.pkl")

	randomPredict=clf.predict(testX)
	outputfile(6,randomPredict)

if __name__=="__main__":
	main()