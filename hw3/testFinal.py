import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
import csv
import numpy as np
from sys import argv



BATCH_SIZE= 256




def Mytrainsform():
	transform1 = transforms.Compose([
 	transforms.TenCrop(44),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
	])
	return transform1   

class TestDataset(Dataset):
	def __init__(self,data,target=None,trainsform=None):
		super(TestDataset,self).__init__()
		#self.data=torch.from_numpy(data).float()
		#self.target=torch.from_numpy(target).long()
		self.data=data
		#self.target=target.astype(np.long)
		self.trainsform=trainsform

	def __getitem__(self,index):
		x=self.data[index]
		#y=self.target[index]
		x=transforms. ToPILImage()(x.astype(np.uint8))

		if self.trainsform:
			x=self.trainsform(x)			
		return x.float()
	def __len__(self):
		return len(self.data)


def get_cuda(x):
	return x.cuda() if torch.cuda.is_available() else x

def outputfile(n, ans):
	f = open(n,"w+")
	w = csv.writer(f)
	title = ['id','label']
	w.writerow(title) 
	for i in range(len(ans)):
		content = [i,ans[i]]
		w.writerow(content) 
	f.close()
def readTestData(filename):
	
	feature = []
	file = open(filename, 'r')
	n_row = 0
	for row in csv.reader(file):
		if n_row > 0:
			feature.append([float(i) for i in row[1].split(' ')])
		n_row += 1

	feature = np.array(feature)
	feature = feature.reshape(feature.shape[0],48,48,1)
	return feature

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential( # input shape (1, 48, 48)
			nn.Conv2d(1, 64, 3, 1, 1),  # [64, 24, 24]
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),      # [64, 12, 12]

			nn.Conv2d(64, 128, 3, 1, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, 1, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),      # [128, 6, 6]

			nn.Conv2d(128, 256, 3, 1, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 256, 3, 1, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 256, 3, 1, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0), 

			nn.Conv2d(256, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0), 

			nn.Conv2d(512, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0), 

			
		  
			)

		self.fc = nn.Sequential(
			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(128, 7),
			)
		self.conv1.apply(gaussian_weights_init)
		self.fc.apply(gaussian_weights_init)
		'''							
		self.fc1=nn.Linear(12544, 1024)		
		self.fc2=nn.Linear(1024,512)
		self.fc3=nn.Linear(512,7)'''
		#self.fc4=nn.Linear(10,7)

	def forward(self, x):
		out = self.conv1(x)
		out = out.view(out.size()[0], -1)
		return self.fc(out)
def gaussian_weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1 and classname.find('Conv') == 0:
		m.weight.data.normal_(0.0, 0.02)

def main():
	cnn=CNN()
	cnn2=CNN()
	cnn3=CNN()
	cnn4=CNN()
	cnn5=CNN()
	#cnn6=CNN()
	#cnn7=CNN()
	if torch.cuda.is_available():
		cnn = cnn.cuda()
		cnn2=cnn2.cuda()
		cnn3=cnn3.cuda()
		cnn4=cnn4.cuda()
		cnn5=cnn5.cuda()
		#cnn6=cnn6.cuda()
		#cnn7=cnn7.cuda()
	#print(cnn)

		
	cnn.load_state_dict(torch.load('./cnn_best67.pkl'))	
	cnn2.load_state_dict(torch.load('./cnn_best66.98014629049112.pkl'))
	cnn3.load_state_dict(torch.load('./cnn_best67.3284569836294.pkl'))
	cnn4.load_state_dict(torch.load('./cnn_best67.04980842911877.pkl'))
	cnn5.load_state_dict(torch.load('./cnn_best67.43295019157088.pkl'))
	#cnn6.load_state_dict(torch.load('./vgg_7/cnn_best66.84082201323581.pkl'))
	#cnn7.load_state_dict(torch.load('./vgg_8/cnn_best66.87565308254963.pkl'))

	'''0.71579
	cnn.load_state_dict(torch.load('./vgg/cnn_best67.pkl'))	
	#cnn2.load_state_dict(torch.load('./vgg_6/cnn_best67.74642981539533.pkl'))
	cnn3.load_state_dict(torch.load('./vgg_5/cnn_best67.3284569836294.pkl'))
	cnn4.load_state_dict(torch.load('./vgg_4/cnn_best67.04980842911877.pkl'))
	cnn5.load_state_dict(torch.load('./vgg_1/cnn_best67.43295019157088.pkl'))
	'''
	cnn.eval()
	cnn2.eval()
	cnn3.eval()
	cnn4.eval()
	cnn5.eval()
	#cnn6.eval()
	#cnn7.eval()
	x_test= readTestData(argv[1])
	dataset_test=TestDataset(data=x_test,trainsform=Mytrainsform())
	x_test=DataLoader(dataset=dataset_test,batch_size=BATCH_SIZE,shuffle=False,num_workers=8)	
	ans = []
	correct = 0
	total = 0
	print('Testing...')
	with torch.no_grad():
		for data in x_test:
			bs, ncrops, c, h, w = np.shape(data)
			data = data.view(-1, c, h, w)
			data= get_cuda(data)

			outputs = cnn(data).view(bs, ncrops, -1)
			outputs2=cnn2(data).view(bs, ncrops, -1)
			outputs3=cnn3(data).view(bs, ncrops, -1)
			outputs4=cnn4(data).view(bs,ncrops,-1)
			outputs5=cnn5(data).view(bs,ncrops,-1)
			#output6=cnn6(data).view(bs,ncrops,-1)
			#output7=cnn7(data).view(bs,ncrops,-1)

			outputs=torch.cat([outputs,outputs2,outputs3,outputs4,outputs5],1)
			outputs_avg = outputs.mean(1)
			#outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
			#outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
			_, predicted = torch.max(outputs_avg.data, 1)
			for i in range(predicted.size(0)):
				ans.append(int(predicted[i]))
		print('Done!')
	outputfile(argv[2], ans)

if __name__ == '__main__':
	main()