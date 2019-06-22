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
from matplotlib import cm
from PIL import Image
import random
from sys import argv
from tqdm import *
#等著轉成pil

EPOCH=300
BATCH_SIZE= 256
LR = 0.001
FACE_SENTIMENT=7


def readData(filename):
	x_train = []
	y_train = []
	x_val = []
	y_val = []
	label = []
	feature = []
	file = open(filename, 'r')
	n_row = 0
	for row in csv.reader(file):
		if n_row > 0:
			label.append(float(row[0]))
			feature.append([float(i) for i in row[1].split(' ')])
		n_row += 1

	label = np.array(label) 
	feature = np.array(feature)
	delete=[]
	for i in range(feature.shape[0]):		
		if (5>feature[i]).all() or (250<feature[i]).all() or np.mean(feature[i])<40:
			delete.append(i)
	print(len(delete))
	feature=np.delete(feature,delete,axis=0)
	label=np.delete(label,delete,axis=0)
	print(feature.shape[0])
	feature = feature.reshape(feature.shape[0],48,48,1)

	x_train = feature[2871:]
	x_val = feature[0:2871]
	y_train = label[2871:]
	y_val = label[0:2871]
	x_train=x_train
	x_val=x_val




	x_train_mr = np.flip(x_train,axis=2)
	x_train = np.concatenate((x_train,x_train_mr),axis=0)
	y_train = np.concatenate((y_train,y_train),axis=0)

	return x_train, y_train, x_val, y_val


def Mytrainsform():
	transform1 = transforms.Compose([
	#transforms.ToPILImage(),

	#transforms.RandomHorizontalFlip(),	
	#transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
	#transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
	transforms.TenCrop(44),
	transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
	])
	return transform1    


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
		'''
		out = self.conv1(x)	
		
		out = out.view(out.size(0), -1)  # reshape
		out = F.relu(self.fc1(out))
		out= torch.nn.Dropout(p=0.5)(out)
		out = F.relu(self.fc2(out))
		out= torch.nn.Dropout(p=0.5)(out)
		out = self.fc3(out)'''
		#out= torch.nn.Dropout(p=0.5)(out)
		#out = self.fc4(out)
		#return out

class MyDataset(Dataset):
	def __init__(self,data,target=None,trainsform=None):
		super(MyDataset,self).__init__()
		#self.data=torch.from_numpy(data).float()
		#self.target=torch.from_numpy(target).long()
		self.data=data
		self.target=target.astype(np.long)
		self.trainsform=trainsform

	def __getitem__(self,index):
		x=self.data[index]
		y=self.target[index]

		x=transforms. ToPILImage()(x.astype(np.uint8))
		if self.trainsform:
			#x = x.resize((40, 40))
			x=self.trainsform(x)
		else:
			x=self.transform(x)	
		return x.float(),y

	def transform(self, image):

		# Random crop
		#i, j, h, w = transforms.RandomCrop.get_params(
		#	image, [40, 40])
		#image = transforms.functional.crop(image, i, j, h, w)
		#if random.random() > 0.5:
		#image=transforms.RandomAffine(degrees=20, translate=[0.1,0.1])(image),
		#image=transforms.CenterCrop(46)(image)
		image=transforms.RandomCrop(44)(image)
		image=transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0)(image)		
		image=transforms.RandomRotation(20)(image)
		# Random horizontal flipping
		#if random.random() > 0.5:
		#	image = transforms.functional.hflip(image)

		# Random vertical flipping
		#if random.random() > 0.5:
		#	image = transforms.functional.vflip(image)
		# Transform to tensor
		#image = image.resize((48, 48))
		image = transforms.functional.to_tensor(image)
		return image

	def __len__(self):
		return len(self.data)
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


def outputfile(n, ans):
	f = open(n,"w+")
	w = csv.writer(f)
	title = ['id','label']
	w.writerow(title) 
	for i in range(len(ans)):
		content = [i,ans[i]]
		w.writerow(content) 
	f.close()


def get_cuda(x):
	return x.cuda() if torch.cuda.is_available() else x
def gaussian_weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1 and classname.find('Conv') == 0:
		m.weight.data.normal_(0.0, 0.02)
def outputGraph(n, epo,ans):
	f = open(n,"w+")
	w = csv.writer(f)
	title = ['EPOCH','Loss']
	w.writerow(title) 
	for i in range(len(ans)):
		content = [epo[i],ans[i]]
		w.writerow(content) 
	f.close()

def main():
	cnn = CNN()
	if torch.cuda.is_available():
		cnn = cnn.cuda()
	print(cnn)

#	pytorch_params = sum(s.numel() for s in cnn.parameters() if s.requires_grad)
#	print(pytorch_params)
	
	x_train, y_train, x_val, y_val = readData(argv[1])

	
	dataset_train=MyDataset(data=x_train,target=y_train)
	dataset_test=MyDataset(data=x_val,target=y_val,trainsform=Mytrainsform())
	print(dataset_train.__len__())
	
	train_loader = DataLoader(dataset=dataset_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=8)
	x_val=DataLoader(dataset=dataset_test,batch_size=BATCH_SIZE,shuffle=False,num_workers=8)
	
	
	loss_func = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(cnn.parameters(), lr=LR,weight_decay=5e-4)
	#accuracyBest=0

	lossArray=[]
	epochArray=[]
	for epoch in tqdm(range(EPOCH)):
		cnn.train()
		correct = 0
		total = 0
		for i,(data,target) in enumerate(train_loader) :
			data = get_cuda(data)
			target = get_cuda(target)
			
			
			optimizer.zero_grad()
			outputs = cnn(data)
			
			
			loss = loss_func(outputs, target)
			loss.backward()
			optimizer.step()


			_, predicted = torch.max(outputs.data, 1)
			total += target.size(0)
			correct_pred = (predicted==target.data).sum()
			correct += correct_pred

			if (i + 1) % 100 == 0:
				
				tqdm.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
				% (epoch + 1, EPOCH, i + 1, len(dataset_train) // BATCH_SIZE, loss.item()))
			
		tqdm.write('	train accuracy: %d %%' % (100 * correct / total))		

		cnn.eval() 

		correct = 0.0
		total = 0.0
		with torch.no_grad():
			for data,target in x_val:
				bs, ncrops, c, h, w = np.shape(data)
				data = data.view(-1, c, h, w)
				data= get_cuda(data)
				target = get_cuda(target)
				outputs = cnn(data)
				outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

				_, predicted = torch.max(outputs_avg.data, 1)			
				total += target.size(0)
				correct += (predicted == target.data).sum()
	
		
		tqdm.write('	test accuracy: %d %%' % (100 * correct.item() / total))
		if (20<(100 * correct / total)):
			#accuracyBest=100 * correct / total
			torch.save(cnn.state_dict(), './cnn_Acc'+str(float(100 * correct.item() / total))+'.pkl')
		lossArray.append(loss.item())
		epochArray.append(epoch)
	#outputGraph("./VGGgraph5.csv", epochArray,lossArray)
	'''
	x_test= readTestData('test.csv')
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

			outputs = cnn(data)
			outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
			_, predicted = torch.max(outputs_avg.data, 1)
			for i in range(predicted.size(0)):
				ans.append(int(predicted[i]))
		print('Done!')
	outputfile('predictinBest_vgg.csv', ans)'''

'''
	cnn.eval() 
	correct = 0
	total = 0
	with torch.no_grad():
		for data,target in x_val:
			data= get_variable(data)
			target = get_variable(target)
			outputs = cnn(data)
			_, predicted = torch.max(outputs.data, 1)	
			print(predicted)		
			total += target.size(0)
			correct += (predicted == target.data).sum()

		print('	test accuracy: %d %%' % (100 * correct / total))'''
	#torch.save(cnn.state_dict(), 'cnn.pkl')	





if __name__ == '__main__':
	main()

'''
nn.Conv2d(1, 32, 3, 1, 1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.2),
			nn.MaxPool2d(2, 2, 1),


			nn.Conv2d(32, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.3),
			nn.Conv2d(64, 128, 3, 1, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2, 1),
			nn.Dropout(p=0.3),

			nn.Conv2d(128, 256, 3, 1, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, 1, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2, 1),
			nn.Dropout(p=0.3),
'''