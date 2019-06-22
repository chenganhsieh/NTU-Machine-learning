import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import csv
import numpy as np
from sys import argv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE= 64
FACE_SENTIMENT=7




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
	return x.to(device)

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

def conv_bn(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)


def conv_1x1_bn(inp, oup):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)


class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super(InvertedResidual, self).__init__()
		self.stride = stride

		self.use_res_connect = self.stride == 1 and inp == oup

		self.conv = nn.Sequential(
			# pw
			nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
			nn.BatchNorm2d(inp * expand_ratio),
			nn.ReLU(inplace=True),
			# dw
			nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
			nn.BatchNorm2d(inp * expand_ratio),
			nn.ReLU(inplace=True),
			# pw-linear
			nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
			nn.BatchNorm2d(oup),
		)

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)


class MobileNetV2(nn.Module):
	def __init__(self, n_class=FACE_SENTIMENT, input_size=44, width_mult=1.):
		super(MobileNetV2, self).__init__()
		# setting of inverted residual blocks
		self.interverted_residual_setting = [
			# t, c, n, s
			[1, 32, 1, 1],
			[6, 48, 2, 2],
			[6, 56, 4, 2],
			#[6, 64, 4, 2],
			#[6, 96, 3, 1],
			#[6, 160, 3, 2],
			#[6, 320, 1, 1],
		]

		# building first layer
		input_channel = int(11 * width_mult)
		self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
		self.features = [conv_bn(1, input_channel, 2)]
		# building inverted residual blocks
		for t, c, n, s in self.interverted_residual_setting:
			output_channel = int(c * width_mult)
			for i in range(n):
				if i == 0:
					self.features.append(InvertedResidual(input_channel, output_channel, s, t))
				else:
					self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
				input_channel = output_channel
		# building last several layers
		self.features.append(conv_1x1_bn(input_channel, self.last_channel))
		self.features.append(nn.AvgPool2d(int(input_size/22)))
		# make it nn.Sequential
		self.features = nn.Sequential(*self.features)

		# building classifier
		self.classifier = nn.Sequential(
			nn.Linear(11520, n_class),
			
		)


	def forward(self, x):
		x = self.features(x)
		#x = x.view(-1, self.last_channel)
		x = x.view(-1, 11520)
		x = self.classifier(x)
		return x


def main():
	print("==Loading Model==")
	net=MobileNetV2()
	net=net.to(device)
	#net.load_state_dict(torch.load('./mobilenet_ACC_63.18355973528387.pkl'))
	#state_dict = net.state_dict()
	#np.savez_compressed("./test/model_test.pkl",state_dict) 
	state_env=np.load("mobilenet_model.npz",allow_pickle=True)
	a = {key:state_env[key].item() for key in state_env}
	net.load_state_dict(a['arr_0'])
	net.eval()
	
	print("==Reading Data==")
	x_test= readTestData(argv[1])
	dataset_test=TestDataset(data=x_test,trainsform=Mytrainsform())
	x_test=DataLoader(dataset=dataset_test,batch_size=BATCH_SIZE,shuffle=False,num_workers=8)	
	ans = []
	correct = 0
	total = 0
	print('==Testing==')
	with torch.no_grad():
		for data in x_test:
			bs, ncrops, c, h, w = np.shape(data)
			data = data.view(-1, c, h, w)
			data= get_cuda(data)

			outputs = net(data).view(bs, ncrops, -1).mean(1)
			
			_, predicted = torch.max(outputs.data, 1)
			for i in range(predicted.size(0)):
				ans.append(int(predicted[i]))
		print('Done!')
	outputfile(argv[2], ans)

if __name__ == '__main__':
	main()