import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import csv
import numpy as np
from PIL import Image
import random
from sys import argv
from collections import OrderedDict
import math

EPOCH=500
BATCH_SIZE= 64
LR = 0.001
FACE_SENTIMENT=7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
	transforms.TenCrop(44),
	transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
	])
	return transform1    
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
	def __init__(self, inp, oup, stride, expand):
		super(InvertedResidual, self).__init__()
		self.stride = stride

		self.use_res_connect = self.stride == 1 and inp == oup

		self.conv = nn.Sequential(
			nn.Conv2d(inp, inp * expand, 1, 1, 0, bias=False),
			nn.BatchNorm2d(inp * expand),
			nn.ReLU(inplace=True),
			nn.Conv2d(inp * expand, inp * expand, 3, stride, 1, groups=inp * expand, bias=False),
			nn.BatchNorm2d(inp * expand),
			nn.ReLU(inplace=True),
			nn.Conv2d(inp * expand, oup, 1, 1, 0, bias=False),
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
			#[6, 22, 4, 2],
			#[6, 24, 3, 1],
			#[6, 160, 3, 2],
			#[6, 320, 1, 1],
			#12 16 20

			#[1, 12, 1, 1],
			#[6, 16, 2, 2],
			#[6, 24, 4, 2], best!!
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
		#print(x.shape)
		#x = x.view(-1, self.last_channel)
		x = x.view(-1, 11520)
		x = self.classifier(x)
		return x
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

		image=transforms.RandomCrop(44)(image)
		image=transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0)(image)		
		image=transforms.RandomRotation(20)(image)
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
	print("==Loading Model==")
	mobilenet = MobileNetV2()
	mobilenet=mobilenet.to(device)
	print(mobilenet)

#	pytorch_params = sum(s.numel() for s in cnn.parameters() if s.requires_grad)
#	print(pytorch_params)
	print("==Reading Data==")
	x_train, y_train, x_val, y_val = readData(argv[1])
	print(y_train)
	
	
	dataset_train=MyDataset(data=x_train,target=y_train)
	dataset_test=MyDataset(data=x_val,target=y_val,trainsform=Mytrainsform())

	
	train_loader = DataLoader(dataset=dataset_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=8)
	x_val=DataLoader(dataset=dataset_test,batch_size=BATCH_SIZE,shuffle=False,num_workers=8)
	
	
	loss_func = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(mobilenet.parameters(), lr=LR,weight_decay=5e-4)
	#accuracyBest=0
	print("==Training==")
	lossArray=[]
	epochArray=[]
	for epoch in range(EPOCH):

		mobilenet.train()
		correct = 0
		total = 0
		for i,(data,target) in enumerate(train_loader) :
			data = get_cuda(data)
			target = get_cuda(target)
			
			
			optimizer.zero_grad()
			outputs = mobilenet(data)
			
			
			loss = loss_func(outputs, target)
			loss.backward()
			
			optimizer.step()


			_, predicted = torch.max(outputs.data, 1)
			total += target.size(0)
			correct_pred = (predicted==target.data).sum()
			correct += correct_pred

			if (i + 1) % 100 == 0:
				
				print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
				% (epoch + 1, EPOCH, i + 1, len(dataset_train) // BATCH_SIZE, loss.item()))
			
		print('	train accuracy: %d %%' % (100 * correct / total))		

		mobilenet.eval() 

		correct = 0.0
		total = 0.0
		with torch.no_grad():
			for data,target in x_val:
				bs, ncrops, c, h, w = np.shape(data)
				data = data.view(-1, c, h, w)
				data= get_cuda(data)
				target = get_cuda(target)
				outputs = mobilenet(data)
				outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

				_, predicted = torch.max(outputs_avg.data, 1)			
				total += target.size(0)
				correct += (predicted == target.data).sum()
	
		
		print('	test accuracy: %d %%' % (100 * correct.item() / total))
		if (50<(100 * correct / total)):
			#accuracyBest=100 * correct / total
			#torch.save(mobilenet.state_dict(), './mobilenet_ACC_'+str(float(100 * correct.item() / total))+'.pkl')
			param_bits=8
			bn_bits=32
			fwd_bits=8
			overflow_rate=0.0
			n_sample=10
			state_dict = mobilenet.state_dict()
			state_dict_quant = OrderedDict()
			sf_dict = OrderedDict()
			for k, v in state_dict.items():
				bits = param_bits
				sf = bits - 1. -compute_integral_part(v, overflow_rate=overflow_rate)
				quantize  = linear_quantize(v, sf, bits=bits)
				state_dict_quant[k] = quantize
			np.savez_compressed('./mobilenet_ACC_'+str(float(100 * correct.item() / total)),state_dict_quant)	
		lossArray.append(loss.item())
		epochArray.append(epoch)
		

		

def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]   #max
    if isinstance(v, Variable):
        v = v.data.cpu().numpy()
    sf = math.ceil(math.log2(v+1e-12))
    return sf		
def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input.float() / delta + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value    

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