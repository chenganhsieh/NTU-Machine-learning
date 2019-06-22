import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
import csv
import numpy as np
import seaborn as sn
import os
import torchvision.transforms as transforms
import lime
from lime import lime_image
from sys import argv
import skimage.segmentation 

BATCH_SIZE= 256
torch.manual_seed(1)
np.random.seed(1)
nb_classes=7;

def Mytrainsform():
	transform1 = transforms.Compose([
	transforms.CenterCrop(44),	
	transforms.ToTensor(),
	#transforms.TenCrop(44),
	#transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
	])
	return transform1   

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
	feature = feature.reshape(feature.shape[0],48,48,1)

	x_train = feature[2871:]
	x_val = feature[0:2871]
	y_train = label[2871:]
	y_val = label[0:2871]


	return x_train, y_train, x_val, y_val


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

	def forward(self, x):
		out = self.conv1(x)
		out = out.view(out.size()[0], -1)
		return self.fc(out)
def gaussian_weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1 and classname.find('Conv') == 0:
		m.weight.data.normal_(0.0, 0.02)


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
		#print(x.size())
		return x,y

	def transform(self, image):
		image=transforms.RandomCrop(44)(image)
		image=transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0)(image)
		image=transforms.RandomRotation(20)(image)
		image = transforms.functional.to_tensor(image)
		return image
	def __len__(self):
		return len(self.data)
def main():
	imageData=np.empty(1)

	cnn=CNN()
	if torch.cuda.is_available():
		cnn = cnn.cuda()
	print(cnn)

	x_train, y_train, x_test, y_test = readData(argv[1])

	cnn.load_state_dict(torch.load('cnn_best67.pkl')) 
	

	dataset_test=MyDataset(data=x_test,target=y_test,trainsform=Mytrainsform())
	x_test=DataLoader(dataset=dataset_test,batch_size=1,shuffle=False,num_workers=8)   
	i=0
	s=0
	check=np.zeros(7)
	image=[]
	index=[]
	for data,target in x_test:
		#if check[target.item()]==1:
		#	if check.all()==1:
		#		break
		#	else:
		#		continue
		if 1==1:
			#17:index2 16:index3 388:index1 11:index6 29:index5  6:index4  22:index0
			if (i==17 or i==16 or i==388 or i==11 or i==29 or i==6 or i==22):
				fig = plt.figure(figsize=(14, 4))
				x=data.numpy()
				image.append(x.squeeze(0))
				index.append(target.item())
				x=x.squeeze()
		  #plt.title("index:"+str(target.item()))
		  #plt.imshow(x, plt.cm.gray)
				#plt.imsave('./origin_'+str(target.item()), x, cmap=plt.cm.gray)
		  #plt.show()
				ax1=fig.add_subplot(1, 3,1,xticks=[], yticks=[])
				ax2=fig.add_subplot(1, 3,2,xticks=[], yticks=[])
				ax3=fig.add_subplot(1, 3,3,xticks=[], yticks=[])

				ax1.set_title("index:"+str(target.item()))
				heat=sn.heatmap(x,cmap=plt.cm.gray,ax=ax1)
				#plt.title("index:"+str(target.item()))
				#plt.show()
				
				#fig=heat.get_figure()
				#fig.savefig('./image/'+str(i)+'_origin_'+str(target.item())+'.jpg')
				#plt.clf()

				data= get_cuda(data)
				target = get_cuda(target)
				data.requires_grad_()
				outputs = cnn(data)
				loss_func = torch.nn.CrossEntropyLoss()
				loss = loss_func(outputs,target)
				loss.backward()
				grads =data.grad
				grads = grads.abs()
				mx, index_mx = torch.max(grads, 1)
				saliency=mx.data
				saliency=saliency.detach().cpu().numpy()
				saliency=saliency.squeeze()
				#print(saliency)
				maxPoint= np.amax(saliency)
				saliency=saliency*(1/maxPoint)
				heat=sn.heatmap(saliency,cmap=plt.cm.jet,ax=ax2)
				#plt.show()
				
				
				#ig=heat.get_figure()
				#fig.savefig('./image/'+str(i)+'_saliency_'+str(target.item())+'.jpg')
				#plt.clf()

				heat=sn.heatmap(x,cmap=plt.cm.gray,mask=saliency<0.1,ax=ax3)
				
				fig.savefig(argv[2]+'fig1_'+str(target.item())+'.jpg')
				#plt.show()
				#fig=heat.get_figure()
				#fig.savefig('./image/'+str(i)+'_mask_'+str(target.item())+'.jpg')
				plt.clf()

		i=i+1
		if(i>388):
			imageData=np.array(image)
			#np.save('image.npy',image)
			#np.save('index.npy',index)
			break
	print('Saliency Done!')
	#------------------------------------------filter Graph
	cnn=CNN()
	#if torch.cuda.is_available():
	#	cnn = cnn.cuda()
	cnn.load_state_dict(torch.load('cnn_best67.pkl')) 
	x_train = imageData
	x_test=x_train[0]
	x_test=x_test.reshape(1,1,44,44)
	x_test=torch.tensor(x_test)
	#if torch.cuda.is_available():
	#	x_test=x_test.cuda()
	  #  print(list(cnn.children())[0])
  
	#pretrained_model = models.vgg16(pretrained=True).features
	aa=0
	for i in list(cnn.children())[0]:
				x_test=i(x_test)
				if aa>4:
					break
				aa=aa+1
	x=np.round(np.abs(x_test[0].data.numpy())*255)
	

	sample = np.random.uniform(10, 180, (44, 44, 1))
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	for channel in range(1):
		sample[:, :, channel]/=255
		sample[:, :, channel]-=mean[channel]
		sample[:, :, channel]/=std[channel]

	sample = sample.reshape((1,)+sample.shape)
	sample = sample.transpose(0, 3, 1, 2)
	sample=torch.Tensor(sample)
	loss_func = nn.CrossEntropyLoss()	
	result=[]
	for ss in range(32):
		optimizer = Adam([sample], lr=0.1, weight_decay=1e-6)
		for i in range(5):
			optimizer.zero_grad()
			temp=sample
			temp.requires_grad_()
			aa=0
			#temp=j(temp)
			for i in list(cnn.children())[0]:
				temp=i(temp)
				if aa>4:
					break
				aa=aa+1
			loss = -torch.mean(temp[0,ss])
			print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
			loss.backward()
			optimizer.step()
		reverse_mean = [-0.485, -0.456, -0.406]
		reverse_std = [1/0.229, 1/0.224, 1/0.225]
		sampleGraph=sample.data.numpy()[0]
		sampleGraph[0]/=reverse_std[0]
		sampleGraph[0]-= reverse_mean[0]
		sampleGraph[sampleGraph > 1] = 1
		sampleGraph[sampleGraph < 0] = 0
		sampleGraph =  np.uint8(np.round(sampleGraph * 255))
		
		result.append(sampleGraph)
	result=np.array(result)
	
	result=result.reshape(1,32,44,44)
	filterImage=result[0]
	
	fig = plt.figure(figsize=(14, 8))
	filterRes = plt.figure(figsize=(14, 8))
	for num in range(32):
		ax = fig.add_subplot(4, 8, num+1)
		ax.imshow(x[num], cmap=plt.cm.gray)
		plt.xticks(np.array([]))
		plt.yticks(np.array([]))
		plt.xlabel('filter {}'.format(num))
		plt.tight_layout()
		plt.clf()
	fig.savefig(argv[2]+'fig2_2.jpg')

	for num in range(32):
		ax1=filterRes.add_subplot(4, 8, num+1)
		ax1.imshow(filterImage[num], cmap=plt.cm.gray)
		plt.xticks(np.array([]))
		plt.yticks(np.array([]))
		plt.xlabel('filter {}'.format(num))
		plt.tight_layout()
	filterRes.savefig(argv[2]+'fig2_1.jpg')
	print("Filter Done!")
	#------------------------------------------------------------

	x_train = imageData
	x_train = np.transpose(x_train, (0, 2, 3, 1))
	x_label = index

	# Lime needs RGB images
	# TODO:
	# x_train_rgb = ?
	x_train_rgb=np.concatenate((x_train,x_train,x_train),axis=3)
	

	model = CNN()
	if torch.cuda.is_available():
		model = model.cuda()
	model.load_state_dict(torch.load('cnn_best67.pkl'))
	#model.load_state_dict(torch.load('cnn_best67.pkl',map_location=device))
	model.eval()
	# two functions that lime image explainer requires
	def predict(input):
	# Input: image tensor
	# Returns a predict function which returns the probabilities of labels ((7,) numpy array)
	# ex: return model(data).numpy()
	# TODO:
	# return ?

		input1=input[:,:,:,0]
		input1=input1.reshape(10,44,44,1)
		input1 = np.transpose(input1, (0, 3, 1, 2))
		input1=torch.tensor(input1)
		if torch.cuda.is_available():
		   input1=input1.cuda()
		
		outputs=model(input1)
	#_, predicted = torch.max(outputs.data, 1)
	#print(predicted.shape)
	
		return outputs.detach().cpu().numpy()


	def segmentation(input):   
		#input= np.concatenate(input[0].reshape(44,44,1),input[1].reshape(44,44,1),input[2].reshape(44,44,1),axis=2)
		input=np.round(input* 255)
		output=np.uint8(input)
		a=skimage.segmentation.slic(output, n_segments=20, max_iter=20, multichannel=True)
	
	#print(input.shape)
	#output=input[0]
	#output=output.reshape(44,44)
	#print(output.shape)
	# Input: image numpy array
	# Returns a segmentation function which returns the segmentation labels array ((48,48) numpy array)
	# ex: return skimage.segmentation.slic()
	# TODO:
	# return ?
   # a=skimage.segmentation.slic(output, n_segments=43, max_iter=30, multichannel=False)
   # a=skimage.segmentation.quickshift(output,kernel_size=4,max_dist=200, ratio=0.2)
   # a=a.reshape(1,44,44)
   # a=a.transpose(2,1,0)
   
	#a=np.concatenate((a,a,a),axis=0)
	#print(a.shape)
		return a

  
	for i in range(7):
		idx=i
	
		# Initiate explainer instance
		explainer = lime_image.LimeImageExplainer()

		# Get the explaination of an image
		explaination = explainer.explain_instance(
							image=x_train_rgb[idx], 
							classifier_fn=predict,
							segmentation_fn=segmentation
						)

		# Get processed image
		image, mask = explaination.get_image_and_mask(
								label=x_label[idx],
								positive_only=False,
								hide_rest=False,
								num_features=7,
								min_weight=0.0
							)

		# save the image
		plt.imsave(argv[2]+'fig3_'+str(x_label[idx])+'.jpg', image)
	print("Lime Graph Done")	




	
if __name__ == '__main__':
	main()


'''

data= get_cuda(data)
		target = get_cuda(target)
		data.requires_grad_()
		outputs = cnn(data)
		print(outputs.size())
		print(outputs)
		loss_func = torch.nn.CrossEntropyLoss()
		loss = loss_func(outputs,target)
		loss.backward()
		grads =data.grad
		grads = grads.abs()
		mx, index_mx = torch.max(grads, 1)
		saliency = mx.data
		print(saliency)
		saliency=saliency.numpy()
		heat=sn.heatmap(saliency, annot=True)
		fig=heat.get_figure()
		fig.savefig('saliency.jpg')
		print('Done!')
		input()
		scores = outputs.gather(1, target.view(-1, 1)).squeeze() 
		print(scores)
		input()
'''