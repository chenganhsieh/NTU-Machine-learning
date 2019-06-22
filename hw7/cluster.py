import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.utils import save_image
import csv
import numpy as np
import random
from sys import argv
from skimage import io
import pandas as pd
from multiprocessing import Pool
from sklearn import cluster
from sklearn.decomposition import PCA
import os

torch.manual_seed(1)
np.random.seed(36)  
random.seed(36)


num_epochs = 30
BATCH_SIZE = 32
labels = 2
lr = 0.001 
PHOTO_SIZE=32
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		# Encoder
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(16)
		# Latent vectors mu and sigma
		self.fc1 = nn.Linear(8*8 * 16, 512)
		self.fc_bn1 = nn.BatchNorm1d(512)
		self.fc21 = nn.Linear(512, 512)
		self.fc22 = nn.Linear(512, 512)
		# Sampling vector
		self.fc3 = nn.Linear(512, 512)
		self.fc_bn3 = nn.BatchNorm1d(512)
		self.fc4 = nn.Linear(512, 8*8 * 16)
		self.fc_bn4 = nn.BatchNorm1d(8*8 * 16)
		# Decoder
		self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
		self.bn5 = nn.BatchNorm2d(64)
		self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn6 = nn.BatchNorm2d(32)
		self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
		self.bn7 = nn.BatchNorm2d(16)
		self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU()	
	def forward(self, x):
		conv1 = self.relu(self.bn1(self.conv1(x)))
		conv2 = self.relu(self.bn2(self.conv2(conv1)))
		conv3 = self.relu(self.bn3(self.conv3(conv2)))
		conv4 = self.relu(self.bn4(self.conv4(conv3))).view(x.size(0), 8*8* 16)
		fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
		mu = self.fc21(fc1)
		std = self.fc22(fc1)


		std = std.mul(0.5).exp_()                        
		eps = Variable(std.data.new(std.size()).normal_())
		z = eps.mul(std).add_(mu)                        #mul->Matrix Multiplication

		fc3 = self.relu(self.fc_bn3(self.fc3(z)))
		fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(x.size(0), 16, 8, 8)
		conv5 = self.relu(self.bn5(self.conv5(fc4)))
		conv6 = self.relu(self.bn6(self.conv6(conv5)))
		conv7 = self.relu(self.bn7(self.conv7(conv6)))
		out = self.conv8(conv7).view(-1, 3, PHOTO_SIZE, PHOTO_SIZE)
		return out,mu,std,z
def gaussian_weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1 and classname.find('Conv') == 0:
		m.weight.data.normal_(0.0, 0.02)
class MyDataset(Dataset):
	def __init__(self,data,target=None,trainsform=None):
		super(MyDataset,self).__init__()
		self.data=data
	def __getitem__(self,index):
		x=self.data[index]
		return x
def to_img(x):
	x = 0.5 * (x + 1)
	x = x.clamp(0, 1)
	x = x.view(3, PHOTO_SIZE, PHOTO_SIZE)
	return x
def loadImage(directory_name):
	array_of_img = []
	path_list=os.listdir(directory_name)
	path_list.sort()
	for filename in path_list:
		img=io.imread(directory_name+"/"+filename)
		#img=np.array(img)
		#img=img.reshape(3,PHOTO_SIZE,PHOTO_SIZE)
		array_of_img.append(img)
	return array_of_img
def get_vector(data):
	output_data=[]
	for image in data:
		pre_img = Variable(normalize(to_tensor(image)).cuda())	
		output_data.append(pre_img)
	return output_data
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    def forward(self, recon_x, x, mu, logvar):
        MSE = self.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2)-logvar.exp())
        return MSE + KLD
def readTest(data_dir):
	if data_dir!=None:
		dm = pd.read_csv(data_dir)
		data1 = dm['image1_name']
		data2=dm['image2_name']
		# Tokenize with multiprocessing
		# List in list out with same order
		# Multiple workers
		return data1,data2
def outputfile(n, ans):
	f = open(n,"w+")
	w = csv.writer(f)
	title = ['id','label']
	w.writerow(title) 
	for i in range(len(ans)):
		content = [i,ans[i]]
		w.writerow(content) 
	f.close()
def main():
	print("===Loading images...===")
	data=loadImage(argv[1])
	dataset=get_vector(data)


	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
	model = CNN().cuda()
	model.load_state_dict(torch.load('conv_autoencoder_512.pth'))
	optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)
	cos = nn.CosineSimilarity(dim=0, eps=1e-6)
	loss_mse = Loss()
	batch=0
	all_data=0
	similarity=0
	latent=[]

	print("===Testing...===")
	for data in dataloader:
		all_data+=data.size(0)
		batch+=1
		img= data
		'''
		for i in range(data.size(0)):
			if i<64:
			#im = Image.fromarray(data[i])
			#pic=to_img(data[i])
			#im.save("./test/"+str(i)+".jpg")
				pic = to_img(data[i].cpu().data)
				save_image(pic, "./test/"+str(i)+".jpg")
		print("Done")'''
		
		
		out, mu, std,latent_code= model(data) 
		for i in range(out.size(0)):
			latent.append(latent_code[i].cpu().data.numpy())
			temp1=out[i].view(-1)
			temp2=img[i].view(-1)
			#pic = to_img(out[i].cpu().data)
			#save_image(pic, "./test/"+str(i)+".jpg")
			cos_sim1=cos(temp1,temp2)
			similarity+=cos_sim1.cpu().data.numpy()
		#input()
		if batch%100==0:
			print("===Batch:"+str(batch)+" similarity:"+str(float(similarity/all_data))+" ===",end="\r")
	
	data1,data2=readTest(argv[2])
	latent=np.array(latent)
	X_pca = PCA(n_components=128,whiten=True,svd_solver='full').fit_transform(latent)
	clf = cluster.KMeans(n_clusters=2, random_state=28,max_iter=3000)
	k_mean=clf.fit(X_pca)
	#outputfile(argv[3], k_mean.labels_)
	#input()
	#input()
	#fig, ax = plt.subplots(1, 1, figsize=(8, 4))
	#fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
	#fig.subplots_adjust(top=0.85)
	#ax[0].scatter(X_pca[:, 0], X_pca[:, 1])
	#ax[0].set_title('Predicted Training Labels')
	#plt.show()
	#print(len(X_pca[:,0]))
	#print(len(X_pca[:,1]))
#
	#input()
	result=[]
	yes=0
	no=0
	for i in range(len(data1)):
		predict=k_mean.labels_[data1[i]-1]
		predict2=k_mean.labels_[data2[i]-1]
		#predict=k_mean.predict(X_pca[data1[i]-1].reshape(1,-1))
		#predict2=k_mean.predict(X_pca[data2[i]-1].reshape(1,-1))
		#similarity=cos(latent[data1[i]-1],latent[data2[i]-1])
		if predict==predict2:
			yes+=1
			result.append(1)
		else:
			no+=1
			result.append(0)
		print("===Yes:"+str(yes)+", No:"+str(no)+" ===",end="\r")

	outputfile(argv[3], result)

	print("Done")

	#result=[]
	#yes=0
	#no=0
	#for i in range(len(data1)):
	#	similarity=cos(latent[data1[i]-1],latent[data2[i]-1])
	#	if similarity>0.5:
	#		yes+=1
	#		result.append(1)
	#	else:
	#		no+=1
	#		result.append(0)
	#	print("===Yes:"+str(yes)+", No:"+str(no)+" ===",end="\r")

	#outputfile('ans_256.csv', result)

	#print("Done")

if __name__ == '__main__':
	main()
