import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import csv
import numpy as np
import random
from sys import argv
import jieba
from gensim.models import word2vec
import re
import pandas as pd
from multiprocessing import Pool

torch.manual_seed(1)
np.random.seed(1337)  
jieba.set_dictionary(argv[2])

num_epochs = 20
embed_size = 300	
num_hiddens = 300
num_layers = 3
bidirectional = True
BATCH_SIZE = 32
labels = 2
lr = 0.001

class Preprocess():
	def __init__(self, data_dir, label_dir):
		# Load jieba library
		#jieba.load_userdict(args.jieba_lib)
		self.embed_dim = embed_size
		#self.seq_len = args.seq_len
		#self.wndw_size = args.wndw
		#self.word_cnt = args.cnt
		self.save_name = 'word2vec2.model'
		self.index2word = []
		self.word2index = {}
		self.vectors = []
		self.unk = "<UNK>"
		self.pad = "<PAD>"
		# Load corpus
		if data_dir!=None:
			# Read data
			dm = pd.read_csv(data_dir)
			data = dm['comment']
			# Tokenize with multiprocessing
			# List in list out with same order
			# Multiple workers
			P = Pool(processes=4) 
			data = P.map(self.tokenize, data)
			P.close()
			P.join()
			self.data = data
			
			
		if label_dir!=None:
			# Read Label
			dm = pd.read_csv(label_dir)
			self.label = [int(i) for i in dm['label']]

	def tokenize(self, sentence):
		row=sentence.strip('\n')
		#RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
		#row = RE_EMOJI.sub(r'', row)
		row = re.sub("8+[(+*)]+9", "廢物", row)
		#row = re.sub("B+\d+", "", row)
		#row = re.sub("b+\d+", "", row)
		row = re.sub("ㄅㄔ", "白癡", row)
		row = re.sub("ㄐㄅ", "靠夭", row)
		row = re.sub("ㄌㄙ", "垃圾", row)
		row = re.sub("ㄍㄢˋㄋㄧˇㄋㄧㄤˊ", "幹你娘", row)
		row = re.sub("森77", "生氣氣", row)
		row = re.sub("80", "霸凌", row)
		row = re.sub("87", "白癡", row)
		row = re.sub("Hen", "很", row)
		row = re.sub("沒關C", "沒關係", row)
		row = re.sub("ㄋㄊㄇ", "笨蛋", row)
		row = re.sub("ㄉ", "的", row)
		row = re.sub("ㄏ", "呵", row)
		row = re.sub("ㄅ", "吧", row)
		row = re.sub("ㄚ", "啊", row)
		row=re.sub(" ", "", row)
		words = jieba.cut(row, cut_all=False)
		tokens=[]
		for word in words:
			tokens.append(word)
		return tokens

	def get_embedding(self, testData,load=False):
		print("=== Get embedding")
		# Get Word2vec word embedding
		if load:
			embed = word2vec.Word2Vec.load(self.save_name)
		else:
			embed = word2vec.Word2Vec(self.data+testData, size=self.embed_dim, window=5, iter=16, workers=8)
			embed.save(self.save_name)
		for i, word in enumerate(embed.wv.vocab):
			print('=== get words #{}'.format(i+1), end='\r')
			self.word2index[word] = len(self.word2index)
			self.index2word.append(word)
			self.vectors.append(embed[word])
		self.add_embedding(self.pad)
		self.add_embedding(self.unk)
		print("=== total words: {}".format(len(self.vectors)))
		return self.vectors

	def add_embedding(self, word):
		vector = torch.empty(1, self.embed_dim)
		torch.nn.init.uniform_(vector)
		self.word2index[word] = len(self.word2index)
		self.index2word.append(word)
		self.vectors = np.concatenate([self.vectors, vector], 0)

	def get_indices(self,test=False):
		all_indices = []
		# Use tokenized data
		for i, sentence in enumerate(self.data):
			print('=== sentence count #{}'.format(i+1), end='\r')
			sentence_indices = []
			for word in sentence:
				if word in self.index2word:
					sentence_indices.append(self.word2index[word])
				else:
					 sentence_indices.append(self.word2index["<UNK>"])
			all_indices.append(sentence_indices)
		if test:
			return all_indices         
		else:
			return all_indices, self.label      

def outputfile(n, ans):
	f = open(n,"w+")
	w = csv.writer(f)
	title = ['id','label']
	w.writerow(title) 
	for i in range(len(ans)):
		content = [i,ans[i]]
		w.writerow(content) 
	f.close()

def pad_features(features, maxlen=200, PAD=0):   #pad_feature: 可以在每個batch進來的時候找最長的去fit他長度就好
	padded_features = []
	for feature in features:
		if len(feature) >= maxlen:
			padded_feature = feature[:maxlen]
		else:
			padded_feature=feature
		padded_features.append(padded_feature)
	return padded_features
class RNN(nn.Module):
	def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
				 bidirectional, weight, labels, **kwargs):
		super(RNN, self).__init__(**kwargs)
		self.num_hiddens = num_hiddens
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.embedding = nn.Embedding.from_pretrained(weight)
		self.embedding.weight.requires_grad = False
		self.dropout=nn.Dropout(0.1)
		self.attn = nn.Linear(num_hiddens * 2, 1)
		self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
							   num_layers=num_layers, bidirectional=self.bidirectional,
							   dropout=0.5)
		if self.bidirectional:
			self.decoder = nn.Sequential(
							#nn.Linear(num_hiddens * 10, 50),
							nn.Linear(num_hiddens * 10, 50),
							nn.ReLU(),
							nn.Linear(50, labels),
							nn.ReLU(),
							)
		else:
			self.decoder = nn.Sequential(
							nn.Linear(num_hiddens * 10, 50),
							nn.ReLU(),
							nn.Linear(50, labels),
							#nn.ReLU(),
							)

	def forward(self, inputs,sorted_length):
		embeddings = self.embedding(inputs) #64,150,300
		embeddings=embeddings.float()
		embeddings_pad=torch.nn.utils.rnn.pack_padded_sequence(embeddings,sorted_length,batch_first=True)
		states, hidden = self.encoder(embeddings_pad)
		states, unpacked_len=torch.nn.utils.rnn.pad_packed_sequence(states,batch_first=True,padding_value=0)
		hidden=hidden[-1].permute(1,2,0).contiguous().view(embeddings.size(0),-1)
		
		att = self.attn(states.permute(1,0,2)).squeeze(-1)
		att=F.softmax(att, dim=-1)
		r_att = torch.sum(att.unsqueeze(-1) * states.permute(1,0,2), dim=1) 
		#print(states.shape)#150 32 200
		#states.permute(1,2,0) 32 200 150
		
		avg_pool = F.adaptive_avg_pool1d(states.permute(0,2,1),1).view(embeddings.size(0),-1) #32 200 150 
		max_pool = F.adaptive_max_pool1d(states.permute(0,2,1),1).view(embeddings.size(0),-1)
		
		out = torch.cat([hidden,max_pool,avg_pool],dim=1)
		#encoding = torch.cat([states[0],states[-1]], dim=1)          #concate  maxpooling  或是全部變ㄧ個?
		outputs = self.decoder(out)
		#outputs=F.softmax(outputs, dim=-1)
		return outputs

class MyDataset(Dataset):
	def __init__(self,data,target=None):
		super(MyDataset,self).__init__()
		self.data=data
		self.target=target
	def __getitem__(self,index):
		x=self.data[index]
		y=self.target[index]
		return x,y
	def __len__(self):
		return len(self.data)

def collate_fn(batch):
	batch.sort(key=lambda x: len(x[0]), reverse=True)
	data, labels = zip(*batch)
	sorted_length=[len(size) for size in data]
	dataTensor=[torch.tensor(detail) for detail in data]
	dataTensor=torch.nn.utils.rnn.pad_sequence(dataTensor,batch_first=True,padding_value=0)
	label=torch.tensor(labels)
	return dataTensor,label,sorted_length
def test_collate_fn(batch):
	batch = [ (batch[i],i) for i in range(len(batch)) ]
	batch.sort(key=lambda x: len(x[0]), reverse=True)
	batch,originIndex = zip(*batch)
	sorted_length=[len(size) for size in batch]
	dataTensor=[torch.tensor(detail) for detail in batch]
	dataTensor=torch.nn.utils.rnn.pad_sequence(dataTensor,batch_first=True,padding_value=0)
	label=torch.tensor(labels)
	return dataTensor,sorted_length,originIndex
class TestDataset(Dataset):
	def __init__(self,data):
		super(TestDataset,self).__init__()
		self.data=data
	def __getitem__(self,index):
		x=self.data[index]
		return x
	def __len__(self):
		return len(self.data)		

def main():
	print("Data processing")
	preprocess=Preprocess(None,None)
	testpreprocess=Preprocess(argv[1],None)
	#embeding= preprocess.get_embedding(testpreprocess.data,load=False)
	#train, label = preprocess.get_indices()
	embedingTest=testpreprocess.get_embedding(testData=None,load=True)
	test=testpreprocess.get_indices(test=True)

	#print(label)
	#print(len(label))
	embeding= preprocess.get_embedding(None,load=True)

	#f = open('train.csv',"w+")
	#w = csv.writer(f)
	#w.writerows(train)
	#f.close()
	
	#with open('label.csv',"w+") as f:
	#	w = csv.writer(f)
	#	w.writerows([[onelabel] for onelabel in label]) 
	#f.close()
	
	#f = open('test_oneSentence.csv',"w+")
	#w = csv.writer(f)
	#w.writerows(test) 
	#f.close()

	
	X_train=[]
	ans=[]
	test_data=[]

	
	#ans=readLabel('train_y.csv')
	#ans=np.array(ans)
	
	
	#with open('test_oneSentence.csv') as f:
	#	for row in csv.reader(f):
	#		test_data.append(list(map(int, row)))
	#f.close()

	
	X_test = pad_features(test)
	
	
	
	test_data=X_test

	embedding_weights=torch.from_numpy(embeding).cuda()

	rnn = RNN(vocab_size=len(embedding_weights), embed_size=embed_size,
				   num_hiddens=num_hiddens, num_layers=num_layers,
				   bidirectional=bidirectional, weight=embedding_weights,
				   labels=labels)
	if torch.cuda.is_available():
		rnn = rnn.cuda()


	rnn.load_state_dict(torch.load('rnn_Acc0.7626666666666667.pkl'))
	
	dataset_test=TestDataset(data=test_data)

	test_loader=DataLoader(dataset=dataset_test,batch_size=BATCH_SIZE,shuffle=False,num_workers=8,collate_fn=test_collate_fn)
	

	ans=[]
	rnn.eval()
	#-----------------------testData
	print("testing...")
	with torch.no_grad():
		for output_feature,sorted_length,originIndex in test_loader:
			output_feature = output_feature.cuda()
			output_score = rnn(output_feature,sorted_length)
			predicted=torch.argmax(output_score.cpu().data,dim=1)

			for i in range(predicted.size(0)):
				origin=originIndex.index(i)
				ans.append(int(predicted[origin]))
	outputfile(argv[3], ans)
	print("Done")	


if __name__ == '__main__':
	main()















