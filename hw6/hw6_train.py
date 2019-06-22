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
jieba.set_dictionary(argv[4])

num_epochs = 4
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
		self.save_name = argv[4]
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
		#words=row.split("",len(row))
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
			embed = word2vec.Word2Vec(self.data+testData, size=self.embed_dim, window=5, iter=20, workers=8)
			embed.save(self.save_name)
		# Create word2index dictinonary
		# Create index2word list
		# Create word vector list
		for i, word in enumerate(embed.wv.vocab):
			print('=== get words #{}'.format(i+1), end='\r')
			#e.g. self.word2index['魯'] = 1 
			#e.g. self.index2word[1] = '魯'
			#e.g. self.vectors[1] = '魯' vector
			self.word2index[word] = len(self.word2index)
			self.index2word.append(word)
			self.vectors.append(embed[word])
		#self.vectors = torch.tensor(self.vectors)
		# Add special tokens
		self.add_embedding(self.pad)
		self.add_embedding(self.unk)
		print("=== total words: {}".format(len(self.vectors)))
		return self.vectors

	def add_embedding(self, word):
		# Add random uniform vector
		vector = torch.empty(1, self.embed_dim)
		torch.nn.init.uniform_(vector)
		self.word2index[word] = len(self.word2index)
		self.index2word.append(word)
		self.vectors = np.concatenate([self.vectors, vector], 0)

	def get_indices(self,test=False):
		# Transform each words to indices
		# e.g. if 機器=0,學習=1,好=2,玩=3 
		# [機器,學習,好,好,玩] => [0, 1, 2, 2,3]
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
				# if word in word2index append word index into sentence_indices
				# if word not in word2index append unk index into sentence_indices
				# TODO
			# pad all sentence to fixed length
			#sentence_indices = self.pad_to_len(sentence_indices, self.seq_len, self.word2index[self.pad])
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
							nn.Linear(num_hiddens * 6, 64),
							nn.ReLU(),
							nn.Linear(64, labels),
							nn.ReLU(),
							)
		else:
			self.decoder = nn.Sequential(
							nn.Linear(num_hiddens * 10, 50),
							nn.ReLU(),
							nn.Linear(50, labels),
							nn.ReLU(),
							)

	def forward(self, inputs,sorted_length):
		embeddings = self.embedding(inputs) #64,150,300
		embeddings=embeddings.float()
		embeddings_pad=torch.nn.utils.rnn.pack_padded_sequence(embeddings,sorted_length,batch_first=True)
		states, hidden = self.encoder(embeddings_pad)
		states, unpacked_len=torch.nn.utils.rnn.pad_packed_sequence(states,batch_first=True,padding_value=0)
		hidden=hidden[-1].permute(1,2,0).contiguous().view(embeddings.size(0),-1)
		
		#print(states.shape)#150 32 200
		#states.permute(1,2,0) 32 200 150
		
		avg_pool = F.adaptive_avg_pool1d(states.permute(0,2,1),1).view(embeddings.size(0),-1) #32 200 150 
		max_pool = F.adaptive_max_pool1d(states.permute(0,2,1),1).view(embeddings.size(0),-1)
		
		out = torch.cat([states[:,-1,:],max_pool,avg_pool],dim=1)
		#encoding = torch.cat([states[0],states[-1]], dim=1)          #concate  maxpooling  或是全部變ㄧ個?
		outputs = self.decoder(out)
		
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
	for i in range(15):

		print("Data processing")

		preprocess=Preprocess(argv[1],argv[2])
		testpreprocess=Preprocess(argv[3],None)
		embeding= preprocess.get_embedding(testpreprocess.data,load=False)
		train, label = preprocess.get_indices()
		embedingTest=testpreprocess.get_embedding(testData=None,load=True)
		test=testpreprocess.get_indices(test=True)
		##print(label)
		##print(len(label))
		#embeding= preprocess.get_embedding(None,load=True)


		X_train = pad_features(train)	
		X_test = pad_features(test)
		
		X_train=X_train[:119017]
		ans=label[:119017]
		temp=zip(X_train,ans)
		temp=list(temp)
		random.shuffle(temp)
		X_train, ans = zip(*temp)

		#train_data=torch.tensor(X_train[12000:])
		train_data=X_train[12000:]
		train_label=torch.tensor(ans[12000:])
		
		val_data=X_train[:12000]
		val_label=torch.tensor(ans[:12000])
		test_data=X_test

		embedding_weights=torch.from_numpy(embeding).cuda()

		rnn = RNN(vocab_size=len(embedding_weights), embed_size=embed_size,
					   num_hiddens=num_hiddens, num_layers=num_layers,
					   bidirectional=bidirectional, weight=embedding_weights,
					   labels=labels)
		if torch.cuda.is_available():
			rnn = rnn.cuda()



		dataset_train=MyDataset(data=train_data,target=train_label)
		dataset_val=MyDataset(data=val_data,target=val_label)
		dataset_test=TestDataset(data=test_data)

		train_loader=DataLoader(dataset=dataset_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=8,collate_fn=collate_fn)
		val_loader=DataLoader(dataset=dataset_val,batch_size=BATCH_SIZE,shuffle=False,num_workers=8,collate_fn=collate_fn)
		test_loader=DataLoader(dataset=dataset_test,batch_size=BATCH_SIZE,shuffle=False,num_workers=8,collate_fn=test_collate_fn)
		

		loss_function = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
		print('Start trainging...')
		for epoch in range(num_epochs):
			rnn.train()
			train_loss, test_losses = 0, 0
			train_acc, val_acc = 0, 0
			totalTrain, totalVal = 0, 0
			train_correct=0.0
			m,n=0,0
			for feature, label, sorted_length in train_loader:
				m=m+1
				totalTrain += label.size(0)
				rnn.zero_grad()

				feature = Variable(feature.cuda())
				label = Variable(label.cuda())
				score = rnn(feature,sorted_length)
				loss = loss_function(score, label)
				loss.backward()
				optimizer.step()
				predicted = torch.argmax(score.cpu().data,dim=1)
				train_correct += (predicted==label.cpu().data).sum()
				train_loss += loss
				if m%10==0:
						print("===Batch:"+str(m)+" Loss:"+str(float(train_loss.data/m))+" ===",end="\r")
			train_acc=float(train_correct.item()/totalTrain)  
			rnn.eval()
			val_correct=0.0    
			ans=[]					
			with torch.no_grad():
				for val_feature, val_label,sorted_length in val_loader:
					n=n+1
					totalVal += val_label.size(0)
					val_feature = val_feature.cuda()
					val_label = val_label.cuda()
					val_score = rnn(val_feature,sorted_length)
					val_loss = loss_function(val_score, val_label)
					predicted =torch.argmax(val_score.cpu().data,dim=1)
					val_correct += (predicted==val_label.cpu().data).sum()
					test_losses += val_loss
				val_acc=float(val_correct.item()/totalVal) 
				for output_feature,sorted_length,originIndex in test_loader:
						output_feature = output_feature.cuda()
						output_score = rnn(output_feature,sorted_length)
						predicted=torch.argmax(output_score.cpu().data,dim=1)
						for i in range(predicted.size(0)):
							origin=originIndex.index(i)
							ans.append(int(predicted[origin]))
			outputfile('test'+str(val_acc)+'.csv', ans)
			print('epoch: %d, train loss: %.4f, train acc: %.2f, test loss: %.4f, test acc: %.2f' %
			  (epoch, train_loss.data / m, train_acc, test_losses.data / n, val_acc))

			torch.save(rnn.state_dict(), 'rnn_Acc'+str(val_acc)+'.pkl')
			print("Done")
	
		


if __name__ == '__main__':
	main()















