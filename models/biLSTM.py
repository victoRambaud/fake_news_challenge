import torch

import torch.nn as nn

from torch.autograd import Variable 

#decentralization leads to 3 dimensions -> same as input (actually)

class FakeNewsBiLSTM(nn.Module):

	def __init__(self,embedding_dim,hidden_size,num_layers,n_classes,device):
		super(FakeNewsBiLSTM,self).__init__()
		self.device = device
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.num_directions = 2

		#self.head_BiLSTM = nn.LSTM(input_size=embedding_dim,
		#	hidden_size=hidden_size,
		#	num_layers=num_layers,
		#	bidirectional=True,
		#	batch_first=True)

		#self.body_BiLSTM = nn.LSTM(input_size=embedding_dim,
		#	hidden_size=hidden_size,
		#	num_layers=num_layers,
		#	bidirectional=True,
		#	batch_first=True)

		self.BiLSTM = nn.LSTM(input_size=embedding_dim,
			hidden_size=hidden_size,
			num_layers=num_layers,
			bidirectional=True,
			batch_first=True)

		self.fc = nn.Linear(hidden_size*self.num_directions,256)
		self.fc2 = nn.Linear(256,n_classes)
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self,input):
		# input must be a 2 dim vec with body and head
		#the lstm is batch first so input(batch_size,seq_len,n_features) and output(batch_size,seq_len,n_features)
		input_head = torch.transpose(input[0],1,2)
		input_body = torch.transpose(input[1],1,2)

		batch_size = input_head.shape[0] # same for both 

		hidden_forward = Variable(torch.zeros(self.num_layers*self.num_directions,batch_size,self.hidden_size, device=self.device))
		hidden_backward = Variable(torch.zeros(self.num_layers*self.num_directions,batch_size,self.hidden_size, device=self.device))
		hidden = (hidden_forward,hidden_backward,)
		head,h = self.BiLSTM(input_head,hidden)

		hidden_forward = Variable(torch.zeros(self.num_layers*self.num_directions,batch_size,self.hidden_size, device=self.device))
		hidden_backward = Variable(torch.zeros(self.num_layers*self.num_directions,batch_size,self.hidden_size, device=self.device))
		hidden = (hidden_forward,hidden_backward,)
		body,h = self.BiLSTM(input_body,hidden)

		input_features = torch.cat((head, body),axis=1)

		# taking last embedding of hidden layer
		output = self.fc(input_features[:,-1,:])
		#output = self.fc2(output)

		return self.logsoftmax(output)