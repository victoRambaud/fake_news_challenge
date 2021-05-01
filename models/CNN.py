import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable 

def max(out):
    m = out.max(axis=-1).values
    return torch.reshape(m,(m.size()[0], m.size()[1]))

class FakeNewsCNN(nn.Module):

	def __init__(self,embedding_dim=300,hidden_size=256,n_classes=4):
		super(FakeNewsCNN,self).__init__()

		# 1d Convolutional part
		self.conv1 = nn.Conv1d(in_channels=embedding_dim,out_channels=hidden_size,kernel_size=3,padding=3)
		self.bn_conv1 = nn.BatchNorm1d(hidden_size)

		self.conv2 = nn.Conv1d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=3,padding=3)
		self.bn_conv2 = nn.BatchNorm1d(hidden_size)

		self.conv3 = nn.Conv1d(in_channels=hidden_size,out_channels=hidden_size*2,kernel_size=3,padding=3)
		self.bn_conv3 = nn.BatchNorm1d(hidden_size*2)

		self.conv4 = nn.Conv1d(in_channels=hidden_size*2,out_channels=hidden_size*2,kernel_size=3,padding=1)
		self.bn_conv4 = nn.BatchNorm1d(hidden_size*2)

		self.conv5 = nn.Conv1d(in_channels=hidden_size*2,out_channels=hidden_size*3,kernel_size=3,padding=1)
		self.bn_conv5 = nn.BatchNorm1d(hidden_size*3)

		# Fully Connected part
		self.dense1 = nn.Linear(in_features=hidden_size*3*2,out_features=hidden_size*4)
		self.bn_dense1 = nn.BatchNorm1d(hidden_size*4)

		self.dense2 = nn.Linear(in_features=hidden_size*4,out_features=hidden_size*4)
		self.bn_dense2 = nn.BatchNorm1d(hidden_size*4)

		self.dense3 = nn.Linear(in_features=hidden_size*4,out_features=hidden_size*4)
		self.bn_dense3 = nn.BatchNorm1d(hidden_size*4)

		self.dense4 = nn.Linear(in_features=hidden_size*4,out_features=n_classes)

		self.dropout = nn.Dropout(p=0.3)
		#self.avg_pool = nn.AdaptiveAvgPool1d()
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self,input):
		# input must be a 2 dim vec with body and head
		input_head = input[0]
		input_body = input[1]

		body = F.relu(self.bn_conv1(self.conv1(input_body)))
		body = F.relu(self.bn_conv2(self.dropout(self.conv2(body))))
		body = F.relu(self.bn_conv3(self.dropout(self.conv3(body))))
		body = F.relu(self.bn_conv4(self.dropout(self.conv4(body))))
		body = F.relu(self.bn_conv5(self.dropout(self.conv5(body))))

		head = F.relu(self.bn_conv1(self.conv1(input_head)))
		head = F.relu(self.bn_conv2(self.dropout(self.conv2(head))))
		head = F.relu(self.bn_conv3(self.dropout(self.conv3(head))))
		head = F.relu(self.bn_conv4(self.dropout(self.conv4(head))))
		head = F.relu(self.bn_conv5(self.dropout(self.conv5(head))))

		head_vec = max(head)
		body_vec = max(body)
		input_features = torch.cat((head_vec, body_vec),axis=1)

		output = F.relu(self.bn_dense1(self.dense1(input_features)))
		output = F.relu(self.bn_dense2(self.dense2(output)))
		output = F.relu(self.bn_dense3(self.dense3(output)))
		output = F.relu(self.dense4(output))

		return self.logsoftmax(output)

#decentralization leads to 3 dimensions -> same as input (actually)

class FakeNewsBiLSTM(nn.Module):

	def __init__(self,embedding_dim,hidden_size,num_layers,n_classes,device):
		super(FakeNewsBiLSTM,self).__init__()
		self.device = device
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.num_directions = 2

		self.head_BiLSTM = nn.LSTM(input_size=embedding_dim,
			hidden_size=hidden_size,
			num_layers=num_layers,
			bidirectional=True,
			batch_first=True)

		self.body_BiLSTM = nn.LSTM(input_size=embedding_dim,
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
		head,h = self.head_BiLSTM(input_head,hidden)

		hidden_forward = Variable(torch.zeros(self.num_layers*self.num_directions,batch_size,self.hidden_size, device=self.device))
		hidden_backward = Variable(torch.zeros(self.num_layers*self.num_directions,batch_size,self.hidden_size, device=self.device))
		hidden = (hidden_forward,hidden_backward,)
		body,h = self.body_BiLSTM(input_body,hidden)

		input_features = torch.cat((head, body),axis=1)

		# taking last embedding of hidden layer
		output = self.fc(input_features[:,-1,:])
		output = self.fc2(output)

		return self.logsoftmax(output)