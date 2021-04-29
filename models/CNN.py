import torch
import torch.nn as nn
import torch.nn.functional as F


def max_over_time(out):
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

		head_vec = max_over_time(head)
		body_vec = max_over_time(body)
		input_features = torch.cat((head_vec, body_vec),axis=1)

		output = F.relu(self.bn_dense1(self.dense1(input_features)))
		output = F.relu(self.bn_dense2(self.dense2(output)))
		output = F.relu(self.bn_dense3(self.dense3(output)))
		output = F.relu(self.dense4(output))

		return self.logsoftmax(output)