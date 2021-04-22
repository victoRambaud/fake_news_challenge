import torch 
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.CNN import FakeNewsCNN
from models.biLSTM import FakeNewsBiLSTM
from models.dataloader import FakeNewsDataSet

from utils.saveload import create_checkpoint, update_checkpoint

from embedding.GoogleEmbedding import GoogleVectors

import tqdm
import os

TRAIN_PATH = 'data/train'
TEST_PATH = 'data/test'
SAVE_PATH = 'models/saves'
CHPT_NAME = 'checkpoint.pt'
PATH_CHPT = os.path.join(SAVE_PATH, CHPT_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device {device}")

os.makedirs(SAVE_PATH, exist_ok=True)
checkpoint = create_checkpoint()

# model for word to vector embedding
model_vectors = GoogleVectors(path='embedding/vectors/GoogleNews-vectors-negative300.bin')

# train dataset and dataloader
train_dataset = FakeNewsDataSet(stances=os.path.join(TRAIN_PATH,'train_stances.csv'),
								bodies=os.path.join(TRAIN_PATH,'train_bodies.csv'),
								vec_embedding=model_vectors, shuffle=True)

# test dataset and dataloader
test_dataset = FakeNewsDataSet(stances=os.path.join(TEST_PATH,'test_stances.csv'),
								bodies=os.path.join(TEST_PATH,'test_bodies.csv'),
								vec_embedding=model_vectors)

model = FakeNewsCNN(hidden_size=64)
#model = FakeNewsBiLSTM(300,4,50,1,device)
model.to(device)

criterion = CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

n_ite = 100000
size_epoch = train_dataset.get_len()
batch_size = 16
print("------- Train Data -------")
print(f'Number of news {train_dataset.get_len()}')
#print(f'Number of bacth {len(train_dataloader)}')
print("\n------- Test Data -------")
print(f'Number of news {test_dataset.get_len()}')
#print(f'Number of bacth {len(test_dataloader)}')

def scoring(output,stances):
	score=0.
	labels = torch.max(output,1).indices
	for l,s in zip(labels,stances):
		# unrelated
		if s == 3:
			if l == s:
				score += 0.25
		# related
		else:
			# not unrelated
			if l != 3:
				score += 0.25
			if l == s:
				score += 0.75
	return score

epoch=0
best_acc = 0.
train_acc_history = []
test_acc_history = []
train_n_correct = 0
train_score = 0
for ite in tqdm.tqdm(range(n_ite)):

	if train_dataset.ite_epoch == 0:
		print(f"\n----- EPOCH {epoch +1} -----")
		epoch+=1
		model.train()

	# get batch
	heads_emb, bodies_emb, stances = train_dataset.get_batch(batch_size)

	heads_emb = torch.Tensor(heads_emb).to(device)
	bodies_emb = torch.Tensor(bodies_emb).to(device)
	stances = torch.LongTensor(stances).to(device)

	output = model([heads_emb,bodies_emb])

	train_loss = criterion(output,stances)

	optimizer.zero_grad()
	train_loss.backward()
	optimizer.step()

	train_n_correct += (torch.max(output,1).indices==stances).sum().item()
	train_score += scoring(output,stances)

	if train_dataset.ite_epoch == 0:
		train_acc = 100. * train_n_correct / size_epoch
		print(f"Train accuracy : {train_acc}%")
		print(f"Train score : {train_score}")
		train_n_correct = 0
		train_score = 0

		model.eval()
		test_n_correct = 0
		test_score = 0
		while test_dataset.ite_epoch < test_dataset.get_len():
			# get batch
			heads_emb, bodies_emb, stances = test_dataset.get_batch(batch_size)

			heads_emb = torch.Tensor(heads_emb).to(device)
			bodies_emb = torch.Tensor(bodies_emb).to(device)
			stances = torch.LongTensor(stances).to(device)

			output = model([heads_emb,bodies_emb])

			test_loss = criterion(output,stances)

			test_n_correct += (torch.max(output,1).indices==stances).sum().item()
			test_score += scoring(output,stances)

		test_acc = 100. * test_n_correct / test_dataset.get_len()
		print(f"Test Accuracy : {test_acc}%")
		print(f"Test score : {test_score}")

		# Saving accuracy history
		train_acc_history.append(train_acc)
		test_acc_history.append(test_acc)

		if test_acc > best_acc:
			print(epoch)
			checkpoint = update_checkpoint(checkpoint,epoch,model.state_dict(),optimizer.state_dict(),best_acc,train_acc_history,test_acc_history)
			best_acc = test_acc
			torch.save(checkpoint, PATH_CHPT)

