import torch 
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.CNN import FakeNewsCNN
from models.biLSTM import FakeNewsBiLSTM
from models.dataloader import FakeNewsDataSet

from utils.saveload import create_checkpoint, update_checkpoint
from utils.metrics import scoring, max_scoring, null_scoring

from embedding.GoogleEmbedding import GoogleVectors

import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description="Train process of a Fake News Classifier")

# data arguments
parser.add_argument("--train_path","-tp", type=str, default='data/train',
					help="Directory path with stances.csv and bodies.csv")
parser.add_argument("--val_path","-vp", type=str, default='data/val',
					help="Directory path with stances.csv and bodies.csv")
# saving arguments
parser.add_argument("--save_path","-sp", type=str, default='models/saves',
					help="Directory path to save checkpoint model")
parser.add_argument("--checkpoint_name","-cn", type=str, default='checkpoint_CNN.pt',
					help="Checkpoint model filename (.pt file)")

# classifier model arguments

# text embedding model arguments

# processus arguments
parser.add_argument("--epoch","-e", type=int, default=20, help="Number of epochs")
parser.add_argument("--batch_size","-b", type=int, default=16, help="Batch size (padding is done by batch)")
parser.add_argument("--learning_rate","-lr", type=float, default=0.001, help="Learning rate (Adam optimizer)")


args = parser.parse_args()

TRAIN_PATH = args.train_path
TEST_PATH = args.val_path
SAVE_PATH = args.save_path
CHPT_NAME = args.checkpoint_name
PATH_CHPT = os.path.join(SAVE_PATH, CHPT_NAME)

os.makedirs(SAVE_PATH, exist_ok=True)
checkpoint = create_checkpoint()

# model for word to vector embedding
model_vectors = GoogleVectors()

# train dataset and dataloader
train_dataset = FakeNewsDataSet(stances=os.path.join(TRAIN_PATH,'stances.csv'),
								bodies=os.path.join(TRAIN_PATH,'bodies.csv'),
								vec_embedding=model_vectors, shuffle=True)

# test dataset and dataloader
test_dataset = FakeNewsDataSet(stances=os.path.join(TEST_PATH,'stances.csv'),
								bodies=os.path.join(TEST_PATH,'bodies.csv'),
								vec_embedding=model_vectors)

n_epoch = args.epoch
batch_size = args.batch_size
lr = args.learning_rate
train_size = train_dataset.get_len()
test_size = test_dataset.get_len()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training process perform on device : {device}")

model = FakeNewsCNN(hidden_size=256)
#model = FakeNewsBiLSTM(n_features=300,classes=4,hidden_size=50,num_layers=1,device=device)
model.to(device)

criterion = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

print("** Train Data **")
print(f'Number of news {train_size}')
print(f'Maximum score {max_scoring(train_dataset)}')
print(f'Null score {null_scoring(train_dataset)}')
print("\n** Val Data **")
print(f'Number of news {test_size}')
print(f'Maximum score {max_scoring(test_dataset)}')
print(f'Null score {null_scoring(test_dataset)}')

epoch=0
best_score = 0.
train_acc_history = []
test_acc_history = []
train_n_correct = 0
train_score = 0

for epoch in range(n_epoch):
#for ite in tqdm.tqdm(range(n_ite)):

	#if train_dataset.ite_epoch == 0:
	print(f"\n----- EPOCH {epoch +1} -----")

	train_n_correct = 0
	train_score = 0
	test_n_correct = 0
	test_score = 0

	model.train()
	i_sample = 0
	while train_dataset.is_epoch == False:
		i_sample += batch_size
		print(f"Training samples {i_sample}/{train_size}",end='\r')
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

	#if train_dataset.ite_epoch == 0:
	train_acc = 100. * train_n_correct / train_size
	print(f"Train accuracy : {train_acc}%")
	print(f"Train score : {train_score}")

	model.eval()
	i_sample = 0
	while test_dataset.is_epoch == False:
		i_sample += batch_size
		print(f"Testing samples {i_sample}/{test_size}",end='\r')
		# get batch
		heads_emb, bodies_emb, stances = test_dataset.get_batch(batch_size)

		heads_emb = torch.Tensor(heads_emb).to(device)
		bodies_emb = torch.Tensor(bodies_emb).to(device)
		stances = torch.LongTensor(stances).to(device)

		output = model([heads_emb,bodies_emb])

		test_loss = criterion(output,stances)

		test_n_correct += (torch.max(output,1).indices==stances).sum().item()
		test_score += scoring(output,stances)

	test_acc = 100. * test_n_correct / test_size
	print(f"Test Accuracy : {test_acc}%")
	print(f"Test score : {test_score}")

	# Saving accuracy history
	train_acc_history.append(train_acc)
	test_acc_history.append(test_acc)

	if test_score > best_score:
		checkpoint = update_checkpoint(checkpoint,epoch,model.state_dict(),optimizer.state_dict(),test_score,train_acc_history,test_acc_history)
		best_score = test_score
		torch.save(checkpoint, PATH_CHPT)

	## specify that a new epoch must begin
	train_dataset.is_epoch = False
	test_dataset.is_epoch = False
