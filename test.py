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

parser = argparse.ArgumentParser(description="Evaluation process of a Fake News Classifier")

# data arguments
parser.add_argument("--test_path","-tp", type=str, default='data/test',
					help="Directory path with stances.csv and bodies.csv")
# loading arguments
parser.add_argument("--load_path","-sp", type=str, default='models/saves',
					help="Directory path to load checkpoint model")
parser.add_argument("--checkpoint_name","-cn", type=str, default='checkpoint_CNN.pt',
					help="Checkpoint model filename (.pt file)")

# classifier model arguments

# text embedding model arguments

args = parser.parse_args()

TEST_PATH = args.test_path
LOAD_PATH = args.load_path
CHPT_NAME = args.checkpoint_name
PATH_CHPT = os.path.join(LOAD_PATH, CHPT_NAME)

# loading checkpoint
print('Loading checkpoint')
checkpoint = torch.load(PATH_CHPT,map_location=torch.device('cpu'))

# model for word to vector embedding
model_vectors = GoogleVectors()

# test dataset and dataloader
test_dataset = FakeNewsDataSet(stances=os.path.join(TEST_PATH,'stances.csv'),
								bodies=os.path.join(TEST_PATH,'bodies.csv'),
								vec_embedding=model_vectors)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Evaluation process perform on device : {device}")

model = FakeNewsCNN(hidden_size=256)
#model = FakeNewsBiLSTM(n_features=300,classes=4,hidden_size=200,num_layers=1,device=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

batch_size = 64
test_size = test_dataset.get_len()

print("\n** Test Data **")
print(f'Number of news {test_size}')
print(f'Maximum score {max_scoring(test_dataset)}')
print(f'Null score {null_scoring(test_dataset)}')

test_n_correct = 0
test_score = 0

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

	test_n_correct += (torch.max(output,1).indices==stances).sum().item()
	test_score += scoring(output,stances)

test_acc = 100. * test_n_correct / test_size
print(f"Test Accuracy : {test_acc}%")
print(f"Test score : {test_score}")
