import torch 
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.CNN import FakeNewsCNN
from models.biLSTM import FakeNewsBiLSTM
from models.dataloader import FakeNewsDataSet

from utils.saveload import create_checkpoint, update_checkpoint
from utils.metrics import scoring, max_scoring, null_scoring

from embedding.GoogleEmbedding import GoogleVectors
from embedding.ELMoEmbedding import ELMoVectors

import tqdm
import os
import argparse
import shutil
import yaml
import time

############################ PROCESS ARGUMENTS ##################################

parser = argparse.ArgumentParser(description="Train process of a Fake News Classifier")

# data arguments
parser.add_argument("--test_path","-tp", type=str, default='data/test',
					help="Directory path with stances.csv and bodies.csv")
parser.add_argument("--load_path","-lp", type=str, default='models/saves/checkpoint_GOOGLE_CNN',
					help="Directory path to load checkpoint if resume is True")
parser.add_argument("--config_name","-cfn", type=str, default='config_CNN.yaml',
					help="Configuration name in the loaded checkpoint if resume is True")

args = parser.parse_args()

TEST_PATH = args.test_path

############################ PROCESS DEVICE ##################################

# Get device for process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Process perform on device : {device}")


############################ CHECKPOINT INITIALISATION ##################################

PATH_CHPT = args.load_path
# Saving path
score_path = os.path.join(PATH_CHPT, 'best_score.pt')
acc_path = os.path.join(PATH_CHPT, 'best_acc.pt')
# Loading of the checkpoint
acc_checkpoint = torch.load(acc_path, map_location=torch.device(device))
score_checkpoint = torch.load(score_path, map_location=torch.device(device))

print(f"Best score checkpoint path : {score_path}")
print(f"Best acc checkpoint path : {acc_path}")


############################ CONFIGURATION EXTRACTION ##################################

# get configuration checkpoint
CONFIG_NAME = args.config_name
CONFIG_PATH = os.path.join(PATH_CHPT,CONFIG_NAME)

# Reading configuration of the process
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

print(f"Checkpoint configuration path : {CONFIG_PATH}")


############################ EMBEDDER MODEL ##################################

name_embedder = config['embedder'].pop('name')

if name_embedder == 'GOOGLE':
	model_vectors = GoogleVectors()
	embedding_dim = 300

if name_embedder == 'ELMO':
	model_vectors = ELMoVectors(device)
	embedding_dim = 1024

print(f"\nEmbedder model : {name_embedder} -> dim:{embedding_dim}")


############################ NETWORK MODEL  ##################################

name_network = config['network'].pop('name')

if name_network == 'CNN':
	hidden_size = config['network'].pop('hidden_size')
	n_classes = config['network'].pop('n_classes')
	model = FakeNewsCNN(embedding_dim=embedding_dim,hidden_size=hidden_size,n_classes=n_classes)
	print(f"Network model : {name_network}")
	print(f" * hidden_size : {hidden_size}")
	print(f" * n_classes : {n_classes}")

if name_network == 'BILSTM':
	hidden_size = config['network'].pop('hidden_size')
	num_layers = config['network'].pop('num_layers')
	n_classes = config['network'].pop('n_classes')
	model = FakeNewsBiLSTM(embedding_dim=embedding_dim,hidden_size=hidden_size,num_layers=num_layers,
						   n_classes=n_classes,device=device)
	print(f"Network model : {name_network}")
	print(f" * hidden_size : {hidden_size}")
	print(f" * num_layers : {num_layers}")
	print(f" * n_classes : {n_classes}")

model.to(device)
# weights will be charge after for best acc and best score checkpoint


############################ TEST DATASET  ##################################

test_dataset = FakeNewsDataSet(stances=os.path.join(TEST_PATH,'stances.csv'),
								bodies=os.path.join(TEST_PATH,'bodies.csv'),
								vec_embedding=model_vectors)
test_size = test_dataset.get_len()

print("\n** Test Data **")
print(f'Number of news {test_size}')
print(f'Maximum score {max_scoring(test_dataset)}')
print(f'Null score {null_scoring(test_dataset)}')


############################ PROCESS PARAMETERS  ##################################

batch_size = 32


############################ PROCESS LOOP  ##################################

for chpt in ['Accuracy','Score']:

	if chpt == 'Accuracy': main_checkpoint = acc_checkpoint
	if chpt == 'Score': main_checkpoint = score_checkpoint

	# Load model
	model.load_state_dict(main_checkpoint["model_state_dict"])

	#*********** Version initialisation *********#
	print(f"\n----- BEST CHECKPOINT : {chpt} -----")

	#*********** Testing *********#

	n_correct = 0
	score = 0

	i_sample = 0
	t_0 = time.time()
	while test_dataset.is_epoch == False:
		print(f"Training samples {i_sample}/{test_size}",end='\r')

		# get batch as tensor
		heads_emb, bodies_emb, stances = test_dataset.get_batch(batch_size)

		heads_emb = heads_emb.to(device)
		bodies_emb = bodies_emb.to(device)
		stances = stances.to(device)
		
		output = model([heads_emb,bodies_emb])

		n_correct += (torch.max(output,1).indices==stances).sum().item()
		score += scoring(output,stances)
		i_sample += batch_size
	
	t_n = time.time() - t_0
	print(f'Execution time :{t_n}')
	
	acc = 100. * n_correct / size
	print(f"Accuracy : {acc}%")
	print(f"Score : {score}")

	# specify that a new epoch must begin
	dataset.is_epoch = False

