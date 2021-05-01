##### PYTORCH ##########
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

##### MODELS #######
from models.architectures import FakeNewsCNN, FakeNewsBiLSTM

##### EMBEDDER ######
from embedding.GoogleEmbedding import GoogleVectors
from embedding.ELMoEmbedding import ELMoVectors

##### DATASET ######
from models.dataloader import FakeNewsDataSet

##### FUNCTIONS #####
from utils.saveload import create_checkpoint, update_checkpoint
from utils.metrics import scoring, max_scoring, null_scoring

##### SYSTEM ######
import os
import argparse
import shutil
import yaml
import time
import tqdm

############################ PROCESS ARGUMENTS ##################################

parser = argparse.ArgumentParser(description="Train process of a Fake News Classifier")

# data arguments
parser.add_argument("--train_path","-tp", type=str, default='data/train',
					help="Directory path with stances.csv and bodies.csv")
parser.add_argument("--val_path","-vp", type=str, default='data/val',
					help="Directory path with stances.csv and bodies.csv")
# saving arguments
parser.add_argument("--save_path","-sp", type=str, default='models/saves',
					help="Directory path to save checkpoint model")
parser.add_argument("--checkpoint_name","-cn", type=str, default='checkpoint_CNN',
					help="Checkpoint model filename (.pt file)")

# classifier and embedder model arguments
parser.add_argument("--config_path","-cp", type=str, default='config/config.yaml',
					help="Training process configuration path (.yaml)")

# reusme arguments
parser.add_argument("--resume", "-r", action="store_true",
					help="Name of the checkpoint to resume")
parser.add_argument("--load_path","-lp", type=str, default='models/saves/checkpoint_GOOGLE_CNN',
					help="Directory path to load checkpoint if resume is True")
parser.add_argument("--config_name","-cfn", type=str, default='config.yaml',
					help="Configuration name in the loaded checkpoint if resume is True")
parser.add_argument("--acc","-a", action="store_true",
					help="If True, use best acc as checkpoint else, the best score one (default)")

args = parser.parse_args()

TRAIN_PATH = args.train_path
VAL_PATH = args.val_path
SAVE_PATH = args.save_path
RESUME = args.resume
ACC = args.acc

print(f"Resuming a training process : {RESUME}")

############################ PROCESS DEVICE ##################################

# Get device for process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Process perform on device : {device}")


############################ CHECKPOINT INITIALISATION ##################################

if RESUME:
	PATH_CHPT = args.load_path
	# Saving path
	score_path = os.path.join(PATH_CHPT, 'best_score.pt')
	acc_path = os.path.join(PATH_CHPT, 'best_acc.pt')
	# Loading of the checkpoint
	acc_checkpoint = torch.load(acc_path, map_location=torch.device(device))
	score_checkpoint = torch.load(score_path, map_location=torch.device(device))
	# Checkpoint used for optimizer and network weights and, parameters and history
	main_checkpoint = score_checkpoint
	if ACC : main_checkpoint = acc_checkpoint

else:
	CHPT_NAME = args.checkpoint_name
	PATH_CHPT = os.path.join(SAVE_PATH, CHPT_NAME)
	os.makedirs(PATH_CHPT,exist_ok=True)
	# Saving path
	score_path = os.path.join(PATH_CHPT, 'best_score.pt')
	acc_path = os.path.join(PATH_CHPT, 'best_acc.pt')
	# Creation of the checkpoint
	acc_checkpoint = create_checkpoint()
	score_checkpoint = create_checkpoint()
	main_checkpoint = score_checkpoint

print(f"Best score checkpoint path : {score_path}")
print(f"Best acc checkpoint path : {acc_path}")


############################ CONFIGURATION EXTRACTION ##################################

if RESUME:
	CONFIG_NAME = args.config_name
	CONFIG_PATH = os.path.join(PATH_CHPT,CONFIG_NAME)
else:
	CONFIG_PATH = args.config_path

# Reading configuration of the process
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
# copy of the config in checkpoint
if not RESUME: shutil.copy2(CONFIG_PATH, PATH_CHPT)

print(f"Process configuration path : {CONFIG_PATH}")


############################ EMBEDDER MODEL ##################################

name_embedder = config['embedder'].pop('name')

if name_embedder == 'GOOGLE':
	model_vectors = GoogleVectors()
	embedding_dim = model_vectors.get_embedding_size()

if name_embedder == 'ELMO':
	size_elmo = config['embedder'].pop('size')
	model_vectors = ELMoVectors(size_elmo,device)
	embedding_dim = model_vectors.get_embedding_size()

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
if RESUME: model.load_state_dict(main_checkpoint["model_state_dict"])


############################ TRAIN AND VAL DATASET  ##################################

shuffle = config['training'].pop('shuffle')

train_dataset = FakeNewsDataSet(stances=os.path.join(TRAIN_PATH,'stances.csv'),
								bodies=os.path.join(TRAIN_PATH,'bodies.csv'),
								vec_embedding=model_vectors, shuffle=shuffle)
train_size = train_dataset.get_len()

# val dataset and dataloader
val_dataset = FakeNewsDataSet(stances=os.path.join(VAL_PATH,'stances.csv'),
								bodies=os.path.join(VAL_PATH,'bodies.csv'),
								vec_embedding=model_vectors)
val_size = val_dataset.get_len()

print("\n-- Train Data --")
print(f' * Number of news {train_size}')
print(f' * Maximum score {max_scoring(train_dataset)}')
print(f' * Null score {null_scoring(train_dataset)}')

print("\n-- Valdation Data --")
print(f' * Number of news {val_size}')
print(f' * Maximum score {max_scoring(val_dataset)}')
print(f' * Null score {null_scoring(val_dataset)}')


############################ PROCESS LOSS OPTIMIZER AND PARAMETERS  ##################################

# Process configuration parameters
n_epoch = config['training'].pop('n_epoch')
batch_size = config['training'].pop('batch_size')
lr = config['optimizer'].pop('learning_rate')

criterion = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
if RESUME: optimizer.load_state_dict(main_checkpoint["optimizer_state_dict"])

# Checkpoint initial values
best_score = main_checkpoint['best_score']
best_acc = main_checkpoint['best_acc']
train_acc_history = main_checkpoint['train_acc_history']
val_acc_history = main_checkpoint['val_acc_history']
train_score_history = main_checkpoint['train_score_history']
val_score_history = main_checkpoint['val_score_history']
start_epoch = main_checkpoint['epoch']
end_epoch = start_epoch + n_epoch

print("\n-- Initial process parameters --")
print(f" * Number epoch : {n_epoch}")
print(f" * Start epoch : {start_epoch}")
print(f" * End epoch : {end_epoch}")
print(f" * Batch size : {batch_size}")
print(f" * Adam learning rate : {lr}")
print(f" * Val best accuracy : {best_acc}%")
print(f" * Val best score : {best_score}")


############################ PROCESS LOOP  ##################################

for epoch in range(start_epoch,end_epoch):

	#*********** Epoch initialisation *********#
	print(f"\n----- EPOCH {epoch +1}/{end_epoch} -----")

	#*********** Training and Validation *********#

	for part in ['Train','Val']:

		n_correct = 0
		score = 0

		if part == 'Train': 
			model.train()
			dataset = train_dataset
			size = train_size

		if part == 'Val': 
			model.eval()
			dataset = val_dataset
			size = val_size
	
		i_sample = 0
		t_0 = time.time()
		pbar = tqdm.tqdm(total = size)
		while dataset.is_epoch == False:

			# get batch as tensor
			heads_emb, bodies_emb, stances = dataset.get_batch(batch_size)

			heads_emb = heads_emb.to(device)
			bodies_emb = bodies_emb.to(device)
			stances = stances.to(device)
			
			output = model([heads_emb,bodies_emb])

			if part == 'Train':
				loss = criterion(output,stances)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			n_correct += (torch.max(output,1).indices==stances).sum().item()
			score += scoring(output,stances)
			i_sample += batch_size
			pbar.update(batch_size)
			
			# del useless variables from here
			del heads_emb, bodies_emb, stances, output

		pbar.close()

		t_n = time.time() - t_0
		print(f'Execution time :{t_n/60}min')
		
		acc = 100. * n_correct / size
		print(f"{part} accuracy : {acc}%")
		print(f"{part} score : {score}")

		#*********** Checkpoint update *********#
		
		if part == 'Train':
			# Saving accuracy history
			train_acc_history.append(acc)
			train_score_history.append(score)

		if part == 'Val':
			# Saving accuracy history
			val_acc_history.append(acc)
			val_score_history.append(score)

			# Best score checkpoint update
			if score > best_score:
				score_checkpoint = update_checkpoint(score_checkpoint,epoch+1,model.state_dict(),optimizer.state_dict(),acc,score,
											   		 train_acc_history,val_acc_history,train_score_history,val_score_history)
				best_score = score
				torch.save(score_checkpoint, score_path)
				print("Best score checkpoint saved")

			# Best acc checkpoint update
			if acc > best_acc:
				acc_checkpoint = update_checkpoint(acc_checkpoint,epoch+1,model.state_dict(),optimizer.state_dict(),acc,score,
											   	   train_acc_history,val_acc_history,train_score_history,val_score_history)
				best_acc = acc
				torch.save(acc_checkpoint, acc_path)
				print("Best acc checkpoint saved")

		# specify that a new epoch must begin
		dataset.is_epoch = False

