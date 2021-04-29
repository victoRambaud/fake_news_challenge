def create_checkpoint():
	checkpoint = {
		"epoch": 0,
		"model_state_dict": None,
		"optimizer_state_dict": None,
		"best_acc": 0.,
		"best_score": 0.,
		"train_acc_history":[],
		"val_acc_history":[],
		"train_score_history":[],
		"val_score_history":[],
	}

	return checkpoint

def update_checkpoint(checkpoint,epoch,state_dict,optim_dict,best_acc,best_score,train_acc_hist,val_acc_hist,train_score_hist,val_score_hist):
	checkpoint["epoch"] = epoch
	checkpoint["model_state_dict"] = state_dict
	checkpoint["optimizer_state_dict"] = optim_dict
	checkpoint["best_acc"] = best_acc
	checkpoint["best_score"] = best_score
	checkpoint["train_acc_history"] = train_acc_hist
	checkpoint["val_acc_history"] = val_acc_hist
	checkpoint["train_score_history"] = train_score_hist
	checkpoint["val_score_history"] = val_score_hist
	return checkpoint

if __name__== '__main__':
	import os
	import torch
	PATH_LOAD = '../models/saves/checkpoint_BILSTM_double.pt'
	checkpoint = torch.load(PATH_LOAD,map_location=torch.device('cpu'))
	print(checkpoint['epoch'],':',checkpoint['best_train_acc'])
	print(checkpoint['epoch'],':',checkpoint['val_acc_history'][checkpoint['epoch']])
