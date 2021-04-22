def create_checkpoint():
	checkpoint = {
		"epoch": 0,
		"model_state_dict": None,
		"optimizer_state_dict": None,
		"best_train_acc": 0.,
		"train_acc_history":None,
		"test_acc_history":None,
	}

	return checkpoint

def update_checkpoint(checkpoint,epoch,state_dict,optim_dict,best_acc,train_hist,test_hist):
	checkpoint["epoch"] = epoch
	checkpoint["model_state_dict"] = state_dict
	checkpoint["optimizer_state_dict"] = optim_dict
	checkpoint["best_train_acc"] = best_acc
	checkpoint["train_acc_history"] = train_hist
	checkpoint["test_acc_history"] = test_hist
	return checkpoint