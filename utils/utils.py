import numpy as np

import matplotlib.pyplot as plt

def plot_history(filename,checkpoint,title,xlabel,ylabel,save=True):
	
	fig, axs = plt.subplots(1)

	fig.suptitle(title)
	
	plt.plot(np.arange(len(train_values)), train_hist)
	plt.plot(np.arange(len(val_values)), test_hist)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	plt.savefig(filename)