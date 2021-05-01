import numpy as np

import matplotlib.pyplot as plt

def plot_history(filename,train_values,val_values,title,xlabel,ylabel):

	fig, axs = plt.subplots(1)

	fig.suptitle(title)
	
	plt.plot(np.arange(len(train_values)), train_values)
	plt.plot(np.arange(len(val_values)), val_values)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	plt.savefig(filename)