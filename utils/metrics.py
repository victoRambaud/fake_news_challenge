import torch
import numpy as np
import csv
import pandas as pd

def scoring(output,stances):
	score=0.
	labels = torch.max(output,1).indices
	for l,s in zip(labels,stances):
		# unrelated
		if s == l == 3:
			score += 0.25
		# related
		if s != 3 and l != 3:
			score += 0.25
		if l == s != 3:
			score += 0.75
	return score

# dataset is a FakeNewsDataset
# if accuracy is 100%
def max_scoring(dataset):
	max_score = 0
	for i in dataset.index:
		stance = dataset.headlines[i][2]
		s = dataset.stances_label[stance]
		#unrelated
		if s == 3:
			max_score += 0.25
		#related
		else:
			max_score += 1.0

	return max_score

# dataset is a FakeNewsDataset
# if every stance is predicted unrelated
def null_scoring(dataset):
	null_score = 0
	l = 3
	for i in dataset.index:
		stance = dataset.headlines[i][2]
		s = dataset.stances_label[stance]
		# unrelated
		if s == l == 3:
			null_score += 0.25

	return null_score