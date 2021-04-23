import torch

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
