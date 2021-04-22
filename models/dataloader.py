import csv
from torch.utils.data import Dataset, DataLoader
import numpy as np

def process_bodies(file):
    tmp = {}
    with open(file,'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            body_id, body_text = line
            tmp[body_id]=body_text
    return tmp

def process_headlines(file):
    tmp = []
    with open(file,'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            headline, body_id, stance = line
            tmp.append((headline,body_id,stance))
    return tmp

class FakeNewsDataSet(object):

    def __init__(self, stances='../data/train_stances.csv',bodies='../data/train_bodies.csv',vec_embedding=None,shuffle=False):
        self.bodies = process_bodies(bodies)
        self.headlines = process_headlines(stances)
        self.vec=vec_embedding
        self.shuffle=shuffle
        self.index= np.arange(self.get_len()) 
        self.stances_label = {'agree':0,'disagree':1,'discuss':2,'unrelated':3}
        self.ite_epoch = 0

    def get_batch(self, batch_size):
    	heads_text = []
    	bodies_text = []
    	stances = []

    	start = self.ite_epoch
    	end = start+batch_size
    	batch_index=self.index[start:end]

    	self.ite_epoch += len(batch_index)

    	for i in batch_index:
    		head = self.headlines[i]
    		head_text, body_id , stance = head
    		body_text = self.bodies[body_id]
    		heads_text.append(head_text)
    		bodies_text.append(body_text)
    		stances.append(self.stances_label[stance])

    	heads_vec = self.vec.transform(heads_text)
    	bodies_vec = self.vec.transform(bodies_text)

    	heads_emb = np.apply_along_axis(self.get_emb, 1, heads_vec)
    	bodies_emb = np.apply_along_axis(self.get_emb, 1, bodies_vec)

    	# Update of ite_epoch and if it is the last shuffle and back to 0
    	if self.ite_epoch >= self.get_len():
    		if self.shuffle: self.shuffle_index()
    		print(self.ite_epoch)

    	heads_emb = np.transpose(heads_emb, (0, 2, 1))
    	bodies_emb = np.transpose(bodies_emb, (0, 2, 1))

    	return heads_emb, bodies_emb, stances

    def get_len(self):
        return len(self.headlines)

    def get_emb(self,x):
    	return self.vec.model[x]

    def shuffle_index(self):
    	print('Fake News dataset is shuffled')
    	np.random.shuffle(self.index)
    	self.ite_epoch=0

if __name__ == '__main__':

	dataset = FakeNewsDataSet()
	dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
	for i,input in enumerate(dataloader):
		print(input[0])
		print(input[1])
		print(input[2])
		print("\n\n")
		if i > 0:
			break


