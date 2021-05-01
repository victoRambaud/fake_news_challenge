import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import csv

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

    def __init__(self, stances,bodies,vec_embedding=None,shuffle=False):
        self.bodies = process_bodies(bodies)
        self.headlines = process_headlines(stances)
        self.vec=vec_embedding
        self.shuffle=shuffle
        self.index= np.arange(self.get_len()) 
        self.stances_label = {'agree':0,'disagree':1,'discuss':2,'unrelated':3}
        self.ite_epoch = 0
        self.is_epoch = False

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

        # get embedding and stances as tensor
        heads_vec = self.vec.transform(heads_text)
        bodies_vec = self.vec.transform(bodies_text)
        stances = torch.LongTensor(stances)

        heads_emb = torch.transpose(heads_vec, 1, 2)
        bodies_emb = torch.transpose(bodies_vec, 1, 2)

        # Update of ite_epoch and if it is the last shuffle and back to 0
        if self.ite_epoch == self.get_len():
            self.ite_epoch=0
            self.is_epoch = True
        if self.shuffle: self.shuffle_index()

        return heads_emb, bodies_emb, stances

    def get_len(self):
        return len(self.headlines)

    def get_len_bodies(self):
        return len(self.bodies)

    def shuffle_index(self):
        np.random.shuffle(self.index)


