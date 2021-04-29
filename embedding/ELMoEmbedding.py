import string
import time 
import numpy as np
import torch

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

class ELMoVectors(object):
    def __init__(self,device):
        self.device = device
        self.model = Elmo(options_file, weight_file, 1, dropout=0).to(device)

    def transform(self, X):
        # split all text by sentence for character embeding of a sentence
        for i in range(len(X)):
          X[i]=X[i].split('.')
        c = batch_to_ids(X)
        c = torch.LongTensor(c).to(self.device)
        v = self.model(c)

        return v['elmo_representations'][0]

if __name__ == '__main__':
    elmov = ELMoVectors()

    t = [['the quick brown fox jumped over the lazy dog','the. quick, brown! fox,, !']]
    x=elmov.transform(t)
    print(x)
    print(x.shape)