import torch
from torch.autograd import Variable

from allennlp.modules.elmo import Elmo, batch_to_ids

import numpy as np

import string
import time 

# SMALL (13.6M params) -> dim:256
# MEDIUM (26M params) -> dim:512
# BIG ORIGINAL (93M params) -> dim:1024

elmo_emb_size = {
    'Small': 256,
    'Medium': 512,
    'Big': 1024
}

options_files = {
    'Small': "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",
    'Medium': "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json",
    'Big': "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
}

weight_files = {
    'Small': "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
    'Medium': "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5",
    'Big': "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
}

class ELMoVectors(object):
    def __init__(self,size_elmo,device):
        self.size_elmo = size_elmo
        self.device = device
        self.model = Elmo(options_files[size_elmo], weight_files[size_elmo], 1, dropout=0.,requires_grad=False)
        self.model.to(device)

    def get_embedding_size(self):
        return elmo_emb_size[self.size_elmo]

    def transform(self, X):
        # split all text by sentence for character embeding of a sentence
        X = self.tokenize(X)
        word_token = batch_to_ids(X).to(self.device)
        #word_emb = torch.LongTensor(word_emb).to(self.device)
        word_emb = self.model(word_token)

        # del useless varaibles
        del word_token

        return word_emb['elmo_representations'][0]

    def tokenize(self, X):
        for i in range(len(X)):
            X[i]=X[i].split(' ')
        return X