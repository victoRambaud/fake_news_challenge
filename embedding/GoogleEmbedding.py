import torch

from gensim.models import KeyedVectors
import gensim.downloader as api

import numpy as np

import string

class GoogleVectors(object):
    def __init__(self):
        self.model = self.load_google_vec()
        self.vocab = self.load_vocab()
        self.translator = str.maketrans('', '', string.punctuation)

    def get_embedding_size(self):
        return 300

    def load_google_vec(self):
        model = api.load("word2vec-google-news-300")
        return model

    def load_vocab(self):
        vocab = {}
        for i in range(len(self.model.index_to_key)):
            vocab[self.model.index_to_key[i]]=i
        return vocab

    def transform(self, X, pad=0):
        tmp = []
        maxlen=0
        for t in X:
            t = t.translate(self.translator)
            v = self.vectorize(t, vocab=self.vocab)
            if len(v)>maxlen:
                maxlen=len(v)
            tmp.append(v)
        e = np.zeros((len(X),maxlen)).astype('int32')
        e.fill(pad)

        for i in range(len(tmp)):
            v = tmp[i]
            e[i,:len(v)]=v

        emb = np.apply_along_axis(self.get_emb, 1, e)
        return torch.Tensor(emb)

    def get_emb(self,x):
        return self.model[x]

    def vectorize(self,text, vocab={}):
        opts = [i for i in text.split(' ') if len(i)>0]

        cidx = 0
        tmp = []
        while cidx<len(opts):
            c0 = opts[cidx]
            if cidx+1<len(opts):
                c1 = opts[cidx+1]
            else:
                c1 = False
            if cidx+2<len(opts):
                c2 = opts[cidx+2]
            else:
                c2 = False
            if c2:
                s = c0+'_'+c1+'_'+c2
                if s in vocab:
                    tmp.append(vocab[s])
                    cidx+=3
                    continue
            else:
                pass
            if c1:
                s = c0+'_'+c1
                if s in vocab:
                    tmp.append(vocab[s])
                    cidx+=2
                    continue
            else:
                pass
            if c0 in vocab:
                tmp.append(vocab[c0])
            elif c0.lower() in vocab:
                tmp.append(vocab[c0.lower()])
            else:
                tmp.append(vocab['</s>'])
                pass
            cidx+=1
        return tmp