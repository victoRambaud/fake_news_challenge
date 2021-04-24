import string
import time 
import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api

def vectorize(text, vocab={}):
    opts = [i for i in text.split(' ') if len(i)>0]

    cidx = 0
    tmp = []
    #iterate over the entire text, construct 3/2 grams first
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
        #no 3/2 grams, check the word
        if c0 in vocab:
            tmp.append(vocab[c0])
        elif c0.lower() in vocab:
            tmp.append(vocab[c0.lower()])
        else:
            #we have no token at this timestep, we could add a default?
            tmp.append(vocab['</s>'])
            pass
        cidx+=1
    return tmp

class GoogleVectors(object):
    def __init__(self):
        self.model = self.load_google_vec()
        self.vocab = self.load_vocab()
        self.translator = str.maketrans('', '', string.punctuation)

    def load_google_vec(self):
        print("Loading Google News Vectors Embedding ...")
        model = api.load("word2vec-google-news-300")
        return model

    def load_vocab(self):
        print("Loading vocabulary from embedding model ...")
        vocab = {}
        for i in range(len(self.model.index_to_key)):
            vocab[self.model.index_to_key[i]]=i
        return vocab

    def transform(self, X, pad=0):
        tmp = []
        maxlen=0
        for t in X:
            t = t.translate(self.translator)
            v = vectorize(t, vocab=self.vocab)
            if len(v)>maxlen:
                maxlen=len(v)
            tmp.append(v)
        e = np.zeros((len(X),maxlen)).astype('int32')
        e.fill(pad)

        for i in range(len(tmp)):
            v = tmp[i]
            e[i,:len(v)]=v
        return e

if __name__ == '__main__':
    gv = GoogleVectors()

    t = ['the quick brown fox jumped over the lazy dog','the. quick, brown! fox,, !']
    x=gv.transform(t)
    print(x)
    def myfunc(x):
        return gv.model[x]
    print(np.apply_along_axis(myfunc, 1, x))