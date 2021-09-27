import gensim
from sklearn.model_selection import train_test_split
from numpy import array, append, zeros, reshape
import tensorflow as tf
# from tensorflow.contrib import rnn
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

vocab = model.vocab.keys()
wordsInVocab = len(vocab)
print 'Number of Words in Google Model:', wordsInVocab

##############################
# Part 1, Sentences Embeddings

