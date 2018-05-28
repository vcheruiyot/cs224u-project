
from collections import Counter
from rnn_classifier import RNNClassifier
from sklearn.linear_model import LogisticRegression
import sst
import tensorflow as tf
from tf_rnn_classifier import TfRNNClassifier
from tree_nn import TreeNN
import sys
import numpy as np
import re, math
import pandas as pd

#Regular Expression needed to Compute the Cosine Similarity:
WORD = re.compile(r'\w+')


#COSINE SIMILARITY IMPLEMENTATION

#----------------------------------------------------------------------------#
#Note this last part will make sure to compute the cosine similarity
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)    

def computeContentSimilarity(tweet1, tweet2):
    #for the most basic model we are going to compute cosine similarity
    text1 = tweet1
    text2 = tweet2
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)
    cosine = get_cosine(vector1, vector2)
    return cosine


#----------------------------------------------------------------------------#
#IMPORTING TWITTER HASHTAG DATA

#Note: These are all tweets listed with the hashtag #immigration
#The Goal is to use the Tree Structure to Create and Develop This Code
text_file = open("immigration")
lines = text_file.read().split('\n')

#----------------------------------------------------------------------------#
#Baseline Accuracy of Model

def unigrams_phi(tree):
    """The basis for a unigrams feature function.
    
    Parameters
    ----------
    tree : nltk.tree
        The tree to represent.
    
    Returns
    -------    
    defaultdict
        A map from strings to their counts in `tree`. (Counter maps a 
        list to a dict of counts of the elements in that list.)
    
    """
    return Counter(tree.leaves())

def fit_maxent_classifier(X, y):
    #initialize an empty X for this model:
    X = np.zeros(shape = (len(lines)-244980, len(lines)-244980))
    for i in range(0, len(lines)-244980):
        print ("Working on the " + str(i), " result")
        for j in range(0, len(lines)-244980):
            if i == j:
                X[i][j] = 100000 #discounted cosine similarity value
            else:
                X[i][j] = computeContentSimilarity(lines[i], lines[j])
    
    print (X)
    #the goal is to build an empty numpy d array based on 

    mod = LogisticRegression(fit_intercept=True)
    mod.fit(X, y)
    return mod

_ = sst.experiment(
    unigrams_phi,                     
    fit_maxent_classifier,           
    train_reader=sst.train_reader,     
    assess_reader=sst.dev_reader,      
    class_func=sst.binary_class_func)  

#----------------------------------------------------------------------------#
#TfRNNClassifier wrapper

def rnn_phi(tree):
    return tree.leaves()  

def fit_tf_rnn_classifier(X, y):
    vocab = sst.get_vocab(X, n_words=3000)
    mod = TfRNNClassifier(
        vocab, 
        eta=0.05,
        batch_size=2048,
        embed_dim=50,
        hidden_dim=50,
        max_length=52, 
        max_iter=500,
        cell_class=tf.nn.rnn_cell.LSTMCell,
        hidden_activation=tf.nn.tanh,
        train_embedding=True)
    mod.fit(X, y)
    return mod

_ = sst.experiment(
    rnn_phi,
    fit_tf_rnn_classifier, 
    vectorize=False,  # For deep learning, use `vectorize=False`.
    assess_reader=sst.dev_reader)

#----------------------------------------------------------------------------#
#TREENN WRAPPER FOR SST CONSTRUCTION

def tree_phi(tree):
    return tree

def fit_tree_nn_classifier(X, y):
    vocab = sst.get_vocab(X, n_words=3000)
    mod = TreeNN(
        vocab, 
        embed_dim=100, 
        max_iter=100)
    mod.fit(X, y)
    return mod

_ = sst.experiment(
    rnn_phi,
    fit_tree_nn_classifier, 
    vectorize=False,  # For deep learning, use `vectorize=False`.
    assess_reader=sst.dev_reader)

#----------------------------------------------------------------------------#
#As the last step we are going to create clusters based on cosine and SST benchmark

top_10 = [0,0,0,0,0,0,0,0,0,0]
top_10_index = [None, None, None, None, None, None, None, None, None, None]

#I might need this later down the line to see if level of accuracy increases

if content_similarity > top_10[0]:
    top_10[0]=content_similarity
    top_10_index[0] = index_2
elif content_similarity > top_10[1]:
    top_10[1] = content_similarity
    top_10_index[1] = index_2
elif content_similarity > top_10[2]:
    top_10[2] = content_similarity
    top_10_index[2] = index_2
elif content_similarity > top_10[3]:
    top_10[3] = content_similarity
    top_10_index[3] = index_2
elif content_similarity > top_10[4]:
    top_10[4] = content_similarity
    top_10_index[4] = index_2
elif content_similarity > top_10[5]:
    top_10[5] = content_similarity
    top_10_index[5] = index_2
elif content_similarity > top_10[6]:
    top_10[6] = content_similarity
    top_10_index[6] = index_2
elif content_similarity > top_10[7]:
    top_10[7] = content_similarity
    top_10_index[7] = index_2
elif content_similarity > top_10[8]:
    top_10[8] = content_similarity
    top_10_index[8] = index_2
elif content_similarity > top_10[9]:
    top_10[9] = content_similarity
    top_10_index[9] = index_2

single_array.append(content_similarity)

