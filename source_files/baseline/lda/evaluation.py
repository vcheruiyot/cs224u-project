import numpy as np
import matrix_dev
import matplotlib.pyplot as plt
import _pickle
import gensim
import gensim.corpora as corpora
import os
from pprint import pprint
from gensim.models import CoherenceModel
from gensim.corpora import MalletCorpus
from nltk.corpus import stopwords
import spacy
import pandas as pd
import json

class evaluator:
	def __init__(self):
		pass

	def load_test_y(self, name, path = '../../feature_groups/hashTags'):
		abs_path = os.path.join(path, name)
		with open(abs_path, "r") as f:
			self.test = json.load(f)

	def set_model(self, model):
		self.model = model

	def load_test_x(self, name, path = '../../feature_groups/tweets_dev'):
		abs_path = os.path.join(path, name)
		with open(abs_path, "r") as f:
			self.tweets = f.readlines()


	def load_model_pickle(self, name, path ='../../feature_groups/lda_pickles'):
		print("loading model: "+name)
		path = os.path.join(path, name)
		with open(path, "rb") as f:
			self.model = _pickle.load(f)

	def load_bigrams(self, name, path ='../../feature_groups/lda_pickles'):
		print("loading bigram: "+name)
		path = os.path.join(path, name)
		with open(path, "rb") as f:
			self.bigrams = _pickle.load(f)

	def get_hashtags(self, tweet): #stub function
		split = gensim.utils.simple_preprocess(str(tweet), deacc=True)
		bigram = self.bigrams[split]
		bow = self.model.id2word.doc2bow(bigram)
		topics = self.model[bow]
		best_topic_id = max(topics, key= lambda x: x[1])[0]
		keywords = self.model.show_topic(best_topic_id)
		return set(i[0] for i in keywords[:5])



	def calculate_recall(self):
		count = 0.
		hits = 0.
		for index, tweet in enumerate(self.tweets):
			sugested_tags = self.get_hashtags(tweet)
			print(index, sugested_tags)
			actual_tags = self.test[str(index)]
			for elem in actual_tags:
				if elem in sugested_tags:
					hits += 1
				count +=1
		return hits/count


