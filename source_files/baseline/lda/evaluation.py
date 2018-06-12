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
			self.test = json.loads(f)

	def set_model(self, model):
		self.model = model

	def load_test_x(self, name, path = '../../feature_groups/tweets_dev'):
		with open(abs_path, "r") as f:
			self.lines = f.readlines()


	def load_pickle(self, dir, name):
		print("loading pickle: "+name)
		path = os.path.join(dir, name)
		with open(path, "rb") as f:
			return _pickle.load(f)

	def process_line(self, line): #stub function
		return set()

	def calculate_recall(self):
		count = 0.
		hits = 0.
		for index, line in enumerate(self.lines):
			sugested_tags = self.process_line(line)
			actual_tags = self.test[index]
			for elem in actual_tags:
				if elem in sugested_tags:
					hits += 1
				count +=1
		return hits/count


