import os, sys
import numpy as np
import json
from nltk.corpus import stopwords


class matrix_dev:
	def __init__(self, fileName):
		self.fileName = fileName
		self.docs = []
		self.vocab = set()
		self.vocab_index = {}
		dirName = "../../feature_groups/tweets_dev"
		absPath = os.path.join(dirName, fileName)
		with open(absPath, "r", encoding="utf8") as f:
			for line in f:
				self.docs.append(line)
				for word in list(line):
					self.vocab.add(word)			
		self.vocab_index = {w:i for i,w in enumerate(self.vocab)}			
		# dirName = "../../feature_groups/hashTags"
		# absFileName = self.fileName + ".json"
		# absPath = os.path.join(dirName, absFileName)
		# self.topics = []
		# with open(absPath, "r") as f:
		# 	dictObject = json.load(f)
		# try:
		# 	for key in dictObject:
		# 		if len(dictObject[key]) > 0 and dictObject[key][0] != self.fileName:
		# 			_topics = []
		# 			for i in dictObject[key]:
		# 				_topics.append(i)
		# 			self.topics.append(_topics)
						
		# except KeyError:
		# 	print("key error unexpectedly found\n")

	def raw_docs(self):
		return self.docs

	def tf_idf(self):
		"""
		Come up with tf-idf weighting mechanism as a baseline
		"""
		pass

	def get_topics(self):
		return self.topics

	def word_document_matrix(self, remove_stopwords = False):
		"""
		Returns a matrix with words as the rows and documents
		as the columns
		"""
		if remove_stopwords:
			stop_words = set(stopwords.words('english'))
		matrix = np.zeros((len(self.vocab),len(self.docs)))
		for doc_index, doc in enumerate(self.docs):
			for w in doc:
				if remove_stopwords and w in stop_words:
					continue
				word_index = self.vocab_index[w]
				matrix[word_index, doc_index] += 1
		return matrix
	



