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
		with open(absPath, "r") as f:
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




	
#	def vertical_topics(self, width, topic_index, document_length):
#		"""
#		Generate a topic whose words form a horizontal bar
#		"""
#		m = np.zeros((width, width))
#		m[:, topic_index] = int(document_length/ width)
#		return m.flatten()
#
#	def horizontal_topics(self, width, topic_index, d_length):
#		m = np.zeros((width, width))
#		m[topic_index, :] = int(document_length/ width)
#		return m.flatten()
#
#	def get_word_distribution(self, document_length, index):
#		width = self.n_topics/2
#		print(self.n_topics)
#		print(self.vocab_size[index])
#		m = np.zeros((self.n_topics, self.vocab_size[index]))
#
#		for k in range(int(width)):
#			m[k, :] = self.vertical_topics(int(width), k, document_length)
#
#		for k in range(int(width)):
#			m[k + width, :] = horizontal_topics(int(width), k, document_length)	
#
#		m /= m.sum(axis=1)[:, np.newaxis]
#		return m
#
#	def word_distribution(self):
#		for i in range(0, len(self.docs)):
#			return self.get_word_distribution(self.docs[i], i)
	






