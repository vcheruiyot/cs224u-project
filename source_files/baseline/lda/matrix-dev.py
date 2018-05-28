import os, sys
import numpy as np
import json


class GenereteDistribution:
	def __init__(self, fileName):
		self.fileName = fileName
		self.docs = []
		self.vocab = set()
		dirName = "../feature_groups/tweets"
		absPath = os.path.join(dirName, fileName)
		with open(absPath, "r") as f:
			for line in f:
				self.docs.append(line)
				for word in list(line):
					self.vocab.add(word)

		dirName = "../feature_groups/hashTags"
		absFileName = self.fileName + ".json"
		absPath = os.path.join(dirName, absFileName)
		self.topics = []
		with open(absPath, "r") as f:
			dictObject = json.load(f)
		try:
			for key in dictObject:
				if len(dictObject[key]) > 0 and dictObject[key][0] != self.fileName:
					_topics = []
					for i in dictObject[key]:
						_topics.append(i)
					self.topics.append(_topics)
						
		except KeyError:
			print("key error unexpectedly found\n")

	def tf_idf(self):
		"""
		Come up with tf-idf weighting mechanism as a baseline
		"""
		pass

	def topicsToAssign(self, index):
		return self.topics[index]

		

	"""
	def vertical_topics(self, width, topic_index, document_length):
		"""
		Generate a topic whose words form a horizontal bar
		"""
		m = np.zeros((width, width))
		m[:, topic_index] = int(document_length/ width)
		return m.flatten()

	def horizontal_topics(self, width, topic_index, d_length):
		m = np.zeros((width, width))
		m[topic_index, :] = int(document_length/ width)
		return m.flatten()

	def get_word_distribution(self, document_length, index):
		width = self.n_topics/2
		print(self.n_topics)
		print(self.vocab_size[index])
		m = np.zeros((self.n_topics, self.vocab_size[index]))

		for k in range(int(width)):
			m[k, :] = self.vertical_topics(int(width), k, document_length)

		for k in range(int(width)):
			m[k + width, :] = horizontal_topics(int(width), k, document_length)	

		m /= m.sum(axis=1)[:, np.newaxis]
		return m

	def word_distribution(self):
		for i in range(0, len(self.docs)):
			return self.get_word_distribution(self.docs[i], i)
	
	"""
if __name__ == '__main__':
	gdist = GenereteDistribution("2010")
	print(gdist.word_distribution())





