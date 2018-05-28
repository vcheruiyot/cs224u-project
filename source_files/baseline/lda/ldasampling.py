"""
An implementation of a basic lda gibbs sampling
"""

import numpy as np 
import scipy as sp 
from scipy.special import gammaln

class lda_sampler:
	def(self, nTopics, alpha=0.1, beta=0.1):
		"""
		nTopics -> desired number of topics
		alpha -> scalar
		beta -> scalar
		"""
		self.nTopics = nTopics
		self.alpha = alpha
		self.beta = beta


	def get_indices(self, vec):
		"""
		Turns a document vector of size vocab size to a
		sequence of word indices. 
		"""
		for i in vec.nonzero()[0]:
			for i in xrange(int(vec[i])):
				yield i


	def initialSampling(self, M):
		n_docs, n_vocabSize = M.shape()
		#number of times a document m and topic z co-occur
		self.ntdz = np.zeros((n_docs, self.nTopics))
		#number of times topic z and word w co-occur
		self.ntzw = np.zeros((self.nTopics, n_vocabSize))
		#number of n documets
		self.nd = np.zeros(n_docs)
		#number of topics
		self.nz = np.zeros(self.nTopics)
		self.topics = {}

		for m in xrange(n_docs):
			# i is a number of times document m and topic z occur
			# w is a number of 0 - vocab_size - 1
			for i, w in enumerate(self.get_indices(matrix[m, :]))
				#choose an arbitrary topic as first topic for word i
				z = np.random.randint(self.nTopics)
				self.ntdz[m, z] += 1
				self.nd[m] += 1
				self.ntzw[z, w] += 1
				self.nz[z] += 1
				self.topics[(m, i)] = z

	def conditional_dist(self, m, w):
		"""
		coniditional distribution of n_topics
		"""
		vocab_size = self.ntzw.shape[1]
		left = (self.ntzw[:, w] + self.beta) /(self.nz + self.beta * vocab_size)
		right = (self.ntdz[m,:] + self.alpha)/(self.nd[m] + self.alpha * self.nTopics)

		p_z = left * right
		# normalize
		p_z /= np.sum(p_z)
		return p_z

	def start(self, matrix, iter=30):
		"""
		start the gibbs sampler
		"""
		n_docs, vocab_size = matrix.shape
		self.initialSampling(matrix)

		for it in xrange(iter):
			for m in xrange(n_docs):
				for i, w in enumerate(get_indices(matrix[m. :])):
					z = self.topics[(m, i)]
					self.ntdz[m, z] -= 1
					self.nd[m] -= 1
					self.ntzw[z, w] -= 1
					self.nz[z] -= 1

					p_z = self.coniditional	
