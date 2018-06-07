from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import numpy as np
import matrix_dev
import matplotlib.pyplot as plt
import _pickle
import gensim
import gensim.corpora as corpora
import os
import pprint
from gensim.models import CoherenceModel

class lda_model:
	def __init__(self):
		self.no_features = 1000

	def load_data(self, train, dev):
		#load pickle
		print("loading pickles")

		# path = os.path.join("../../feature_groups/lda_pickles", train)
		# with open(path, "rb") as f:
		# 	self.raw_docs = _pickle.load(f)
		# path = os.path.join("../../feature_groups/lda_pickles", dev)
		# with open(path, "rb") as f:
		# 	self.dev = _pickle.load(f)


		self.raw_docs = matrix_dev.matrix_dev('trainshort').raw_docs()
		path = os.path.join("../../feature_groups/lda_pickles", 'raw_short')
		with open(path, "wb") as f:
			_pickle.dump(self.raw_docs, f)
	    # path = os.path.join("../../feature_groups/lda_pickles", 'dev')
	    # with open(path, "wb") as f:
	    #     _pickle.dump(dev, f)

		self.data_lemmatized = [i.split() for i in self.raw_docs]
		self.id2word = corpora.Dictionary(self.data_lemmatized)	
		self.corpus = [self.id2word.doc2bow(line) for line in self.data_lemmatized]
	

	def write_to_pickle(self, dir, name, obj):
		print("writing "+name+" to pickle")
		path = os.path.join(dir, name)
		with open(path, "wb") as f:
			_pickle.dump(obj, f)

	def load_pickle(self, dir, name):
		print("loading pickle: "+name)
		path = os.path.join(dir, name)
		with open(path, "rb") as f:
			return _pickle.load(f)

	def load_tfidf(self):
		return lda.load_pickle('../../feature_groups/lda_pickles', 'tf_vectorizer')	


	def tf_model(self):
		print("Creating tf model")
		tf_vectorizer = TfidfVectorizer(max_features=self.no_features) #stop_words='english')
		tf = tf_vectorizer.fit_transform(self.raw_docs)
		tf_feature_names = tf_vectorizer.get_feature_names()
		#self.write_to_pickle("../../feature_groups/lda_pickles", 'tf_vectorizer', tf)
		return tf

	def lda(self):
		print("Starting lda")
		#corpus = self.tf_model().todense()
		lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                           id2word=self.id2word,
                                           num_topics=70, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=30,
                                           alpha='auto',
                                           per_word_topics=True)
		pprint(lda_model.print_topics())
		doc_lda = lda_model[corpus]
		#calculate perplexity or how good the model is
		print('perplexity: ', lda_model.log_perplexity(corpus))
		#compute the coherence score
		coherence_model_lda = CoherenceModel(model=lda_model, texts=self.data_lemmatized, dictionary=self.id2word, coherence='c_v')
		coherence_lda = coherence_model_ld.get_coherence()
		print('Coherence score ', coherence_lda)

		#define search params
		"""
		search_params = {'n_components':[70, 140], 'learning_decay':[0.7, 0.9]}
		#init model
		
		lda = LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
					evaluate_every=-1, n_components=70, learning_decay=0.7, learning_method='online',
					learning_offset=10.0, max_doc_update_iter=100, max_iter=50,
					mean_change_tol=0.001, n_jobs=1,
					n_topics=None, perp_tol=0.1, random_state=None)
		#init grid search class
		"""
		"""
		grid_model = GridSearchCV(cv=None, error_score='raise',
				estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
					evaluate_every=-1, learning_decay=0.7, learning_method='online',
					learning_offset=10.0, max_doc_update_iter=100, max_iter=10000,
					mean_change_tol=0.001, n_jobs=1,
					n_topics=None, perp_tol=0.1, random_state=None),
				fit_params=None, iid=True, n_jobs=1,
				param_grid=search_params,
				pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
				scoring=None, verbose=1)
		"""
		return lda

if __name__ == '__main__':
	lda = lda_model()
	lda.load_data('raw_docs', 'dev')
	ld = lda.lda()
	#train_model = lda.tf_model()
	#train_model = lda.load_pickle('../../feature_groups/lda_pickles', 'tf_vectorizer')
	
	#ld.fit(train_model)
	#lda.write_to_pickle("../../feature_groups/lda_pickles", 'lda_model', lda_model)
	#grid_model.fit(train_model)
	#grid_model.write_to_pickle("../../feature_groups/lda_pickles", 'grid_model', grid_model)









