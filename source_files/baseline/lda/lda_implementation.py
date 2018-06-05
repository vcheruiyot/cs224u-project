from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import numpy as np
import matrix_dev
import matplotlib.pyplot as plt
import _pickle
import os

class lda_model:
	def __init__(self, train, dev):
		#load pickle
		print("loading pickles")
		self.no_features = 1000
		path = os.path.join("../../feature_groups/lda_pickles", train)
		with open(path, "rb") as f:
			self.raw_docs = _pickle.load(f)
		path = os.path.join("../../feature_groups/lda_pickles", dev)
		with open(path, "rb") as f:
			self.dev = _pickle.load(f)

	def write_to_pickle(self, dir, name, obj):
		print("writing "+name+" to pickle")
		path = os.path.join(dir, name)
		path = os.path.join("../../feature_groups/lda_pickles", 'tf_vectorizer')
		with open(path, "wb") as f:
			_pickle.dump(obj, f)

	def tf_model(self):
		print("Creating tf model")
		tf_vectorizer = TfidfVectorizer(max_features=self.no_features) #stop_words='english')
		tf = tf_vectorizer.fit_transform(self.raw_docs)
		tf_feature_names = tf_vectorizer.get_feature_names()
		self.write_to_pickle("../../feature_groups/lda_pickles", 'tf_vectorizer', tf)
		return tf

	def grid_search(self):
		print("Starting grid search")
		#define search params
		search_params = {'n_components':[70, 140], 'learning_decay':[0.7, 0.9]}
		#init model
		lda = LatentDirichletAllocation()
		#init grid search class
		grid_model = GridSearchCV(cv=None, error_score='raise',
				estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
					evaluate_every=-1, learning_decay=0.7, learning_method='online',
					learning_offset=10.0, max_doc_update_iter=100, max_iter=10000,
					mean_change_tol=0.001, n_components=70, n_jobs=1,
					n_topics=None, perp_tol=0.1, random_state=None),
				fit_params=None, iid=True, n_jobs=1,
				param_grid=search_params,
				pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
				scoring=None, verbose=1)
		self.write_to_pickle("../../feature_groups/lda_pickles", 'grid_model', grid_model)
		return model

if __name__ == '__main__':
	lda = lda_model('raw_docs', 'dev')
	train_model = lda.tf_model()
	grid_model = lda.grid_search()
	grid_model.fit(train_model)









