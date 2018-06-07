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
from pprint import pprint
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import spacy
import pandas as pd

class lda_model:
	def __init__(self):
		self.no_features = 1000
		self.stop_words = stopwords.words('english')
		self.nlp = spacy.load('en', disable=['parser', 'ner'])

	def sent_to_words(self, sentences):
		for sentence in sentences:
			yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

	def remove_stopwords(self, sentences):
		for sentence in sentences:
			yield([word for word in sentence if word not in self.stop_words])

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
		self.docs = list(self.sent_to_words(self.raw_docs))
		self.no_stopwords = self.remove_stopwords(self.docs)
		self.data_lemmatized = self.bigram_trigram_init()
		self.id2word = corpora.Dictionary(self.data_lemmatized)	
		self.corpus = [self.id2word.doc2bow(line) for line in self.data_lemmatized]

	def bigram_trigram_init(self):
		self.bigram = gensim.models.Phrases(self.docs , min_count=5, threshold=100) # higher threshold fewer phrases.
		self.trigram = gensim.models.Phrases(self.bigram[self.docs], threshold=100)  

		# Faster way to get a sentence clubbed as a trigram/bigram
		self.bigram_mod = gensim.models.phrases.Phraser(self.bigram)
		self.trigram_mod = gensim.models.phrases.Phraser(self.trigram)
		self.data_words_bigrams = self.make_bigrams(self.no_stopwords)
		return self.lemmatization(self.data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

	def make_bigrams(self, texts):
		return [self.bigram_mod[doc] for doc in texts]

	def make_trigrams(self, texts):
		return [self.trigram_mod[bigram_mod[doc]] for doc in texts]

	def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
		texts_out = []
		for sent in texts:
			doc = self.nlp(" ".join(sent)) 
			texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
		return texts_out
	

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
                                           passes=1,
                                           alpha='auto',
                                           per_word_topics=True)
		self.write_to_pickle('../../feature_groups/lda_pickles', 'lda_model', lda_model)
		print(lda_model.print_topics())
		doc_lda = lda_model[self.corpus]
		#calculate perplexity or how good the model is
		print('perplexity: ', lda_model.log_perplexity(self.corpus))
		#compute the coherence score
		coherence_model_lda = CoherenceModel(model=lda_model, texts=self.data_lemmatized, dictionary=self.id2word, coherence='c_v')
		coherence_lda = coherence_model_lda.get_coherence()
		print('Coherence score ', coherence_lda)
		return lda

	def lda_mallet_model(self):
		self.mallet_path = 'mallet-2.0.8/bin/mallet'

		lda_mallet = gensim.models.wrappers.LdaMallet(self.mallet_path, corpus=self.corpus, num_topics=70, id2word=self.id2word)
		#show topics
		print(lda_mallet.show_topics(formatted=False))
		#compute the coherence
		coherence_model_ldamallet = CoherenceModel(model=lda_mallet, texts=self.data_lemmatized, dictionary=self.id2word, coherence='c_v')
		coherence_ldamallet = coherence_model_ldamallet.get_coherence()
		print('Coherence Score: ', coherence_ldamallet)


	def best_model_search(self):
		coherence_values, lda_mallet_list = self.compute_best_lda_model(70, 400, 30)
	"""
	finding the best lda model based on the n_topics chosen
	"""
	def compute_best_lda_model(self, start, limit, step):
		"""
		coherence_values : coherence values corresponding to the number of topics
		lda_model_list : list of lda topic model
		"""
		coherence_values = []
		lda_mallet_list = []

		for topics in range(start, limit, step):
			model = gensim.models.wrappers.LdaMallet(self.mallet_path, corpus=self.corpus, num_topics=topics, id2word=self.id2word)
			lda_mallet_list.append(model)
			coherence_model = CoherenceModel(model=model, texts=self.data_lemmatized, dictionary=self.id2word, coherence='c_v')
			coherence_values.append(coherence_model.get_coherence())
			print('Num_topics = ', topics, ' corresponding coherence value: ', coherence_model.get_coherence())

		return coherence_values, lda_mallet_list

	# Group top 5 sentences under each topic
	def get_representative_docs():
		self.sent_topics_sorteddf_mallet = pd.DataFrame()
		self.sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
		for i, grp in sent_topics_outdf_grpd:
		   self.sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
		                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
		                                            axis=0)
		# Reset Index    
		self.sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
		# Format
		self.sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
		# Show
		pprint(self.sent_topics_sorteddf_mallet.head())

	def topic_distribution():
		# Number of Documents for Each Topic
		self.topic_counts = self.df_topic_sents_keywords['Dominant_Topic'].value_counts()

		# Percentage of Documents for Each Topic
		self.topic_contribution = round(topic_counts/topic_counts.sum(), 4)

		# Topic Number and Keywords
		self.topic_num_keywords = self.df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

		# Concatenate Column wise
		self.df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

		# Change Column names
		self.df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

		# Show
		pprint(self.df_dominant_topics)





if __name__ == '__main__':
	lda = lda_model()
	lda.load_data('raw_docs', 'dev')
	#ld = lda.lda()
	lda.lda_mallet_model()
	lda.best_model_search()
	#train_model = lda.tf_model()
	#train_model = lda.load_pickle('../../feature_groups/lda_pickles', 'tf_vectorizer')
	
	#ld.fit(train_model)
	#lda.write_to_pickle("../../feature_groups/lda_pickles", 'lda_model', lda_model)
	#grid_model.fit(train_model)
	#grid_model.write_to_pickle("../../feature_groups/lda_pickles", 'grid_model', grid_model)









