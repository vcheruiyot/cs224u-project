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
		self.mallet_path = 'mallet-2.0.8/bin/mallet'

	def sent_to_words(self, sentences):
		for sentence in sentences:
			yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

	def remove_stopwords(self, sentences):
		for sentence in sentences:
			yield([word for word in sentence if word not in self.stop_words])

	def load_data(self, train = 'raw_docs.pkl'):
		#load pickle
		print("loading data")

		# path = os.path.join("../../feature_groups/lda_pickles", train)
		# with open(path, "rb") as f:
		# 	self.raw_docs = _pickle.load(f)
		# path = os.path.join("../../feature_groups/lda_pickles", dev)
		# with open(path, "rb") as f:
		# 	self.dev = _pickle.load(f)
		
		path = os.path.join('../../feature_groups/lda_pickles', train)
		if os.path.isfile(path):
			self.raw_docs = self.load_pickle('../../feature_groups/lda_pickles', train)
		else:
			self.raw_docs = matrix_dev.matrix_dev('../../feature_groups/tweets_dev/supertrain_clean').raw_docs()
			#self.write_to_pickle('../../feature_groups/lda_pickles', train, self.raw_docs)
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

	# def lda(self):
	# 	print("Starting lda")
	# 	#corpus = self.tf_model().todense()
	# 	lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
 #                                           id2word=self.id2word,
 #                                           num_topics=70, 
 #                                           random_state=100,
 #                                           update_every=1,
 #                                           chunksize=100,
 #                                           passes=1,
 #                                           alpha='auto',
 #                                           per_word_topics=True)
	# 	self.write_to_pickle('../../feature_groups/lda_pickles', 'lda_model', lda_model)
	# 	#print(lda_model.print_topics())
	# 	doc_lda = lda_model[self.corpus]
	# 	#calculate perplexity or how good the model is
	# 	print('perplexity: ', lda_model.log_perplexity(self.corpus))
	# 	#compute the coherence score
	# 	coherence_model_lda = CoherenceModel(model=lda_model, texts=self.data_lemmatized, dictionary=self.id2word, coherence='c_v')
	# 	coherence_lda = coherence_model_lda.get_coherence()
	# 	print('Coherence score ', coherence_lda)
	# 	return lda

	def domininant_topic(self, lda_model, corpus = None):
		if not corpus:
			corpus = self.corpus
		sent_topics = pd.DataFrame()
		#get main topic in each document
		for i, row in enumerate(lda_model[corpus]):
			row = sorted(row, key=lambda x : (x[1]), reverse=True)
			# Get the Dominant topic, Perc Contribution and Keywords for each document
			for j, (topic_num, perc_prop) in enumerate(row):
				if j == 0 : # -> dominant topic
					found = lda_model.show_topic(topic_num)
					keywords = ", ".join([word for word, prop in found])
					sent_topics = sent_topics.append(pd.Series([int(topic_num), round(perc_prop,4), keywords]), ignore_index=True)

				else:
					break
		sent_topics.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']	
		#add original text
		contents = pd.Series(self.raw_docs)
		sent_topics = pd.concat([sent_topics, contents], axis=1)
		return sent_topics
			


	def lda_mallet_model(self, num_topics=70, iterations = 1):
		"""
		builds the lda model. 
		"""
		lda_mallet = gensim.models.wrappers.LdaMallet(self.mallet_path, corpus=self.corpus, num_topics=num_topics, id2word=self.id2word, iterations=iterations)
		coherence_model_ldamallet = CoherenceModel(model=lda_mallet, texts=self.data_lemmatized, dictionary=self.id2word, coherence='c_v')
		coherence_ldamallet = coherence_model_ldamallet.get_coherence()
		print('Coherence Score: ', coherence_ldamallet)
		return lda_mallet


	def best_model_search(self, start = 5, end = 100, step = 20):
		coherence_values, lda_mallet_list, lda_topics = self.compute_best_lda_model(start, end, step)
		max_tup = max(zip(coherence_values, lda_mallet_list, lda_topics))
		#view_topics = self.domininant_topic(lda_model=max_tup[1])
		#topic_doc_dist = self.get_representative_docs(lda_model=max_tup[1])
		#doc_topic_dist = self.topic_distribution(lda_model=max_tup[1])
		#pprint(view_topics)
		return max_tup[1], coherence_values

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
		lda_topics = []

		for topics in range(start, limit, step):
			model = gensim.models.wrappers.LdaMallet(self.mallet_path, corpus=self.corpus, num_topics=topics, id2word=self.id2word, iterations=50)#2000)
			lda_mallet_list.append(model)
			coherence_model = CoherenceModel(model=model, texts=self.data_lemmatized, dictionary=self.id2word, coherence='c_v')
			coherence_values.append(coherence_model.get_coherence())
			lda_topics.append(topics)
			print('Num_topics = ', topics, ' corresponding coherence value: ', coherence_model.get_coherence())

		return coherence_values, lda_mallet_list, lda_topics

	# Group top 5 sentences under each topic
	def get_representative_docs(self, lda_model):
		#print("representative docs")
		topic_sents_keywords = self.domininant_topic(lda_model=lda_model)
		
		sent_topics = pd.DataFrame()
		sent_topics_group = topic_sents_keywords.groupby('Dominant_Topic')
		for i, grp in sent_topics_group:
		   sent_topics = pd.concat([sent_topics, 
		                                        grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
		                                        axis=0)
		# Reset Index    
		sent_topics.reset_index(drop=True, inplace=True)
		# Format
		sent_topics.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
		# Show
		#pprint(sent_topics.head())
		return sent_topics

	def topic_distribution(self, lda_model):
		print("topic distribution")
		topic_sents_keywords = self.domininant_topic(lda_model=lda_model)
		# Number of Documents for Each Topic
		topic_counts = topic_sents_keywords['Dominant_Topic'].value_counts()

		# Percentage of Documents for Each Topic
		topic_contribution = round(topic_counts/topic_counts.sum(), 4)

		# Topic Number and Keywords
		topic_num_keywords = topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

		# Concatenate Column wise
		df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

		# Change Column names
		df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

		# Show
		#pprint(df_dominant_topics)
		return df_dominant_topics


if __name__ == '__main__':
	lda = lda_model()
	lda.load_data('raw_docs', 'dev')
	lda.best_model_search()
	









