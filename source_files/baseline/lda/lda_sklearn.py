from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
import matrix_dev



raw_docs = matrix_dev.matrix_dev('2010').raw_docs()
#vectorize (raw counts)
no_features = 1000
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(raw_docs)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics =10
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=3, 
			learning_method='online', learning_offset=50.,random_state=0)
lda.fit(tf)
X_new = lda.fit_transform(tf, )

print(tf)