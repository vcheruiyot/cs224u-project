import os, json
import re


class PreProcessText:
	def __init__(self):
		self.dir = "../feature_groups/tweets_dev"


	"""
	Write code to:
	1. removes hashtags, or better yet could remove the # from a hashtagWord to leave
	the words in place. test both to check which affects nlu
	2. remove @something[corresponds to a person]
	3. discard retweets as well
	4. remove any unnecessary junk: http:// ->
	"""
	def formatTweet(self, tweet):
		tweet = tweet.strip()
		"""
		remove extra spacing in the tweet
		"""
		spaces = "(\ +)"
		spaces = re.compile(spaces)
		tweet = re.sub(spaces, ' ', tweet)
		"""
		lowercase everything
		"""
		tweet = tweet.lower()
		"""
		removes hashes from a given word
		"""
		remove_hash = "#"
		remove_hash = re.compile(remove_hash)
		tweet = re.sub(remove_hash, '', tweet)
		"""
		remove @name used for follower choice
		"""
		non_follower = '@(\w+[\. : \? ~ ! \ \$ \% \ *& \( \) - \+ \=]?)?'
		non_follower = re.compile(non_follower)
		tweet = re.sub(non_follower, '', tweet)
		"""
		remove url from text
		"""
		url_ex = "https?:\/\/.*[\r\n]*"
		url_ex = re.compile(url_ex)
		tweet = re.sub(url_ex, '', tweet, re.MULTILINE)

		"""
		remove punctuation marks
		"""
		punct = "[\.\?\&\-\!\[\]\:\;\(\)\/\|]"
		punct = re.compile(punct)
		tweet = re.sub(punct, '', tweet)
		"""
		retweets removed
		"""
		rt = "^(rt)+[^A-Za-z0-9](rt[^A-Za-z0-9])*"
		rt = re.compile(rt)
		tweet = re.sub(rt, '', tweet, re.IGNORECASE)
		
		rt2 = "[^A-Za-z0-9](rt)+[^A-Za-z0-9](rt[^A-Za-z0-9])*"
		rt2 = re.compile(rt2)
		tweet = re.sub(rt2, ' ', tweet, re.IGNORECASE)

		"""
		because -> coz, bcoz, 
		"""
		coz = "((b?e?coz)[^A-Za-z0-9])|([^A-Za-z0-9](b?e?coz)[^A-Za-z0-9])"
		coz = re.compile(coz)
		tweet = re.sub(coz, ' because ', tweet, re.IGNORECASE)

		tweet = tweet.strip()

		return tweet

	def extractText(self, f, name):
		print(name)
		absPath = os.path.join(self.dir, name + '_clean')
		with open(absPath, 'w') as to_write:
			for tweet in f:
				res = self.formatTweet(tweet)
				to_write.write(res + '\n')
			

	def readDir(self):
		for file in os.listdir(self.dir):
			absPath = os.path.join(self.dir, file)
			with open(absPath, "r", encoding='iso-8859-1') as f:
				self.extractText(f, file)	


if __name__ == '__main__':
	gt = PreProcessText()
	gt.readDir()
