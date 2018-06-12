import os, json
import re
from ekphrasis.classes.segmenter import Segmenter

class PreProcessText:
	def __init__(self):
		self.dir = "../feature_groups/groups"




	def ensure_segmentation(self, words):
		print(words)
		print("Performing Segmentation")
		seg_tw = Segmenter(corpus="twitter")
		sentence = []
		for word in words:
			res = seg_tw.segment(word)
			print(res)
			sentence.append(res)
	

		return " ".join(sentence)

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

	def extractText(self, class_type, allTweets):
		fileToWrite = class_type
		file = os.path.join("../feature_groups/tweets", fileToWrite)
		with open(file, "w") as f:
			for tweet in allTweets:
				#res = self.formatTweet(tweet['text'])
				#split_tweet = res.split(" ")
				#formatted_tweet = self.ensure_segmentation(split_tweet)
				f.write(tweet + '\n')
			

	def readJson(self):
		for file in os.listdir(self.dir):
			absPath = os.path.join(self.dir, file)
			with open(absPath, "r") as jsonFile:
				jsonObject = json.load(jsonFile)
				class_type = jsonObject['class_type']
				self.extractText(class_type, jsonObject[class_type])	


if __name__ == '__main__':
	gt = PreProcessText()
	gt.readJson()
