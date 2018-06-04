import os, json
import re

class PreProcessText:
	def __init__(self):
		self.dir = "../feature_groups/groups"


	"""
	Write code to:
	1. removes hashtags, or better yet could remove the # from a hashtagWord to leave
	the words in place. test both to check which affects nlu
	2. remove @something[corresponds to a person]
	3. discard retweets as well
	4. remove any unnecessary junk: http:// ->
	"""
	def formatTweet(self, tweet):
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

		return tweet

	def extractText(self, class_type, allTweets):
		fileToWrite = class_type
		dir = os.path.join("../feature_groups/tweets", fileToWrite)
		with open(dir, "w") as f:
			for tweet in allTweets:
				res = self.formatTweet(tweet['text'])
				f.write(res + '\n')
			

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
