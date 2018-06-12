import os
import json
import re
from json_dump import DumpToJson

class HashTagExtractor:
	def __init__(self):
		self.dir = "../feature_groups/tweets_dev"

	def findHashTags(self, tweet):
		hashTagPattern = "(?:#)([a-zA-Z\d\?]+)"
		hashTagPattern = re.compile(hashTagPattern)
		hashTags = re.findall(hashTagPattern, tweet)
		return hashTags

	def extractHashTag(self, allTweets, file):
		hashTagMap = {}
		hashTagMap['test_type'] = file
		index = 0
		print(file)
		for tweet in allTweets:
			hashTagPerTweet = self.findHashTags(tweet)
			hashTagMap[index] = hashTagPerTweet
			index += 1

		dump = DumpToJson(hashTagMap, "../feature_groups/hashTags")
		dump.writeToJson()

	def readJson(self):
		for file in os.listdir(self.dir):
			absPath = os.path.join(self.dir, file)
			with open(absPath, "r", encoding='iso-8859-1') as f:
				self.extractHashTag(f, file)


if __name__ == '__main__':
	ht = HashTagExtractor()
	ht.readJson()
