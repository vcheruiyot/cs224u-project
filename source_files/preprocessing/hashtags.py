import os
import json
import re
from json_dump import DumpToJson

class HashTagExtractor:
	def __init__(self):
		self.dir = "../feature_groups/groups"

	def findHashTags(self, tweet):
		hashTagPattern = "(?:#)([a-zA-Z\d\?]+)"
		hashTagPattern = re.compile(hashTagPattern)
		hashTags = re.findall(hashTagPattern, tweet)
		return hashTags

	def extractHashTag(self, class_type, allTweets):
		hashTagMap = {}
		hashTagMap['class_type'] = class_type
		index = 0
		print(class_type)
		for tweet in allTweets:
			hashTagPerTweet = self.findHashTags(tweet['text'])
			hashTagMap[index] = hashTagPerTweet
			index += 1

		dump = DumpToJson(hashTagMap, "../feature_groups/hashTags")
		dump.writeToJson()

	def readJson(self):
		for file in os.listdir(self.dir):
			absPath = os.path.join(self.dir, file)
			with open(absPath, "r") as jsonFile:
				jsonObject = json.load(jsonFile)
				class_type = jsonObject['class_type']
				self.extractHashTag(class_type, jsonObject[class_type])


if __name__ == '__main__':
	ht = HashTagExtractor()
	ht.readJson()
