import os, json

class PreProcessText:
	def __init__(self):
		self.dir = "../feature_groups/groups"


	"""
	Write code to:
	1. removes hashtags, or better yet could remove the # from a hashtagWord to leave
	the words in place. test both to check which affects nlu
	2. remove @something[corresponds to a person]
	3. discard retweets as well
	4. remove any unnecessary junk
	"""
	def formatTweet(self, tweet):
		#ToDo
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
