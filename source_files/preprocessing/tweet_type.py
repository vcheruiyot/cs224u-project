#!/usr/bin/python3

class tweet:
	def __init__(self, tweet_id, text, from_usr_id, to_usr_id):
		self.tweet_map = {}
		self.tweet_map['tweet_id'] = tweet_id
		self.tweet_map['text'] = text
		self.tweet_map['from_usr_id'] = from_usr_id
		self.tweet_map['to_usr_id'] = to_usr_id
		
	def get_tweet_map(self):
		return self.tweet_map
		