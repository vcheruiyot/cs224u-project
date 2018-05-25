import csv
import os, sys
import re
import codecs
from tweet_type import tweet
from json_dump import DumpToJson

class tweets:
	def __init__(self, csv_file):
		self.tweets = []
		self.csv_file = csv_file
	
	
	def handleNullBytes(self, f):
		for line in f:
			yield line.replace('\0', '')
		
	def generate_map(self):
		tweet_map = {}
		class_type = self.csv_file[0].replace('.csv', '')
		tweet_map['class_type'] = class_type
		print(class_type)
		tweet_map[class_type] = [] 
		csvReader = csv.DictReader(self.handleNullBytes(codecs.open(self.csv_file[1], 'rU', encoding='iso-8859-1')))
		for row in csvReader:	
			#not handling non-english tweets
			if row['iso_language_code'] != 'en':
				continue
			tw = tweet(row['id'], row['text'], row['from_user_id'], row['to_user_id'])
			cur_tweets = tweet_map[class_type]
			cur_tweets.append(tw.get_tweet_map())
			tweet_map[class_type] = cur_tweets
			
		dump = DumpToJson(tweet_map, "../feature_groups")	
		dump.writeToJson()
				
				
				
def iterate_dir(dir):
	csv_files = []
	for file in os.listdir(dir):
		if re.search('\.csv', file) == None:
			continue
		abs_path = 	os.path.join(dir, file)
		res = [file, abs_path]
		csv_files.append(res)	
	return csv_files
	
if __name__ == "__main__":
	dir_name = '../../data/unzip/export/'
	dir = os.path.dirname(dir_name)
	if os.path.exists(dir):
		csv_files = iterate_dir(dir)
	else:
		printf("You are missing the required directory")
	for vr in csv_files:
		tw = tweets(vr)
		tw.generate_map()
		
		
		
		