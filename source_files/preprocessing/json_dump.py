import json
import os

"""
Remember class_type? check tweets.py
"""
class DumpToJson:
	def __init__(self, dictObject, dirName):
		self.dictObject = dictObject
		self.dir = dirName
		
	def writeToJson(self):
		featureGroup = self.dictObject['class_type']
		jsonFile = featureGroup + ".json"
		absPath = os.path.join(self.dir, jsonFile)
		with open(absPath, "w") as outFile:
			try:
				json.dump(self.dictObject, outFile)
			except ValueError:
				print("dictObject not well formatted")