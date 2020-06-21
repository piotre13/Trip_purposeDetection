import pandas as pd
import os
import pyrebase
from collections import OrderedDict
from collections import Counter
import numpy as np
import json
# https://github.com/thisbejim/Pyrebase 

class Firebase():
	def __init__(self):
		self.config =  {
			"apiKey": "AIzaSyDo-hHs9Y2owGaOQTZue_pDMnAtn9LZ3Ag", 
			"authDomain": "trip-purpose-project.firebaseapp.com", 
			"databaseURL": "https://trip-purpose-project.firebaseio.com",
			"projectId": "trip-purpose-project", 
			"storageBucket": "trip-purpose-project.appspot.com"}
		self.firebase = pyrebase.initialize_app(self.config)

	def authenticate(self):
		"""
			Authentication: allows to access the database
		"""
		self.auth = self.firebase.auth()
		self.user = self.auth.sign_in_with_email_and_password('pino@gmail.com', 'gianni')
		self.db = self.firebase.database()

	def download(self, field):
		"""
			input : 
				'field' = string : should take only 3 values
					'territorial' -> 'key' (string) is 'census_zone'
					'user' -> key (string) is 'user_id'
					''
			output :
				'data_list' = list : list of all the elements in a given field
		"""
		all_data = self.db.child(field).get()
		data_list = [elem.val() for elem in all_data.each()]
		key_list = [elem.key() for elem in all_data.each()]

		return data_list, key_list

	def upload(self, field, data2upload, key=None):
		"""
			input : 
				'field' = string : should take only 3 values
					'territorial' -> 'key' (string) is 'census_zone'
					'user' -> key (string) is 'user_id'
					''
				'data2upload' = json : file to upload on the DB
		"""
		# self.db.child(field).child(key).update(data2upload)
		# print(obj)
		if field == 'trips':
			self.db.child(field).push(data2upload)
		else:
			self.db.child(field).child(key).set(data2upload)

	def csv_to_json(self, csv_file, path=os.getcwd()):
		"""
			input : 
				'csv_file' = string : file name and path
				'path' = os : directory of the folder in which the 
					file is contained
			output : 
				'json_file' = list : list of dictionaries  
				'columns' = list : list of names of the columns
		"""
		df = pd.read_csv(path+'\\'+csv_file)
		columns = df.columns
		json_file  = [] 
		for index, row in df.iterrows():
			tmp = {}
			for x in columns:
				tmp.update({x : row[x]})
			json_file.append(tmp)
		return json_file, columns

	def most_frequent(self, List, n):
		occurence_count = Counter(List)
		return occurence_count.most_common(n)

	def specific_download(self, field, reference, n=1):
		activity_times = [
			'10 min', '10-20 min', '20-30 min',
			'30-60 min', '1-2 h', '2-3 h', 
			'3-8 h', '>8 h'
			]
		modes = ['unknown_activity_type', 'Car', 'Walk', 'Bike',
			'Bus/Tram', 'in_passenger_vehicle', 'Train', 'in_bus', 'Subway',
			'flying', 'motorcycling', 'running']

		query = self.db.child(field).order_by_child("user_id").equal_to(reference).get().val()
		
		pretty_query = ''
		categories = []
		destinations = []
		for i in query.values():
			categories.append(i['category'])
			destinations.append(i['D']['name'])
			s = f"From {i['O']['name']} to {i['D']['name']} for {activity_times[i['activity_time']]} by {modes[i['mode']]}\n"
			pretty_query += s
			
		n_trips = len(query)
		most_categories = self.most_frequent(categories, n)
		most_destinations = self.most_frequent(destinations, n)

		return pretty_query, n_trips, most_categories, most_destinations	

if __name__ == '__main__':
	d = OrderedDict([('127081263', {'age': 24, 'gender': 0, 'occupation': 0})])
	print(d)
	print(d['127081263'])