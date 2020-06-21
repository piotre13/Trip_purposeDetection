import json
import datetime
from pytz import timezone
import numpy as np
from pprint import pprint
import pandas as pd
import os
from pathlib import Path
from CensusMatching import CensusMatching

class M_trip:
	def __init__(self, file_out):
		path_file1 = '../DATA/trip_logs/TripLogs_8_12_19.json'
		path_file2 = '../DATA/trip_logs/trip_details_geojson.json'
		path1 = Path(path_file1)
		path2 = Path(path_file2)
		self.cm = CensusMatching()
		self.system_unification(path1, path2, file_out)

	def system_unification(self, file_1, file_2, file_out):
		with open(file_1) as f1:
			obj_1 = json.loads(f1.read())
		f1.close()
		with open(file_2) as f2:
			obj_2 = json.loads(f2.read())
		f2.close()
		with open(file_out) as f_out:
			out = json.loads(f_out.read())
		f_out.close()

		for i in obj_1:
			print(i['_id'])
			tmp_obj = {}
			tmp_obj['_id'] = i['_id']
			# tmp_obj['Type'] = 'Feature'
			# tmp_obj['properties'] = {}

			tmp_obj['user_id'] = i['userEmailID']
			origin_date, destination_date = self.date_creation(i['tripDate'], i['timeFrame'])
			# tmp_obj['properties']['Date'] = origin_date
			
			
			# tmp_obj['properties']['origin_zone']['census_id'] = ''
			tmp_obj['mode'] = i['detectedMode']

			# tmp_obj['properties']['origin_zone']['loc'] = {}
			# tmp_obj['properties']['origin_zone']['loc']['type'] = 'Point'
			# tmp_obj['properties']['origin_zone']['loc']['coordinates'] = [float(i['longitude'][0]), float(i['latitude'][0])]
			tmp_obj['O'] = {}
			tmp_obj['O']['lat'] = float(i['latitude'][0])
			tmp_obj['O']['lng'] = float(i['longitude'][0])
			tmp_obj['O']['timestamp'] = origin_date
			tmp_obj['O']['census_id'] = self.cm.census_matching(tmp_obj['O']['lat'], tmp_obj['O']['lng'])

			# tmp_obj['properties']['origin_zone']['date'] = origin_date
			# tmp_obj['properties']['origin_zone']['user'] = i['userEmailID']

			# tmp_obj['properties']['destination_zone'] = {}
			# tmp_obj['properties']['destination_zone'] = {}
			# tmp_obj['properties']['destination_zone']['census_id'] = ''
			# tmp_obj['properties']['destination_zone']['mode'] = i['detectedMode']

			# tmp_obj['properties']['destination_zone']['loc'] = {}
			# tmp_obj['properties']['destination_zone']['loc']['type'] = 'Point'
			# tmp_obj['properties']['destination_zone']['loc']['coordinates'] = [float(i['longitude'][-1]), float(i['latitude'][-1])]
			tmp_obj['D'] = {}
			tmp_obj['D']['lat'] = float(i['latitude'][-1])
			tmp_obj['D']['lng'] = float(i['longitude'][-1])
			tmp_obj['D']['timestamp'] = destination_date
			tmp_obj['D']['census_id'] = self.cm.census_matching(tmp_obj['D']['lat'], tmp_obj['D']['lng'])

			# tmp_obj['properties']['destination_zone']['date'] = destination_date
			# tmp_obj['properties']['destination_zone']['user'] = i['userEmailID']

			# tmp_obj['properties']['transit_zones'] = {}
			# tmp_obj['properties']['transit_zones']['zone_list'] = []
			# tmp_obj['properties']['transit_zones']['zone_list'] = self.zone_list_creation(i, origin_date, destination_date, i['timeTaken'])

			# tmp_obj['properties']['trip_duration'] = str(i['timeTaken']) + ' minutes, 00 seconds'

			# tmp_obj['geometry'] = {}
			# tmp_obj['geometry']['type'] = 'LineString'
			# tmp_obj['geometry']['coordinates'] = []
			# tmp_obj['geometry']['coordinates'] = self.coordinates_creation(i)
			tmp_obj['activity_time'] = 0
			tmp_obj['category'] = 0
			out.append(tmp_obj)

		for i in obj_2:
			tmp_obj['_id'] = i['_id']
			tmp_obj['user_id'] = i['properties']['UserId']
			tmp_obj['O'] = {}
			tmp_obj['D'] = {}
			if len(str(i['properties']['Date']).split('.')[0]) == 7:
				tmp_obj['O']['timestamp'] = i['properties']['Date'] *1000 #data was in milliseconds
				tmp_obj['D']['timestamp'] = i['properties']['destination_zone']['date'] *1000
				# i['properties']['origin_zone']['date'] = i['properties']['origin_zone']['date'] *1000
				# i['properties']['destination_zone']['date'] = i['properties']['destination_zone']['date'] *1000

				# for j in i['properties']['transit_zones']['zone_list']:
					# j['date'] = j['date'] *1000

			elif len(str(i['properties']['Date']).split('.')[0]) == 13:
				tmp_obj['O']['timestamp'] = i['properties']['Date'] //1000 #data was in milliseconds
				tmp_obj['D']['timestamp'] = i['properties']['destination_zone']['date'] //1000
				# i['properties']['origin_zone']['date'] = i['properties']['origin_zone']['date'] /1000
				# i['properties']['destination_zone']['date'] = i['properties']['destination_zone']['date'] /1000

				# for j in i['properties']['transit_zones']['zone_list']:
					# j['date'] = j['date'] /1000
			tmp_obj['O']['lat'] = float(i['properties']['origin_zone']['loc']['coordinates'][1])
			tmp_obj['O']['lng'] = float(i['properties']['origin_zone']['loc']['coordinates'][0])
			tmp_obj['O']['census_id'] =  self.cm.census_matching(tmp_obj['O']['lat'], tmp_obj['O']['lng'])

			tmp_obj['D']['lat'] = float(i['properties']['destination_zone']['loc']['coordinates'][1])
			tmp_obj['D']['lng'] = float(i['properties']['destination_zone']['loc']['coordinates'][1])
			tmp_obj['D']['census_id'] =  self.cm.census_matching(tmp_obj['D']['lat'], tmp_obj['D']['lng'])
			tmp_obj['activity_time'] = 0
			tmp_obj['category'] = 0

			out.append(tmp_obj)
		# out = {
		# 	'type':'FeatureCollection',
		# 	'features':''
		# }
		# out['features'] = obj_2
		with open(file_out, 'w+') as f_out:
			f_out.write(json.dumps(out, indent=4))
		f_out.close()

	def date_creation(self, trip_date, time_frame, tzone=timezone('Europe/Paris')):
		start_time = time_frame.split('-')[0]
		finish_time = time_frame.split('-')[1]

		starts = [int(i) for i in start_time.split(':')]
		h_s = starts[0]
		m_s = starts[1]
		s_s = starts[2]

		finishes = [int(i) for i in finish_time.split(':')]
		h_f = finishes[0]
		m_f = finishes[1]
		s_f = finishes[2]

		day = int(trip_date.split('-')[0])
		month = int(trip_date.split('-')[1])
		year = int(trip_date.split('-')[2])

		start = (datetime.datetime(year, month, day, h_s, m_s, s_s, tzinfo=tzone) - 
			datetime.datetime(1970, 1, 1, tzinfo=tzone)).total_seconds()
		finish = (datetime.datetime(year, month, day, h_f, m_f, s_f, tzinfo=tzone) - 
			datetime.datetime(1970, 1, 1, tzinfo=tzone)).total_seconds()

		return start, finish

	def zone_list_creation(self, record, start, end, time_taken):
		transit_zones_list = []
		time_taken = 60*time_taken
		n = len(record['latitude'])
		times = np.linspace(start, end, n) #set date for intermediate date

		for i in range(1,n-1): #first is origin, last is destination -> must not be inserted 
			tmp_obj = {}
			tmp_obj['census_id'] = ''
			tmp_obj['mode'] = record['detectedMode']

			tmp_obj['loc'] = {}
			tmp_obj['loc']['type'] = 'Point'
			tmp_obj['loc']['coordinates'] = [record['longitude'][i], record['latitude'][i]]
			tmp_obj['date'] = times[i]
			tmp_obj['user'] = record['userEmailID']

			transit_zones_list.append(tmp_obj)

		return transit_zones_list

	def coordinates_creation(self, record):
		coordinates = []
		for i in range(len(record['latitude'])):
			coordinates.append([float(record['longitude'][i]), float(record['latitude'][i])])
		return coordinates

# if __name__ == '__main__':
# 	f1 = '../DATA/trip_logs/TripLogs_8_12_19.json'
# 	f2 = '../DATA/trip_logs/trip_details_geojson.json'
# 	mt = M_trip(f1, f2, 'unification.json')