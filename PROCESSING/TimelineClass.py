import os
import json
import xmltodict
import datetime
import time
from pathlib import Path
from CensusMatching import CensusMatching
import traceback 
import shapely
import matplotlib.pyplot as plt
from UserClass import User

class Timeline(object):
	def __init__(self, file_out, type_='takeout', lat_min=45.0070, lat_max=45.1336, lng_min=7.5953, lng_max=7.7691):
		p1 = shapely.geometry.Point(lng_min, lat_min)
		p2 = shapely.geometry.Point(lng_min, lat_max)
		p3 = shapely.geometry.Point(lng_max, lat_max)
		p4 = shapely.geometry.Point(lng_max, lat_min)
		# self.db_names = {}
		self.TorinoPolygon = shapely.geometry.Polygon([p.x,p.y] for p in [p1,p2,p3,p4])
		self.cm = CensusMatching()
		self.file_out = file_out

		if type_ == 'takeout':
			with open('OUTPUT/final_mobility/db.json') as db:
				self.db_names = json.loads(db.read())
			main_folder = Path('../DATA/data_timeline/')

			self.open_json(main_folder)

			with open('OUTPUT/final_mobility/db.json','w') as db:
				db.write(json.dumps(self.db_names, indent=4))

		elif type_ == 'kml':
			main_folder = Path('../DATA/data_kml/')
			self.open_kml(main_folder)

	def process(self, list_obj):
		with open(self.file_out) as f:
			out = json.loads(f.read())
		f.close()
		for i in list_obj:
			out.append(i)

		with open(self.file_out, 'w') as f:
			f.write(json.dumps(out, indent=4))
		f.close()

	def open_json(self, main_folder):
		#main_folder: folder containing folder of user containing all the years
		aldready_processed = os.listdir(Path('../DATA/data_timeline_processed/'))
		for folder in os.listdir(main_folder):
			user = folder
			user = User(user)
			user.process()
			if folder not in aldready_processed:
				#folder: name of timeline_folder, name of user
				print(folder)
				subfolders_path = os.path.join(main_folder, folder)
				subfolders = os.listdir(subfolders_path)
				for subfolder in subfolders:
					#subfolder: year of timeline
					if subfolder != 'config.txt':
						subsubfolder_path = os.path.join(subfolders_path, subfolder)
						user_id = f"{user}_{subfolder}"
						out_path = Path(f'../DATA/data_timeline_processed/{user}/{subfolder}')
						try:
							out_path.mkdir(parents=True, )
						except:
							pass
						for filename in os.listdir(subsubfolder_path):
							print(f'--> {filename}')
							file_path = os.path.join(subsubfolder_path, filename)
							with open(file_path, encoding="utf8") as f:
								obj = json.loads(f.read())
							f.close()

							out_obj = self.process_json(obj, user_id)
							self.process(out_obj)
							out_open = f'{out_path}/{filename}'
							if len(out_obj)>0:
								with open(out_open, 'w+') as f_out:
									f_out.write(json.dumps(out_obj, indent=4))
								f_out.close()
					else:
						pass
			else:
				print(f'{folder} Aldready Processed')

	def process_json(self, obj, user_id):
		out_obj = []
		
		timeline_obj = obj['timelineObjects']
		cnt = 0
		n = len(timeline_obj)
		for timeline in timeline_obj:
			cnt += 1
			try:
				if timeline['activitySegment']:
					new_obj = timeline['activitySegment']
					is_t = self.is_torino(new_obj)
					if is_t:
						act_time = 0
						address = 'Not_Found'
						name = 'Not_Found'
						cat = {
								"HOME": "",
								"WORK": "",
								"EATING": "",
								"ENTERTAINMENT": "",
								"RECREATION": "",
								"SHOPPING": "",
								"TRAVEL": "",
								"ADMINISTRATION_CHORES": "",
								"RELIGION": "",
								"HEALTH": "",
								"POLICE": "",
								"EDUCATION": ""
			        		}
						or_lat = new_obj['startLocation']['latitudeE7']/1e7
						or_lng = new_obj['startLocation']['longitudeE7']/1e7
						de_lat = new_obj['endLocation']['latitudeE7']/1e7
						de_lng = new_obj['endLocation']['longitudeE7']/1e7


						census_id_origin = self.cm.census_matching(or_lat, or_lng)
						census_id_destination = self.cm.census_matching(de_lat, de_lng)
						#if next element in the list is a placeVisit
						if timeline_obj[cnt]['placeVisit']:
							place = timeline_obj[cnt]['placeVisit']
							if 'semanticType' in list(place['location'].keys()):
								cat = place['location']['semanticType']
							#activity_time, time spent in the destination, mesured in seconds
							address = place['location']['address']
							name = place['location']['name']
							act_time = (int(place['duration']['endTimestampMs']) - int(place['duration']['startTimestampMs']))/1000

						mode_ = ""
						#if there is a list of possible mode of transportation, choose the MostLikelihood from the list
						if len(new_obj['activities']) > 0:
							mode_ = new_obj['activities'][0]['activityType']
						#TODO: choose a method to insert the _id
						tmp_obj = {
							"_id":'',
							"user_id":user_id,
							"O":{
								"lat": or_lat,
								"lng": or_lng,
								"timestamp": str(int(new_obj['duration']['startTimestampMs'])//1000),
								"census_id": census_id_origin
							},
							"D":{
								"lat": de_lat,
								"lng": de_lng,
								"timestamp": str(int(new_obj['duration']['endTimestampMs'])//1000),
								"census_id": census_id_destination
							},
							"activity_time": act_time,
							"mode": mode_,
							"address_dest": address,
							"name": name,
							"category": cat
						}
						list_names = list(self.db_names.keys())
						key = f'{name} - {address}'
						if key not in list_names:
							self.db_names[key] = [census_id_destination]
						else:
							self.db_names[key].append(census_id_destination)

						if act_time > 0:
							out_obj.append(tmp_obj)
			except:
				# traceback.print_exc()
				continue
			
		return out_obj

	def json_to_excel(self, year, json_file):
		dataset = tablib.Dataset()
		dataset.json = open(json_file, 'r').read()
		data_export = dataset.export('xlsx')
		xlsx_file = json_file.split('/')[2].replace('.json','.xlsx')
		try:
			os.mkdir(f'data_timeline_processed_xlsx/{year}')
		except:
			pass
		new_path = f'data_timeline_processed_xlsx/{year}/{xlsx_file}'
		with open(new_path, 'wb') as f:
			f.write(data_export)

	def is_torino(self, current_trip, lat_min=45.0070, lat_max=45.1336, lng_min=7.5953, lng_max=7.7691):
		start_lat = current_trip['startLocation']['latitudeE7']/1e7
		start_lng = current_trip['startLocation']['longitudeE7']/1e7
		end_lat = current_trip['endLocation']['latitudeE7']/1e7
		end_lng = current_trip['endLocation']['longitudeE7']/1e7

		origin_point = shapely.geometry.Point(start_lng, start_lat)
		destination_point = shapely.geometry.Point(end_lng, end_lat)

		if self.TorinoPolygon.contains(origin_point) or self.TorinoPolygon.contains(destination_point):
			return 1
		else:
			return 0

	def open_kml(self, path_kml):
		for file_name in os.listdir(path_kml):
			print(file_name)
			f_open = os.path.join(path_kml, file_name)
			with open(f_open) as f:
				file_out = 'kml_to_json/riccardo/'+file_name.replace('.kml', '.json')
				doc = f.read()
				with open(file_out, 'w+') as f_out:
					f_out.write(json.dumps(xmltodict.parse(doc), indent=4))
			
			with open(file_out) as f_new:
				obj = json.loads(f_new.read())

			self.process_kml(obj, 'kml_trips.json')

	def process_kml(self, obj_in, f_out):
		user = ''
		activity = None
		obj = []
		for i in obj_in['kml']['Document']['Placemark']:
			if len(user) < 1:
				user = i['ExtendedData']['Data'][0]['value']

			if 'LineString' in i.keys():
				j = i['LineString']
				coordinates = j['coordinates'].split(" ")
				start = coordinates[0]
				end = coordinates[-1]

				start_lat = start.split(',')[1]
				start_lng = start.split(',')[0]
				end_lat = end.split(',')[1]
				end_lng = end.split(',')[0]

				Date_Begin = i['TimeSpan']['begin'].replace('T', '-').split('.')[0]
				Date_End = i['TimeSpan']['end'].replace('T', '-').split('.')[0]

				start_timestamp = time.mktime(datetime.datetime.strptime(Date_Begin, "%Y-%m-%d-%H:%M:%S").timetuple())
				end_timestamp = time.mktime(datetime.datetime.strptime(Date_End, "%Y-%m-%d-%H:%M:%S").timetuple())

				_id = ''
				user_id = user
				O = {
				'lat':start_lat,
				'lng':start_lng,
				'timestamp':start_timestamp,
				'census_zone_id':''
				}

				D = {
				'lat':end_lat,
				'lng':end_lng,
				'timestamp':end_timestamp,
				'census_zone_id':''
				}

				activity_time = ''
				# if activity != None:
				# 	print('')
				# 	activity_time = activity

				cat = {
					"HOME": "",
					"WORK": "",
					"EATING": "",
					"ENTERTAINMENT": "",
					"RECREATION": "",
					"SHOPPING": "",
					"TRAVEL": "",
					"ADMINISTRATION_CHORES": "",
					"RELIGION": "",
					"HEALTH": "",
					"POLICE": "",
					"EDUCATION": ""
				}
				mode = i['name']
				tmp_obj = {
					'_id':_id,
					'user_id':user_id,
					'O':O,
					'D':D,
					'activity_time': '',
					'category':cat,
					'mode':mode
				}

				obj.append(tmp_obj)
			else:
				begin = i['TimeSpan']['begin']
				end = i['TimeSpan']['end']

				begin = begin.split('.')[0].replace('T','-')
				end = end.split('.')[0].replace('T','-')

				begin = time.mktime(datetime.datetime.strptime(begin, "%Y-%m-%d-%H:%M:%S").timetuple())
				end =  time.mktime(datetime.datetime.strptime(end, "%Y-%m-%d-%H:%M:%S").timetuple())

				activity = end - begin

		with open(f_out, 'w+') as f:
			f.write(json.dumps(obj, indent=4))

if __name__ == '__main__':
	tl = Timeline('MobilityDataset_2.json')
	