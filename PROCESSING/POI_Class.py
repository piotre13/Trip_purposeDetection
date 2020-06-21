import json
import pandas as pd
from pprint import pprint
import os
import math
from math import pi
import pyproj
import geopy.distance as geo_D
import mpu
from shapely.geometry import shape, Point

#part2
from CONFIG import*

class Google_POI:

	def __init__(self,data_path):
		
		data = ['Google_POI_1.json','Google_POI_2.json','Google_POI_3.json']
		self.type_list = ['church','mosque','synagogue','school','secondary_school','university','cemetery','police','pharmacy','local_government_office']
		self.cnt = 0
		self.out = []
		
		
		for file in data: 
			self.processing(data_path+file)

		print ('Total number of valid Google POIs: ',len(self.out))
		df = pd.DataFrame(self.out)
		print (df['Types'])
		df.to_csv('OUTPUT/G_POIs_sezioniCens.csv')
		df.to_excel('OUTPUT/G_POIs_sezioniCens.xlsx')


	def processing(self,file):
		print('processinf file1 G_POI')

		with open(file) as f:
			obj = json.loads(f.read())
	
		for i in obj:
			results = i['results']
			for r in results:
				self.cnt += 1
				try:
					tmp = {}
					tmp['ID'] = r['id']
					#removing the double 
					if tmp['ID'] not in [j['ID'] for j in self.out]:
						tmp['PlaceID'] = r['place_id']
						tmp['Name'] = r['name']
						# print(tmp['Name'])
						tmp['Types'] = r['types']
						tmp['ViewportNE_Latitude'] = r['geometry']['viewport']['northeast']['lat']
						tmp['ViewportNE_Longitude'] = r['geometry']['viewport']['northeast']['lng']
						tmp['ViewportSW_Latitude'] = r['geometry']['viewport']['southwest']['lat']
						tmp['ViewportSW_Longitude'] = r['geometry']['viewport']['southwest']['lng']
						tmp['Latitude'] = r['geometry']['location']['lat']
						tmp['Longitude'] = r['geometry']['location']['lng']
						#tmp['SEZCENS'] = self.census_matching(self.sezCens_file,tmp['Latitude'],tmp['Longitude'])
					

						#remove the points with typology already present on the external Istat file
						present = any(x in tmp['Types'] for x in self.type_list)
						if not present:
							self.out.append(tmp)
							
					else:
						#print('POI already found')
						pass
				except:
					print('Key Not Found')
					pass

		print('Total Results Found: {}'.format(self.cnt))
	
	def census_matching (self,sez_file,trip_lat,trip_lng):
		'''
		this function find the census zone to which the point belongs
		INPUT: 	- geojson file of the census zones
				- float: lat of the point
				- float: long of the point
		OUTPUT: - int : the census zone
				- str : if the zone was not found
		'''
		
		#with open(sez_file) as f:
		#	js = json.load(f)
		js = sez_file

		point = Point(trip_lng, trip_lat)
		#print (point)

		for feature in js['features']:
			data = {"type": "MultiPolygon", "coordinates": feature['geometry']}
			#polygon = shape(data)
			polygon = shape(feature['geometry'])
			if polygon.contains(point):
				SEZ = int(feature['properties']['SEZCENS'])
				#print ('Found containing polygon: {}'.format(int(feature['properties']['SEZCENS'])))
			else:
				#print ('sez census not find!')
				SEZ = 'NOT_found'

		return SEZ



class QGIS_POI:
	def __init__(self,in_path):
		
		self.in_path = in_path
		self.sezCens_file = 'SEZ_cens.geojson'

		self.processing('OUTPUT/')


	def processing(self, out_path):
		all_POI =[]
		
		for filename in os.listdir(self.in_path):
			path = self.in_path+filename
		
			with open(path, errors='ignore') as f:
				try:
					data = json.load(f)
				except:
					print (filename,' Mac system file not in data')

				if filename in ['altri_culti.geojson','chiese.geojson']:
					#coordinates conversion
					epsg = pyproj.Proj(init='epsg:3003')
					wgs84 = pyproj.Proj(init="epsg:4326")
					for feature in data['features']:

						latitude, longitude = feature['geometry']['coordinates'][1], feature['geometry']['coordinates'][0]

						lng, lat = pyproj.transform(epsg, wgs84, longitude, latitude)

						feature['geometry']['coordinates'][1] = lat
						feature['geometry']['coordinates'][0] = lng			


			for feature in data['features']:
				#print (feature['properties']) #feature['properties'] ritorna un dict con il dato singolo
				#print (feature['geometry']['coordinates'])
				tmp_obj = {}
				words = feature['properties']['TYPE'].split(",")
				tmp_types =[]
				for w in words:
					tmp_types.append(w)
				tmp_obj['Types']= tmp_types
				tmp_obj['Name']= feature['properties']['NAME']
				tmp_obj['Latitude']= feature['geometry']['coordinates'][1]
				tmp_obj['Longitude']= feature['geometry']['coordinates'][0]
				#tmp_obj['SEZCENS'] = self.census_matching(self.sezCens_file,tmp_obj['Latitude'],tmp_obj['Longitude'])

				
				if 'area' in feature['properties'].keys():
					tmp_obj['RADIUS'] = math.sqrt(feature['properties']['area']/(pi))
				
				all_POI.append(tmp_obj)	
				#print (tmp_obj)

		#create a dataframe
		df = pd.DataFrame(all_POI)
		#print (df)
		df.to_csv(out_path+'POI_other_sezioniCens.csv')
		df.to_excel(out_path+'POI_other_sezioniCens.xlsx')

	



class POI_analysis:
	def __init__(self,G_POI,Q_POI):
		self.files = [G_POI,Q_POI]
		
		#self.data1 = pd.read_excel(G_POI)
		#self.data2 = pd.read_excel(Q_POI)
	
	def categories_describe(self):
		#extract the dinstinct categories present in the data, in order to make classification
		categories1 = []
		categories2 = []
		categories = []
		for index, row in self.data1.iterrows():
			tmp = self.str_to_list(row['Types'])
			for cat in tmp:
				if cat not in categories:
					categories.append(cat)
				else:
					continue

		for index, row in self.data2.iterrows():
			tmp = self.str_to_list(row['Types'])
			for cat in tmp:
				if cat not in categories:
					categories.append(cat)
				else:
					continue
		f = open('OUTPUT/categories.txt','w+')
		for i in range(len(categories)):
			f.write(categories[i]+'\n')
		print (categories)
		print (len(categories))

	def str_to_list(self, string):
		new_str = string[1:-1]
		new_str = new_str.replace("'",'')
		new_str = new_str.replace(" ",'')
		words = new_str.split(',')
		return words

	def census_matching (self, sez_file):

		with open(sez_file) as f:
			js = json.load(f)
		
		for file in self.files:
			print (file)
			df = pd.read_excel(file)
			df['SEZCENS'] = ''

			for index, row in df.iterrows():

				trip_lat = row['Latitude']
				trip_lng = row['Longitude']

				point = Point(trip_lng, trip_lat)

				for feature in js['features']:
					#polygon = shape(data)
					polygon = shape(feature['geometry'])
					if polygon.contains(point):
						SEZ = int(feature['properties']['SEZCENS'])
						break
					else:
						#print ('sez census not find!')
						SEZ = 'NOT_found'

				df.at[index,'SEZCENS'] = SEZ
			
			#data.to_csv()
			df.to_excel(file)

class further_union:
	def __init__(self,Gpoi,Qpoi):
		self.G_POI = pd.read_excel(Gpoi)
		self.Q_POI = pd.read_excel(Qpoi)
		self.union()
	
	def check_category(self,list1):
		found = False
		cnt = 0
		out = ''
		while not found and cnt != len(categories) :
			cat = categories[cnt]
			set1 = set(TYPES[cat])
			set2 = set(list1)
			common = set1.intersection(set2)
			
			if bool(common):
				print('find!!!')
				out = cat
				found = True
			else:
				cnt+=1
				continue
		
		return out

	def iteration(self,df):
		cnt = 0
		for index, row in df.iterrows():
			#print('POI n: ',index)
			list1 = self.str_to_list(row['Types'])
			print (type(list1))
			df.at[index,'Types'] = self.check_category(list1)
			cnt+=1

		return df,cnt

	def str_to_list(self, string):
		new_str = string[1:-1]
		new_str = new_str.replace("'",'')
		new_str = new_str.replace(" ",'')
		words = new_str.split(',')
		return words

	def union(self):
		self.G_POI,cnt1 = self.iteration(self.G_POI)
		self.G_POI = self.G_POI.drop(columns = ['ID','PlaceID','ViewportNE_Latitude','ViewportNE_Longitude','ViewportSW_Latitude','ViewportSW_Longitude'])
		self.Q_POI,cnt2 = self.iteration(self.Q_POI)
		self.Q_POI = self.Q_POI.drop(columns = ['RADIUS'])
		print(self.G_POI.keys())
		print('===================')
		print(self.Q_POI.keys())

		#frames = [self.G_POI, self.Q_POI]
		result = self.G_POI.append(self.Q_POI, sort=False)
		#result = pd.concat(frames)
		print (result)
		print('TOT number of POIs: ',(cnt1+cnt2))
		result.to_excel('OUTPUT/final_territorial/all_POI.xlsx')

			

		


if __name__ == '__main__':

	#PART ONE FOR PROCESSING THE TERRITORIAL
	data_path = '/Users/pietrorandomazzarino/Documents/UNIVERSITA/interdisciplinary project/TripPurposeDetection_Project/DATA/'
	#G_POI = Google_POI(data_path+'Google_POI/')
	#Q_POI = QGIS_POI(data_path+'other_POI/layer_puntuali_finali/')
	#in1 = 'OUTPUT/G_POIs_sezioniCens.xlsx'
	#in2 = 'OUTPUT/POI_other_sezioniCens.xlsx'
	#Assignation = POI_analysis(in1,in2)
	#Assignation.census_matching('SEZ_cens.geojson')

	#PART TWO FURTHER UNION OF THE DATASETS FOR LABELING PURPOSES
	G_path = 'OUTPUT/G_POIs_sezioniCens.xlsx'
	Q_path = 'OUTPUT/POI_other_sezioniCens.xlsx'
	UN = further_union(G_path,Q_path)
	#UN.check_category(['stocazzo','stocazzo2'])
	UN.union()








