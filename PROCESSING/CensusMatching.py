import json
from shapely.geometry import shape, Point
from pathlib import Path

class CensusMatching:
	def __init__(self):
		file_path = Path('data/SEZ_cens.geojson')
		with open(file_path) as f:
			self.obj = json.loads(f.read())

	def reference_points(self):
		N = {
			'lng':7.6929,
			'lat':45.1335
		}
		NE = {
			'lng':7.7456,
			'lat':45.1185
		}
		E = {
			'lng':7.7613,
			'lat':45.0670
		}
		SE = {
			'lng':7.7139,
			'lat':45.0296
		}
		S = {
			'lng':7.6544,
			'lat':45.0068
		}
		SW = {
			'lng':7.5849,
			'lat':45.0363
		}
		W = {
			'lng':7.5919,
			'lat':45.0851
		}
		NW = {
			'lng':7.6240,
			'lat':45.1128
		}

		pn = Point(N['lng'], N['lat'])
		pne = Point(NE['lng'], NE['lat'])
		pe = Point(E['lng'], E['lat'])
		pse = Point(SE['lng'], SE['lat'])
		ps = Point(S['lng'], S['lat'])
		psw = Point(SW['lng'], SW['lat'])
		pw = Point(W['lng'], W['lat'])
		pnw = Point(NW['lng'], NW['lat'])
		coordinates = [pn,pne,pe,pse,ps,psw,pw,pnw]
		# area = Polygon([p.x,p.y] for p in coordinates)
		# x,y = area.exterior.xy
		# plt.plot(x,y)
		# plt.show()
		return coordinates


	def converter(self, lat, lng):
		coordinates = self.reference_points()
		coordinates_dict = {
			0:'N',
			1:'NE',
			2:'E',
			3:'SE',
			4:'S',
			5:'SW',
			6:'W',
			7:'NW'
		}
		point = Point(lng, lat)
		distances = [point.distance(p) for p in coordinates]
		index_min_dist = distances.index(min(distances))
		census_id = coordinates_dict[index_min_dist]
		print(f'--->{census_id}')
		return census_id

	def census_matching(self, trip_lat, trip_lng):
		'''
		this function find the census zone to which the point belongs
		INPUT: 	- geojson file of the census zones
				- float: lat of the point
				- float: long of the point
		OUTPUT: - int : the census zone
				- str : if the zone was not found
		'''
		point = Point(trip_lng, trip_lat)
		for feature in self.obj['features']:
			polygon = shape(feature['geometry'])
			if polygon.contains(point):
				SEZ = int(feature['properties']['SEZCENS'])
				print(f'--->{SEZ}')
				return SEZ
			else:
				SEZ = 'NOT_found'

		if SEZ == 'NOT_found':
			SEZ = self.converter(trip_lat, trip_lng)
			return SEZ
			
if __name__ == '__main__':
	cm = CensusMatching()
	cm.census_matching(46.0000, 7.0000)