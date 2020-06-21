import json
from pathlib import Path
import pprint
import traceback

class MobilitèDinamique(object):
	"""docstring for MobilitèDinamique"""
	def __init__(self, file_out):
		self.file_out = Path(file_out)
		input_path = Path('../DATA/mobilitè_dinamique/MobilitèDinamique_11_12_19.json')
		self.process(input_path)

	def process(self, file_in, time_threshold=30*60*1000):
		with open(self.file_out) as f_out:
			out = json.loads(f_out.read())
		f_out.close()
		with open(file_in) as f_in:
			obj = json.loads(f_in.read())
		f_in.close()

		tmp_out = []
		obj = sorted(obj, key= lambda x: (x['userId'], x['date']))
		cnt = 0
		while cnt < len(obj):
			print(cnt)
			try:
				for j in range(cnt+1, len(obj)):
					if(obj[cnt]['userId'] == obj[cnt+j]['userId']):
						curr_t = obj[cnt+j]['date']
						next_t = obj[cnt+j+1]['date']
						dt = next_t - curr_t
						if(dt >= time_threshold):
							tmp_obj = {}
							tmp_obj['o_user'] = obj[cnt]['userId']
							tmp_obj['d_user'] = obj[cnt+j]['userId']
							tmp_obj['O'] = {}
							tmp_obj['O']['lat'] = obj[cnt]['lat']
							tmp_obj['O']['lng'] = obj[cnt]['lng']
							tmp_obj['O']['timestamp'] = obj[cnt]['date']
							tmp_obj['D'] = {}
							tmp_obj['D']['lat'] = obj[cnt+j]['lat']
							tmp_obj['D']['lng'] = obj[cnt+j]['lng']
							tmp_obj['D']['timestamp'] = obj[cnt+j]['date']
							tmp_obj['delta'] = dt/60
							tmp_out.append(tmp_obj)
							cnt = j
							break
					else:
						cnt = j
			except:
				break	
			
		with open('test.json', 'w+') as f:
			f.write(json.dumps(tmp_out, indent=4))

if __name__ == '__main__':
	OUTPUT_FILE = 'OUTPUT/final_mobility/MobilityDataset.json'
	mf = MobilitèDinamique(OUTPUT_FILE)