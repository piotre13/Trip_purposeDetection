import json
from pathlib import Path
import os

class User:
	def __init__(self, user):
		self.user = user
		self.timeline_path = Path('../DATA/data_timeline/')
		self.file_out = Path('OUTPUT/final_mobility/UserDataset.json')
		with open(self.file_out) as f:
			self.out = json.loads(f.read())
		# self.out = []

	def process(self):
		n = 2020
		user_path = Path(f'{self.timeline_path}/{self.user}')
		list_dir = os.listdir(user_path)
		config = list_dir[-1]
		path_config = Path(f'{user_path}/{config}')
		years = list_dir[0:len(list_dir)-1]
		f = open(path_config, 'r')
		tmp_obj = {}
		tmp_obj['user_id'] = f'{self.user}_{2020}'
		for line in f.readlines():
			split = line.split(':')
			if split[0] == 'age':
				tmp_obj['age'] = int(split[1].strip())
			if split[0] == 'gender':
				if split[1].strip() == 'male' or split[1].strip() == 'Male':
					tmp_obj['gender'] = 0
				else:
					tmp_obj['gender'] = 1
			if split[0] == 'occupation':
				tmp_obj['occupation'] = (split[1].strip())
		self.out.append(tmp_obj)
		current_age = tmp_obj['age']
		for year in years:
			year = int(year)
			if year == n:
				continue
			else:
				diff = n - year
				new_obj = {}
				new_obj['user_id'] = f"{self.user}_{year}"
				new_obj['age'] = current_age - diff
				new_obj['gender'] = tmp_obj['gender']
				new_obj['occupation'] = tmp_obj['occupation']
				self.out.append(new_obj)

		with open(self.file_out, 'w+') as fout:
			fout.write(json.dumps(self.out, indent=4))

if __name__ == '__main__':
	u = User()
	u.process()