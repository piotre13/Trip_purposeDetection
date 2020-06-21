from telegram.ext import BaseFilter

class AgeFilter(BaseFilter):
	def filter(self, message):
		try:
			n = int(message.text)
			if 1<=n<=120:
				return True
			else:
				return False
		except:
			return False

class ModeFilter(BaseFilter):
	def filter(self, message):
		modes = ['unknown_activity_type', 'Car', 'Walk', 'Bike',
			'Bus/Tram', 'in_passenger_vehicle', 'Train', 'in_bus', 'Subway',
			'flying', 'motorcycling', 'running']

		if message.text in modes:
			return True
		else:
			return False

class GenderFilter(BaseFilter):
	def filter(self, message):
		genders = ['Male', 'Female', 'M', 'F', 'Non-Binary']
		if message.text in genders:
			return True
		else:
			return False

class ActivityTimeFilter(BaseFilter):
	def filter(self, message):
		activities = ['10 min', '10-20 min', '20-30 min','30-60 min', '1-2 h', '2-3 h','3-8 h', '>8 h']
		if message.text in activities:
			return True
		else:
			return False

class PurposeFilter(BaseFilter):
	def filter(self, message):
		categories = ['Home','Work','Eating','Entertainment','Recreation',
				'Shopping','Travel','Admin Chores','Religious','Health','Police','Education', 'Commuting']
		if message.text in categories:
			return True
		else:
			return False

class GameModeFilter(BaseFilter):
	def filter(self, message):
		if message.text == 'Yes' or message.text == 'No':
			return True
		else:
			return False

class OccupationFilter(BaseFilter):
	def filter(self, message):
		occupations = ['Student', 'Worker', 'Retired']
		if message.text in occupations:
			return True
		else:
			return False

class DestinationStartFilter(BaseFilter):
	def filter(self, message):
		hours = [str(i) for i in range(24)]
		if message.text in hours:
			return True
		else:
			return False

class WeekFilter(BaseFilter):
	def filter(self, message):
		if message.text in ['WeekDay Trip', 'WeekEnd Trip']:
			return True
		else:
			return False

class CategoryFilter(BaseFilter):
	def filter(self, message):
		if message.text in  ['Home','Work','Eating','Entertainment','Recreation','Shopping','Travel','Chores','Religious','Health','Police','Education']:
			return True
		else:
			return False