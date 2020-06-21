import pandas as pd
from datetime import datetime
import collections
import sys





class Systematic_mobility:

	def __init__(self,MO, US, out_folder):
		self.MO = pd.read_excel(MO)
		self.MO = self.MO.astype({"category": str}) 
		#conversion of timestamp into datetime
		self.MO.loc[:,'o_datetime'] = pd.to_datetime(self.MO.loc[:,'o_ts'],unit = 's')
		self.MO.loc[:,'d_datetime'] = pd.to_datetime(self.MO.loc[:,'d_ts'],unit = 's')
		self.USER_list = self.MO.user_id.unique() 
		self.US = pd.read_excel(US)
		self.OUT_folder = out_folder
		
		#this function runs the home/education/work trip assignation
		self.final_assignation()	

	def home_detection1(self):
		'''
		this function detects the home places of users and save them into the users csv
		the detection is done by looking at frequencies of evening trips with some conditions and morning departures
		'''
		act_time_th = 5 #activity time for evening sleep
		users_home = {} #initialize the final home dict
		#intervals considered
		int_ev1 = datetime.strptime('18:00:00', '%H:%M:%S').time()
		int_ev2 = datetime.strptime('21:00:00', '%H:%M:%S').time()
		int_ev3 = datetime.strptime('23:59:59', '%H:%M:%S').time()
		int_mo1 = datetime.strptime('6:00:00', '%H:%M:%S').time()
		int_mo2 = datetime.strptime('12:00:00', '%H:%M:%S').time()
		
		
		for user in self.USER_list: 

			user_df = self.MO[self.MO.user_id == user]
			year = user[-4:]
	
			#preparo i criteri per la selezione
			crit_ev1 = user_df['o_datetime'].dt.time.map(lambda x: int_ev1 < x < int_ev2)
			crit_ev2 = user_df['d_datetime'].dt.time.map(lambda x: int_ev2 < x < int_ev3)
			crit_mo = user_df['o_datetime'].dt.time.map(lambda x: int_mo1 < x < int_mo2)
			#faccio la selezione creando dataframe parziali con solo i viaggi interessati dai criteri
			evening = user_df[crit_ev1 | crit_ev2]
			evening = evening[evening.activity_time > (act_time_th*3600)]
			morning = user_df[crit_mo]
			#filtro le evening per activity time >5 ore DORMIRE
			
			try:
				evening_occ = evening['d_census_id'].value_counts()
				morning_occ = morning['o_census_id'].value_counts()
				
			except KeyError:
				pass
		
			try:
				if year in ['2016','2017','2018','2019','2020']:
					home = evening_occ.index[0]
					users_home[user]=home
				else:
					home = morning_occ.index[0]
					users_home[user]= home
		
			except IndexError:
				users_home[user]=0
			
	
		usr = sorted(list(users_home.keys()))
		for user in reversed(usr):
			year = user[-4:]
			year_int = int(user[-4:])
			name = user.replace (year,'')
			if users_home[user]==0:
				users_home[user]= users_home[name+str(year_int-1)]
			elif user == 'Loris_2016':
				users_home[user]= '1278'
			elif user == 'Loris_2015':
				users_home[user]= '1395'
			elif user == 'elena_merola_2014' or user == 'elena_merola_2015':
				users_home[user]= '1504'
			elif user == 'pietro_assandri_2015' or user == 'pietro_assandri_2014':
				users_home[user]='38'

		return users_home
		
	def school_work (self,homes):
		'''this output includes not only work and study but will be corrected in final assignation'''

		act_time_th = 1 # min activity time for work school
		tot_annual_avgTrips = 2.7 * 365
		perc_work = 0.3 #daily
		perc_work_mo = 0.3 * 0.22
		users_w_s = {}

		int_mo1 = datetime.strptime('6:00:00', '%H:%M:%S').time()
		int_mo2 = datetime.strptime('10:00:00', '%H:%M:%S').time()

		for user in self.USER_list: 

			users_w_s[user]=[]
			user_df = self.MO[self.MO.user_id == user]
			year = user[-4:]
			#preparo i criteri per la selezione
			crit_mo = user_df['o_datetime'].dt.time.map(lambda x: int_mo1 < x < int_mo2)
			crit_week = user_df['o_datetime'].dt.date.map(lambda x: x.weekday() in [0,1,2,3,4])
			#faccio le selzeioni
			morning = user_df[crit_mo & crit_week]
			tot_morning_trips = morning.shape[0]
			morning = morning[morning.activity_time > (act_time_th*3600)]
			

			try:
				#TO DO maybe in futher steps highlight people bringing kids to school
				morning_occ = morning['d_census_id'].value_counts()
				morning_occ = dict(morning_occ)
				morning_occ = collections.OrderedDict(morning_occ)
				cnt=0
				for dest in morning_occ:
					#print(dest,':',morning_occ[dest])
					perc = morning_occ[dest]/tot_morning_trips

					if cnt == 0 and str(dest) != str(homes[user]):
						users_w_s[user].append(dest)

					elif cnt!=0 and str(dest)!=str(homes[user]) and (morning_occ[dest]/tot_morning_trips) > perc_work_mo:
						users_w_s[user].append(dest)
					cnt+=1

			except IndexError:
				pass

		#manual refinition
		usr = sorted(list(users_w_s.keys()))
		for user in reversed(usr):
			year = user[-4:]
			year_int = int(user[-4:])
			name = user.replace (year,'')
			try:
				if users_w_s[user]==[] :
					users_w_s[user]= users_w_s[name+str(year_int-1)]
				
			except KeyError:
				if users_w_s[user]==[]:
					users_w_s[user]= users_w_s[name+str(year_int+1)]
	
		return users_w_s

	def working_extraction(self,sez,dict_,list_,occ,user):
		#in this function i extract school work and possible others
		#the key is to recognize among the most 
		#returns a string with the lable found
		#posible lables are work, education, '',
		places = dict_[user]
		occupation = occ
		destination = sez
		working_collection = list_
		if destination in places:
			if str(places[0])== str(destination) and occ =='student':
				#print('00000000000000000')
				return 'education'
			elif str(places[0])== str(destination) and occ =='worker':
				return 'work'
			else:
				if str(destination) in working_collection and occ == 'student':
					#print('00000000000000000')
					return 'education'
				elif str(destination) in working_collection and occ == 'worker':
					return 'work'
				else:
					return ''
		else:
			return ''	

	def final_assignation(self):
		home_dict = self.home_detection1()
		school_work_dict = self.school_work(home_dict)
		list_work = []
		#create the list of work_palces
		for item in school_work_dict:
			list_work.append(school_work_dict[item][0])
	
		
		for index,trip in self.MO.iterrows():
		
			user_id = trip['user_id']
			us = self.US.loc[self.US['user_id'] == user_id]
			if us.empty: us = self.US.loc[self.US['user_id'] == user_id.lower()]
			year = int(user_id[-4:])
			previous_us = user_id[:-4]+str(year-1)
			try:
					old_home = home_dict[previous_us]
					print (old_home)
			except KeyError:
				old_home = 0
				pass 

			i = us.index
			occupation = us.at[i[0],'occupation']

			if trip['category']== 'TYPE_HOME' and user_id not in ['lorenzo_bellone_2018','lorenzo_bellone_2019','lorenzo_bellone_2020','pietro_assandri_2016','pietro_assandri_2017','pietro_assandri_2018','pietro_assandri_2019','pietro_assandri_2020']:
				self.MO.at[index,'category']='home'
				home_dict[user_id]= trip['d_census_id']
			
			elif trip['category']== 'TYPE_WORK' and trip['d_census_id']!= home_dict[user_id] and trip['d_census_id']!= old_home:
			
				if occupation == 'student':
					self.MO.at[index,'category']='education'
					
				else:
					self.MO.at[index,'category']='work'			

			else:
				#celle vuote da riempire con home o work o school
				if trip['d_census_id'] == home_dict[user_id] or trip['d_census_id']== old_home :
					self.MO.at[index,'category']='home'
				
				else:
					label = self.working_extraction(trip['d_census_id'],school_work_dict,list_work,occupation,user_id)
					self.MO.at[index,'category']=label
					
			

		self.MO.to_excel(self.OUT_folder+'final_Dataset.xlsx')
		

class Non_systematic_mobility:
	def __init__(self,FIN,US,POI,out_folder):
		self.FIN = pd.read_excel(FIN)
		self.FIN = self.FIN.astype({"category": str,'user_id': str,'name':str}) #line for types conversion
		self.FIN = self.FIN.astype(str).apply(lambda x: x.str.lower())
		self.US = pd.read_excel(US)
		self.US = self.US.astype({"occupation": str,'user_id': str})
		self.US = self.US.astype(str).apply(lambda x: x.str.lower())
		self.POI = pd.read_excel(POI)
		self.POI = self.POI.astype({"Name": str,'Types': str})
		self.POI = self.POI.astype(str).apply(lambda x: x.str.lower())
		self.output_folder = out_folder
		self.assignation()

	def assignation(self):
		cnt=0
		tot=0
		hand_list=[]
		hand_dict={}
		for index, row in self.FIN.iterrows():
			tot+=1

			if row['category']== 'nan' or row['category']== 'notfound':
				user_id = row['user_id']
				user_index = self.US.loc[self.US['user_id'] == user_id].index
				sez_id = self.FIN.at[index,'d_census_id']
				name = self.FIN.at[index,'name']
			
				if row['d_census_id'] in ['n','w','s','e','ne','nw','se','sw']:
					self.FIN.at[index,'category'] = 'travel'
					#print('YESSSSAAA')
				elif name == 'scuola elementare carlo collodi':
					self.FIN.at[index,'category']= 'admin_chores'
				elif name in hand_list:
					self.FIN.at[index,'category']= hand_dict[name]
				elif name.startswith('via') or name.startswith('corso') or name.startswith('v.'): 
					self.FIN.at[index,'category']= 'NONE'
				elif name.startswith('fermata') : 
					self.FIN.at[index,'category']= 'commuting'
				elif name == 'casa bimbo tagesmutter' and user_id.startswith ('elena_merola_'):
					self.FIN.at[index,'category']= 'home'


				else:
					

					possible_category = self.poi_search(sez_id,name)
					if possible_category =='nan':
						cnt+=1
					def_category = self.definitive_category(possible_category,user_index)
					if def_category == 'nan':
						print('============')
						print ('USER:',user_id, 'position: ',index)
						print(name)
						def_category = input('which category:')
						hand_list.append(name)
						hand_dict[name] = def_category
						print('============')
					self.FIN.at[index,'category']= def_category
			

			else:
				continue

		print('no labeled data at this point: ',cnt,' over tot: ',tot)
		print('percentage of non-labeled data: ',(cnt/tot*100),'%')
		self.FIN.to_excel(self.output_folder+'final_Dataset.xlsx')


	def poi_search(self,sez_id,name):

		index_list = self.POI[(self.POI['SEZCENS'] == sez_id) & (self.POI['Name'] == name)].index.tolist()
		
		if len(index_list)>0:
	
			category = self.POI.at[index_list[0],'Types']
		
			#print(category)
			return category
		else:
		
			index_list = self.POI[(self.POI['Name'] == name) ].index.tolist()
			if len(index_list)>0:
				category = self.POI.at[index_list[0],'Types']
				
				return category
			else:
		
				print('NotFound')
				return 'nan'

	def definitive_category(self,cat,user_index):
		old_cat = cat
		if old_cat == 'education' and self.US.at[user_index[0],'occupation']== 'worker':
			new_cat = 'admin_chores'

		else:
			new_cat=old_cat
		return new_cat


if __name__ == '__main__':

	US_dataset = 'OUTPUT/final_user/UserDataset.xlsx'
	MO_dataset = 'OUTPUT/final_mobility/MobilityDataset.xlsx'
	TE_dataset = 'final_territorial/...'
	output_folder = 'OUTPUT/FINAL_DATASET/'
	POI_dataset = 'OUTPUT/final_territorial/all_POI.xlsx'

	#STEP1 SYSTEMATIC MOBILITY
	#Sis = Systematic_mobility(MO_dataset,US_dataset,output_folder)

	N_Sis = Non_systematic_mobility(output_folder+'final_Dataset.xlsx',US_dataset,POI_dataset,output_folder)







