import pandas as pd
from CONFIG import *

class Territorial:
	def __init__(self,input1,input2,input3,out):

		self.G_POI = pd.read_excel(input1)
		self.Q_POI = pd.read_excel(input2)
		self.Cens_zone = pd.read_csv(input3)
		self.out_folder = out
		self.preprocessing_POI() #decoment when test is over
		self.probability_assignment() #decomment when test is over
		self.FINAL_step()
	
	def preprocessing_POI(self):
	
		#changing the lists of categories with the main belonging one and savings attractors
		cnt=0
		attr_cnt = 0
		main_attactors =['museum','stadium','zoo','park','shopping_mall','airport','train_station','hospital','university']
		self.G_POI['attractor'] = ''
		self.Q_POI['attractor'] = ''

		for index, row in self.G_POI.iterrows():
			list_tipologies = self.str_to_list(row['Types'])
			if any(x in main_attactors for x in list_tipologies):
				attr_cnt+=1
				self.G_POI.at[index, 'attractor'] = 'YES'
			else:
				self.G_POI.at[index, 'attractor'] = 'NO'

			found = False
			k = 0
			while not found:
				for word in list_tipologies:
					if word in TYPES[categories[k]]:
						self.G_POI.at[index,'Types'] = categories[k]
						found = True
					if k >= 11:
						self.G_POI.at[index,'Types'] = list_tipologies[0]
						found = True
				k+=1
			cnt+=1

		for index, row in self.Q_POI.iterrows():
			list_tipologies = self.str_to_list(row['Types'])
			if any(x in main_attactors for x in list_tipologies):
				attr_cnt+=1
				self.Q_POI.at[index, 'attractor'] = 'YES'
			else:
				self.Q_POI.at[index, 'attractor'] = 'NO'

			found = False
			k = 0
			while not found:
				for word in list_tipologies:

					if word in TYPES[categories[k]]:
						self.Q_POI.at[index,'Types'] = categories[k]
						found = True
					if k >= 11:
						self.Q_POI.at[index,'Types'] = list_tipologies[0]
						found = True
				k+=1
			cnt+=1
		print('===============')
		print('pre_processing method OUTPUT:')
		print('total number of POI analyzed:{}'.format(cnt))
		print('total number of attractors:{}'.format(attr_cnt))
		print('===============')

		self.union(self.G_POI,self.Q_POI)

					
	def str_to_list(self, string):
		new_str = string[1:-1]
		new_str = new_str.replace("'",'')
		new_str = new_str.replace(" ",'')
		words = new_str.split(',')
		return words

	def union(self,df1,df2):
		
		#print(df1.keys())
		#print(df2.keys())

		df1 = df1.drop(columns = ['ID','PlaceID','ViewportNE_Latitude','ViewportNE_Longitude','ViewportSW_Latitude','ViewportSW_Longitude'])
		df2 = df2.drop(columns = ['RADIUS'])

		self.FINAL_df = df1.append(df2)


		#print(self.FINAL_df)


	def probability_assignment(self):
		'''
		probability of use is assigned to each census zone making for each
		a 10% for both home and work has been ensured and for the census
		zone in wich no POI has been found a uniform distribution of probability has been assigned
		'''
		df = self.FINAL_df 
		df = df.dropna()
		SEZ_CENS = list(df['SEZCENS'])

		#INITIALIZATION 
		SEZ_CENS_fin = {}
		for sez in SEZ_CENS:
			if sez != 'NOT_found':
				SEZ_CENS_fin[str(int(sez))] = {}
				SEZ_CENS_fin[str(int(sez))]['attractor'] = ''
				for item in TYPES:
					SEZ_CENS_fin[str(int(sez))][item]= 0

		#counting tipologies per sex cens
		for index, row in df.iterrows():
			try:
				SEZ_CENS_fin[str(int(row['SEZCENS']))][row['Types']]+=1
				if row['attractor'] == 'YES':
					#print('yes')
					SEZ_CENS_fin[str(int(row['SEZCENS']))]['attractor'] = row['Types']
				else:
					SEZ_CENS_fin[str(int(row['SEZCENS']))]['attractor'] = 0


			except (KeyError, ValueError) as error:
				pass		

		print(SEZ_CENS_fin)
		#TRANSFORMING INTO PROBABILITY ADDING A FIXED 10% FOR HOME AND WORK
		for sez in SEZ_CENS_fin:

			if SEZ_CENS_fin[sez]['attractor'] != 0:
				att = SEZ_CENS_fin[sez]['attractor'] 
				del SEZ_CENS_fin[sez]['attractor'] 
				tot = sum(SEZ_CENS_fin[sez].values())
				if tot != 0:
					SEZ_CENS_fin[sez][att]+= (0.7*tot)/(1-0.7)
					tot = sum(SEZ_CENS_fin[sez].values())
					SEZ_CENS_fin[sez]['home']+= (0.1*tot)/(1-0.1)
					tot = sum(SEZ_CENS_fin[sez].values())
					SEZ_CENS_fin[sez]['work']+= (0.1*tot)/(1-0.1)
					new_tot = sum(SEZ_CENS_fin[sez].values())
					for i in SEZ_CENS_fin[sez]:
						SEZ_CENS_fin[sez][i]/= new_tot
				else:
					for i in SEZ_CENS_fin[sez]:
						SEZ_CENS_fin[sez][i] = (1 / 12)
			else:
				del SEZ_CENS_fin[sez]['attractor'] 
				tot = sum(SEZ_CENS_fin[sez].values())
				if tot != 0:
					SEZ_CENS_fin[sez]['home']+= (0.1*tot)/(1-0.1)
					tot = sum(SEZ_CENS_fin[sez].values())
					SEZ_CENS_fin[sez]['work']+= (0.1*tot)/(1-0.1)
					new_tot = sum(SEZ_CENS_fin[sez].values())
					for i in SEZ_CENS_fin[sez]:
						SEZ_CENS_fin[sez][i]/= new_tot
				else:
					for i in SEZ_CENS_fin[sez]:
						SEZ_CENS_fin[sez][i] = (1 / 12)


			prob_tot = sum(SEZ_CENS_fin[sez].values())
			print (prob_tot)	
				
		self.CENS_prob = SEZ_CENS_fin
				


	def FINAL_step(self):
		df = self.Cens_zone
		prob = self.CENS_prob
		#print (prob)
		print(len(prob))
		print(df.shape)
		#creating the column for the activities
		for cat in categories:
			df[cat] = ''
		controller = 0
		for index, row in df.iterrows():
			sez = str(int(row['SEZCENS']))
			
			for cat in categories:
				try:
					#print('right: ',controller)
					df.at[index,cat] = prob[sez][cat]
				except KeyError:
					#print('missing: ',controller)
					df.at[index,cat] = (1/12)
			controller+=1
		print(df)
		df.to_excel(self.out_folder+'FINAL_territorial_new.xlsx')
		df.to_csv(self.out_folder+'FINAL_territorial_new.csv')







if __name__ == '__main__':

	input1 = 'OUTPUT/G_POIs_sezioniCens.xlsx'
	input2 = 'OUTPUT/POI_other_sezioniCens.xlsx'
	input3 = 'OUTPUT/SEZ_cens.csv'
	out_folder = 'OUTPUT/final_territorial/'
	print('000000')
	TERRITORIAL = Territorial(input1,input2,input3,out_folder)
	print('11111')













