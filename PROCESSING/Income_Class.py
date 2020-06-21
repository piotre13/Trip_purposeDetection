import pandas as pd
import numpy as np

class Income:
	def __init__(self,data_folder, output_folder):
		self.data_folder = data_folder
		self.output_folder = output_folder
			
	
	def pre_processing(self,data):
		#deleting data not involving Torino
		df = pd.read_excel(data)
		print (df.shape)
		df = df.drop(df[(df.q3_district != 'Torino') & (df.q4_district != 'Torino')].index)
		print(df.shape)
		df = df.dropna()
		
		for index, row in df.iterrows():

			df.loc[index,'q56'] = int(df.loc[index,'q56'])
			if  df.loc[index,'q56']== 1:
				df.loc[index,'Avg_Income']= 1250.5
			elif  df.loc[index,'q56']== 2:
				df.loc[index,'Avg_Income']= 1250.5
			elif  df.loc[index,'q56']== 3:
				df.loc[index,'Avg_Income']= 1750.5
			elif  df.loc[index,'q56']== 4:
				df.loc[index,'Avg_Income']= 2250.5
			elif  df.loc[index,'q56']== 5:
				df.loc[index,'Avg_Income']= 2750.5
			elif  df.loc[index,'q56']== 6:
				df.loc[index,'Avg_Income']= 3250.5
			elif  df.loc[index,'q56']== 7:
				df.loc[index,'Avg_Income']= 3750.5
			elif  df.loc[index,'q56']== 8:
				df.loc[index,'Avg_Income']= 4250.5
			elif  df.loc[index,'q56']== 9:
				df.loc[index,'Avg_Income']= 5500.5
			elif  df.loc[index,'q56']== 10:
				df.loc[index,'Avg_Income']= 6500.5
			elif  df.loc[index,'q56']== 11:
				df.loc[index,'Avg_Income']= 7500.5
			elif  df.loc[index,'q56']== 12:
				df.loc[index,'Avg_Income']= 8500.5
			elif  df.loc[index,'q56']== 13:
				df.loc[index,'Avg_Income']= 9500.5
			elif  df.loc[index,'q56']== 14:
				df.loc[index,'Avg_Income']= 12500.5
			elif  df.loc[index,'q56']== 15:
				df.loc[index,'Avg_Income']= 15000

		print (df.shape)
		print (df.keys())
		df.to_excel(self.data_folder+'pronello/mobility_Pronello.xlsx')
		df.to_csv(self.data_folder+'pronello/mobility_Pronello.csv')

	
	def income_spread(self,origin,destination):
		
		df_Or = pd.read_csv(origin) #dataset of origins
		df_Des = pd.read_csv(destination) #dataset of destinations

		#creation of list of unique ID for stat_zone
		list1 = list(df_Or.ZONASTAT.unique())
		list2 = list(df_Des.ZONASTAT.unique())
		ZONE_stat = sorted(np.unique(list1+list2))  

		count = {}

		#INITIALIZING A DICT count[zone(x92)][income_range(x15)]
		for zone in ZONE_stat:
			zone = str(zone)
			count[zone] = {}
			for i in range(15):
				count[zone][str(i+1)] = 0

		#COUNTING OCCURRENCIES OF INCOME RANGES IN EACH ZONE FOR BOTH DESTINATION AND ORIGIN		
		for index, row in df_Or.iterrows():
			count[str(row['ZONASTAT'])][str(row['q56'])]+= 0.7
		for index, row in df_Des.iterrows():
			count[str(row['ZONASTAT'])][str(row['q56'])]+= 0.3

		income_ranges = {
						'1': 1.000,
						'2': 1250.5,
						'3': 1750.5,
						'4': 2250.5,
						'5': 2750.5,
						'6': 3250.5,
						'7': 3750.5,
						'8': 4250.5,
						'9': 5500.5,
						'10': 6500.5,
						'11': 7500.5,
						'12': 8500.5,
						'13': 9500.5,
						'14': 12500.5,
						'15': 15000
			}
		tot = sum(income_ranges.values())
		print(tot)
		d =[]
		for zone in ZONE_stat:
			samples = sum(count[zone].values())
			tot = 0
			for i in income_ranges:
				tot += count[zone][i]* income_ranges[i]
			avg_income = tot / samples
			count[zone]['zone_income'] = avg_income
			d.append([zone,count[zone]['zone_income']])
		
		dataframe = pd.DataFrame(d,columns=['ZONESTAT','income'])
		print (dataframe)
		dataframe.to_csv(self.output_folder+'ZONESTAT_income.csv')
		dataframe.to_excel(self.output_folder+'ZONESTAT_income.xlsx')





		#EXTRACTING THE TOTAL AND MAKING PERCENTAGE and aasigning the value of weighted income
		


