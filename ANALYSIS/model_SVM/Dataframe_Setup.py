import sklearn
import datetime
import matplotlib.pyplot as plt
import collections as clt
import seaborn as sns
import pandas as pd
import numpy as np
from time import time 
from pprint import pprint
from sklearn import preprocessing, linear_model
from datetime import timedelta
from statistics import stdev
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from random import randint

class DatasetManager:

	def __init__(self, data, territorial, users):
		self.df = pd.read_excel(data)
		self.terr = pd.read_excel(territorial)
		self.terr['SEZCENS'] = self.terr['SEZCENS'].astype(str)
		self.us = pd.read_excel(users)
		self.geo_categories = ['home', 'work', 'eating', 'entertainment', 'recreation', 'shopping', 'travel', 'admin_chores', 'religious', 'health', 'police', 'education']
		self.pop_categories = ["P_TOT", "MALE_TOT", "FEM_TOT", "age 0-9", "age 10-24", "age 25-39", "age 40-64", "age >65", "male 0-9", "male 10-24", "male 25-39", "male 40-64", "male >65"]
		self.stats = ['P47','P48','P49', 'P50','P51', 'P52','P61','P62','P128','P130','P131','P135','P137','P138','ST1','ST3','ST4','ST5','ST9','ST10','ST11','ST12','ST13','PF1','PF2','PF3','PF4','PF5','PF6','PF7','PF8','PF9']
		self.list_cat = [
			'category',
			'work', 
			'mode',
			'd_hour',
			'activity_time'
			]
		self.dur_dict = {
						0:'~1[h]',#900,
						1:'1-3[h]',#2*3600, 
						2:'3-5[h]',#4*3600, 
						3:'5-8[h]',#6.5*3600, 
						4:'8-12[h]',#10*3600, 
						5:'12+[h]',#13*3600
						 }

	def df_extractor(self):
		#adding the data from users and territorial to each trips
		#in the territortial merging I'm only merge destinations sez cens
		df = pd.merge(self.df, self.terr,left_on='d_census_id', right_on='SEZCENS')
		df = pd.merge(df,self.us, left_on='user_id', right_on='user_id')
		df = df.loc[:,~df.columns.str.startswith('Unn')]
		df = df.replace(['helth','admni_chores'],['health','admin_chores'])
		df = df.replace(['n','s','w','e','nw','ne','sw','se'],['100000','200000','300000','400000','130000','140000','240000','230000'])

		#dropping nan and NONE values AND useless columns
		df = df[df.category != 'NONE']
		df = df.dropna(subset=['category'])
		df = df.drop(columns=['_id','SEZCENS','NCIRCO','ZONASTAT','SUPERF','AREA_CENS','CODASC'])
		df = df.fillna(df.mean())
		#pd.set_option('display.max_rows', None)
		#print(df.isna().sum())
	
		# set a column with bbolean values 0 for weekedn 1 for weekday
		df['o_datetime'] = pd.to_datetime(df['o_datetime'], format="%Y-%m-%d %H:%M:%S")
		df['d_datetime'] = pd.to_datetime(df['d_datetime'], format="%Y-%m-%d %H:%M:%S")
		df['o_datetime'] = df['o_datetime'].apply(lambda x: x+timedelta(hours=1))
		df['d_datetime'] = df['d_datetime'].apply(lambda x: x+timedelta(hours=1))
		df['activity_time_h'] = df['activity_time'].apply(lambda x: x/3600)
		# df['o_hour'] = df['o_datetime'].apply(lambda x: (x.hour)//3)
		df['d_hour'] = df['d_datetime'].apply(lambda x: (x.hour)//8)
		
		# self.hist_plot(df[['activity_time','category']])
		
		df['cat_activity_time'] = df['activity_time'].apply(lambda x: self.dur_converter(x))
		df['bin_weekday'] = np.where((df['o_datetime'].dt.dayofweek) < 5, 1, 0)
		df['bin_category'] = np.where((df['category'] == 'work') | (df['category'] == 'home'), 1, 0)

		#factorize the category
		df['cat_category'], cat_labels = df['category'].factorize()
		# df['alpha_category'] = df['category'].apply(lambda x: cat_labels[x])
		# df['alpha_a_time'] = df['activity_time'].apply(lambda x: self.dur_dict[x])
		# labels, sizes = [], []
		# for i in df['mode'].unique():
		# 	if i != 'running' and i != 'flying':
		# 		sum_ = df[df["mode"]==i]['mode'].count()
		# 		labels.append(i)
		# 		sizes.append(sum_)
		# 		print(i, sum_)
		# patches, texts = plt.pie(sizes, labels=labels, autopct=None )
		# # plt.legend(patches, labels, loc="best")
		# plt.axis('equal')
		# plt.tight_layout()
		# # plt.show()
		df['mode'], mode_labels = df['mode'].factorize()
		df['occupation'], occupation_labels = df['occupation'].factorize()

		df = df.drop(['address_dest','name','user_id','o_datetime','d_datetime','d_census_id','d_lat','d_lng','d_ts','o_census_id','o_lat','o_lng','o_ts'],axis='columns')
		df.to_excel(r'df.xlsx', index=False, header=True)
		return df, cat_labels

	def hist_plot(self, A):
		print(A)
		A = A[A.activity_time_h <= 15]
		# print(A)
		# means, names, tmp, a_times = [], [], [], []
		# for cat in A['category'].unique():
		# 	if cat != 
		# 	names.append(cat)
		# 	mask = A['category'] == cat
		# 	means.append(A[mask]['activity_time'].mean()/3600)
		# 	tmp = A[mask]['activity_time']
		# 	a_times.append([x / 60 for x in tmp])
		# font_size = 15
		# plt.rc('font', size=font_size)          # controls default text sizes
		# plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
		# plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
		# plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
		# plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
		# plt.rc('legend', fontsize=font_size)    # legend fontsize
		# plt.rc('figure', titlesize=font_size)  # fontsize of the figure title
		# fig, ax = plt.subplots()
		# print (a_times)
		# for elem, i in zip(a_times, range(len(a_times))):
		# 	sn.distplot(elem, ax=ax, hist=False, kde=True, label=names[i])
		# ax.set_xlim([0,60*5])
		# # ax.set_title(cat)
		# fig.set_size_inches(10, 7)
		# plt.xlabel('Activity time [m]')
		# plt.ylabel('Probability density')
		# plt.title('Kernel Density Estimate of Activity Time')
		# plt.xticks(range(-10,300,10),rotation=45)
		# ax.grid()
		# plt.legend(ncol=2, handleheight=2)
		# fig.tight_layout()
		# fig.savefig("Kernel_Density.png")
		# plt.show()
		# exit()
		# plt.close()

		sns.set(font_scale=1.5)
		fig, ax = plt.subplots()
		fig.set_size_inches(10,27)
		plot = sns.catplot(x="category", y="activity_time_h", kind="box", data=A, height=7, aspect=10/7)
		plot.set_xticklabels(rotation=70)
		# plot.axes.axhline(y=10)
		# plt.axhline(y=10/60, linewidth=2, color='red')
		# plt.axhline(y=20/60, linewidth=2, color='red')
		# plt.axhline(y=0.5, linewidth=2, color='red')
		# plt.axhline(y=1, linewidth=2, color='red')
		# plt.axhline(y=2, linewidth=2, color='red')
		# plt.axhline(y=3, linewidth=2, color='red')
		# plt.axhline(y=8, linewidth=2, color='red')
		plot.set(ylabel="Activity Time [h]", xlabel="", ylim=(0, 12))
		plot.savefig("Mean_Density.png")
		# plt.show()
		exit()
		
	def dur_converter(self, a_time):
		if a_time<60*10:
			a_time = 0
		elif 60*10<=a_time<60*20:
			a_time = 1
		elif 60*20<=a_time<60*30:
			a_time = 2
		elif 60*30<=a_time<3600:
			a_time = 3
		elif 3600<=a_time<2*3600:
			a_time = 4
		elif 2*3600<=a_time<3*3600:
			a_time = 5
		elif 3*3600<=a_time<8*3600:
			a_time = 6
		elif a_time>=8*3600:
			a_time = 7
		return a_time

	def df_reader(self, name):
		df = pd.read_excel(io=name)
		return df

	def pre_processing(self, df, binary_class, save_file=False):
		self.list_sc = ['INCOMEinco']
		[self.list_sc.append(x) for x in self.geo_categories]
		[self.list_sc.append(x) for x in self.pop_categories]
		[self.list_sc.append(x) for x in self.stats]
		ct = ColumnTransformer([
			# ('onehot', preprocessing.OneHotEncoder(), self.list_cat),
			('scale', preprocessing.StandardScaler(), self.list_sc),
			# ('scale', preprocessing.QuantileTransformer(), slf.list_sc),
			# ('scale', preprocessing.RobustScaler(), self.list_sc),
			# ('scale', preprocessing.Normalizer(), self.list_sc),
			# ('scale', preprocessing.MinMaxScaler(feature_range=(-1, 1)), self.list_sc),
			])
		if binary_class == True:
			y = df['bin_category'] 
		else:
			y = df['category'] 
		ct.fit(df)
		temp = ct.transform(df)
		lst_temp = []
		# [lst_temp.append(x) for x in self.list_cat]
		[lst_temp.append(x) for x in self.list_sc]
		for elem,i in zip(lst_temp,range(len(lst_temp))):
			df[elem] = temp[:,i]
		if save_file == True:
			df.to_excel(r'df_preprocessed.xlsx', index=False, header=True)
		X = df.drop(['category','bin_category'], axis='columns')
		return X, y, ct

	def remove_cols(self, X, lst):
		for elem in lst:
			X = X.drop(elem, axis='columns')
		return X 
