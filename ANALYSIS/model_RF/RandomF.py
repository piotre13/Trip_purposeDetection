import pandas as pd
import numpy as np
import json
import datetime
from datetime import timedelta
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import collections as clt
import seaborn as sn
from pprint import pprint
import timeit
from sklearn.tree import export_graphviz
from subprocess import call, check_call
import pydotplus
np.random.seed(0)

print(sklearn.__version__)



class Preporcessing:

	def __init__(self,data,territorial,users):

		self.df = pd.read_excel(data)
		#print(self.df.keys())
		self.terr = pd.read_excel(territorial)
		self.terr['SEZCENS'] = self.terr['SEZCENS'].astype(str)
		#print(self.terr.keys())
		self.us = pd.read_excel(users)
		self.timeslots = [['0:0:0','3:0:0'],['3:0:0','6:0:0'],['6:0:0','9:0:0'],['9:0:0','12:0:0'],['12:0:0','15:0:0'],['15:0:0','18:0:0'],['18:0:0','21:0:0'],['21:0:0','23:59:59']]
		for slot in self.timeslots:
			slot[0] = datetime.datetime.strptime(slot[0], '%H:%M:%S').time()
			slot[1] = datetime.datetime.strptime(slot[1], '%H:%M:%S').time()
		#print(self.us.keys())
	
	def datasets(self):
		#adding the data from users and territorial to each trips
		#in the territortial merging I'm only merge destinations sez cens
		df = pd.merge(self.df,self.terr,left_on ='d_census_id', right_on = 'SEZCENS')
		df = pd.merge(df,self.us,left_on='user_id',right_on='user_id')
		df = df.loc[:,~df.columns.str.startswith('Unn')]
		df = df.replace(['n','s','w','e','nw','ne','sw','se'],['100000','200000','300000','400000','130000','140000','240000','230000'])

		#print(df.shape)
		
		#dropping nan and NONE values AND useless columns
		df = df[df.category!= 'NONE']
		df = df.dropna(subset= ['category'])
		#print(df.columns)
		df = df.drop(columns=['_id','SEZCENS','NCIRCO','ZONASTAT','SUPERF','AREA_CENS','CODASC'])
		df = df.fillna(df.mean())
		#pd.set_option('display.max_rows', None)
		#print(df.isna().sum())
	
		#set a column with bbolean values 0 for weekedn 1 for weekday
		print (df['o_datetime'][0])
		#df['o_datetime']= df['o_datetime'].apply(lambda x: x+timedelta(hour=1))
		#df['d_datetime']= df['d_datetime'].apply(lambda x: x+timedelta(hour=1))
		print (df['o_datetime'][0])
		
		df['o_datetime'] = pd.to_datetime(df['o_datetime'], format="%Y-%m-%d %H:%M:%S")
		df['d_datetime'] = pd.to_datetime(df['d_datetime'], format="%Y-%m-%d %H:%M:%S")
		df['WEEKDAY'] = np.where((df['o_datetime'].dt.dayofweek) < 5,1,0)

		#creating a feature for time slots in a day (range of the slot 3 hours)
		df['t_slot'] = df['d_datetime'].dt.time.apply(lambda x: self.timeslot(x))
		
		#factorize the category
		df['category'],cat_labels = df['category'].factorize()
		df['mode'],mode_labels = df['mode'].factorize()
		df['occupation'],occupation_labels = df['occupation'].factorize()

		#split the data into 80% train and 20% test
		train,test,y_train,y_test = train_test_split(df.drop(['category','address_dest','name','user_id','o_datetime','d_datetime'],axis='columns'),df['category'], test_size=0.2)
		

		return train,test,y_train,y_test,list(cat_labels),list(mode_labels),list(occupation_labels)


	def timeslot(self,x):
		for slot in self.timeslots:
			if slot[0]<=x<slot[1]:
				return self.timeslots.index(slot)




class Tools:
	def __init__(self):
		pass
	def report(self,results, n_top=3):
		for i in range(1, n_top + 1):
			candidates = np.flatnonzero(results['rank_test_score'] == i)
			for candidate in candidates:
				print('=================================')
				print("Model with rank: {0}".format(i))
				print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
				print("Parameters: {0}".format(results['params'][candidate]))
				print("Training time: {0}".format(results['mean_fit_time'][candidate]))
				print("")
				print('=================================')


	def plot_random_res(self,results):

		best_pos = list(results['rank_test_score']).index(min(results['rank_test_score'])) 
		worst_pos = list(results['rank_test_score']).index(max(results['rank_test_score']))
		lables = ['best_model','worst_model'] 
		best_values =[]
		worst_values =[]
		width = 0.20
		width1= 0.3
		ind = np.arange(2)


		for i in ['mean_train_score','mean_test_score','mean_fit_time']:
			best_values.append(results[i][best_pos])
			worst_values.append(results[i][worst_pos])

		
		fig,ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 2]})

		rect1 = ax[0].bar(ind, [best_values[0],worst_values[0]], width, color='g',label= 'train')
		rect2 = ax[0].bar(ind+width, [best_values[1],worst_values[1]], width, color='b',label= 'test')
		ax[0].set_ylabel('Mean accurancy value')
		ax[0].set_xticks(np.arange(2)+width/2)
		ax[0].set_xticklabels( ('best model', 'worst model') )
		ax[0].legend(loc='lower center')
		for rect in rect1:
			height = rect.get_height()
			ax[0].text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % (height*100)+'%', ha='center', va='bottom',fontsize=7)
		for rect in rect2:
			height = rect.get_height()
			ax[0].text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % (height*100)+'%', ha='center', va='bottom',fontsize=7)
		

		rect3 = ax[1].bar(ind, [best_values[2],worst_values[2]], width1, color='r',label= 'time')
		ax[1].set_xticks(np.arange(2))
		ax[1].set_xticklabels( ('best model', 'worst model') )
		ax[1].set_ylabel('fit time [s]')
		for rect in rect3:
			height = rect.get_height()
			ax[1].text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % height, ha='center', va='bottom',fontsize=7)

		
		fig.suptitle('Comparison bestVSworst after RandomizedSearchCV', fontsize=16)
		fig.tight_layout()
		fig.subplots_adjust(top=0.88)
		plt.savefig('model_comparison.png')
		plt.close()

	def plot_feature_impo(self,feat_importances):

		fig = plt.figure()
		feat_importances.nlargest(20).plot(kind='barh')
		#feat_importances.plot(kind='bar')
		plt.title('20 most important features')
		fig.tight_layout()
		plt.savefig('features_importances.png')
		plt.close()

	def confusion_matrix(self,y_test,y_predicted, labels):
		
		y_test = list(y_test.to_numpy())
		y_predicted = list(y_predicted)
		labels = list(labels)

		fig = plt.figure(figsize= (10,10))
		cm = confusion_matrix(y_test,y_predicted, normalize= 'true')
		ax = plt.subplot()
		sn.heatmap(cm, annot=True, ax = ax, fmt='.2f',cmap="coolwarm") #annot=True to annotate cells

		# labels, title and ticks
		ax.set_xlabel('Predicted labels')
		ax.set_ylabel('True labels') 
		ax.set_title('Confusion Matrix') 
		ax.xaxis.set_ticklabels(labels) 
		ax.yaxis.set_ticklabels(labels)
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		 rotation_mode="anchor")
		plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
		 rotation_mode="anchor")
		fig.tight_layout()
		plt.savefig('confusion_matrix.png')
		plt.close()

	

	def plot_grid_res_2(self,results):
		fig, ax1 = plt.subplots()
		color = 'tab:red'
		ax1.set_xlabel('number of estimators')
		ax1.set_ylabel('mean test score', color=color)
		ax1.plot(results['mean_test_score'], color=color,label='test_score')
		#ax1.plot(train_score.index, data_3, color='g',label='train_score')
		ax1.legend(loc='lower right', bbox_to_anchor=(0.99, 0.06))

		ax1.tick_params(axis='y', labelcolor=color)

		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

		color = 'tab:blue'
		ax2.set_ylabel('training time [s]', color=color)  # we already handled the x-label with ax1
		ax2.plot(results['mean_fit_time'], color=color,label='training_time')
		ax2.tick_params(axis='y', labelcolor=color)
		plt.legend(loc ='lower right')
		fig.tight_layout()  # otherwise the right y-label is slightly clipped
		plt.savefig('n_estimators_plot.png')
		plt.close()

	def report_final(self,results,estimators):

		estimators= list(estimators)
		list_best_score = []
		list_best_est = []
		for i in range(3):
			best_pos = list(results['mean_test_score']).index(max(results['mean_test_score'])) 
			best_est = estimators[best_pos]
			list_best_score.append(max(results['mean_test_score']))
			list_best_est.append(best_est)
			results['mean_test_score'].pop(best_pos)
			estimators.pop(best_pos)
		#print(list_best_score)
		#print(list_best_est)
		found = False
		while not found:
			pos = list_best_est.index(min(list_best_est))
			if ( max(list_best_score)-list_best_score[pos])<0.009:
				found = True
				return list_best_est[pos]
		
if __name__ == '__main__':

	#data_path = '/Users/pietrorandomazzarino/Documents/UNIVERSITA/interdisciplinary project/TripPurposeDetection_Project/PROCESSING_1/OUTPUT/FINAL_DATASET/final_Dataset_manual.xlsx'
	data_path = 'df.xlsx'
	territorial_path ='/Users/pietrorandomazzarino/Documents/UNIVERSITA/interdisciplinary project/TripPurposeDetection_Project/PROCESSING_1/OUTPUT/final_territorial/FINAL_territorial_new.xlsx'
	user_path = '/Users/pietrorandomazzarino/Documents/UNIVERSITA/interdisciplinary project/TripPurposeDetection_Project/PROCESSING_1/OUTPUT/final_user/UserDataset.xlsx'
	
	Tools = Tools() # instantiation of the tools class

	#PREPROCESSING STEPS, the data are manipulated to be fitted from Random forest algorithm
	data = pd.read_excel(data_path)
	data = data.replace(['helth','admni_chores'],['health','admin_chores'])
	#y_data,y_labels = data['alpha_category'].factorize()
	data['alpha_a_time'] = data['alpha_a_time'].factorize()[0]
	data['alpha_category'],y_labels = data['alpha_category'].factorize()
	
	X_data = data.drop(columns=['alpha_category','category'])
	y_data = data['alpha_category']
	print(data['alpha_category'].unique())
	print(list(y_labels))


	# ****************************************************
	##RANDOMIZED SEARCH CV FOR PARAMETER TUNING
	##Random Forest hyper-parameter Tuning
	# model = RandomForestClassifier()

	# estimators = np.arange(10,1000,1)
	# impurity = ['gini','entropy']
	# max_features = ['sqrt','log2',None]
	# bootstrap = [True,False] 
	# #preparation of the random grid
	# param_grid = { 'bootstrap': bootstrap,
	# 				'max_features':max_features,
	# 				'criterion':impurity,
	# 				'n_estimators':estimators}

	# rf_random = RandomizedSearchCV(estimator = model, param_distributions = param_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1,return_train_score=True)
	# random_search = rf_random.fit(X_data,y_data)
	# random_results = random_search.cv_results_
	# random_best_params = random_search.best_params_
	# best_model = random_search.best_estimator_ #this is the best model
	# random_results_pd = pd.DataFrame(random_results)
	# random_results_pd.to_excel('Random_results.xlsx')

	# np.save('results.npy', random_results)#for practice porpuses comment when final running
	# ****************************************************


	X_train,X_test,y_train,y_test = train_test_split(data.drop(['category','alpha_category','bin_category'],axis='columns'),data['alpha_category'], test_size=0.2)
	
	
	random_results = np.load('results.npy',allow_pickle='TRUE').item()#for practice porpuses comment when final running
	print('RANDOMIZED SEARCH RESULTS:\n')
	Tools.report(random_results)
	Tools.plot_random_res(random_results)

	# ****************************************************
	#RUNNING THE CODE BY STEP USING SAVED DATA INSTANTIATING THE MODEL WITH SAVED PARAMETERES
	#print(random_results['params'])
	best_pos = list(random_results['rank_test_score']).index(min(random_results['rank_test_score'])) 
	best_params = random_results['params'][best_pos]
	print('best paramas from Randomized:\n',best_params)
	#initialize the best model with the best parameters
	best_model = RandomForestClassifier(n_estimators=best_params['n_estimators'],criterion=best_params['criterion'],max_features=best_params['max_features'], bootstrap = best_params['bootstrap']) # comment when real running

	# ****************************************************
	#MANUAL GRID SEARCH

	n_estimators = np.arange(1,500,1)
	# grid_results ={}
	# grid_results['mean_test_score'] =[]
	# grid_results['mean_fit_time']=[]

	# for i in n_estimators:
	# 	best_model = RandomForestClassifier(n_estimators=i,criterion=best_params['criterion'],max_features=best_params['max_features'], bootstrap = best_params['bootstrap']) # comment when real running
	# 	start = timeit.default_timer()
	# 	best_model.fit(X_train,y_train)
	# 	stop = timeit.default_timer()
	# 	grid_results['mean_test_score'].append(best_model.score(X_test,y_test))
	# 	grid_results['mean_fit_time'].append(stop-start)
	# 	print ('model fitted with number of estimators: ',i)
	# 	print('test score: ',best_model.score(X_test,y_test))
	# 	print('training time: ',(stop-start))
	# 	print('====================')
	# 	print('')
	# np.save('grid_results_2.npy', grid_results)#for practice porpuses comment when final running
	# exit()
	
	results_grid = np.load('grid_results_2.npy',allow_pickle='TRUE').item()
	Tools.plot_grid_res_2(results_grid)
	n_est_final = Tools.report_final(results_grid, n_estimators)

	print('FINAL STEP FITTING AND RUNNING THE MODEL WITH BEST CONFIGURATION:')
	print('n_estimators: ',n_est_final)
	print('criterion: ',best_params['criterion'])
	print('max features: ', best_params['max_features'])
	print('bootstrap: ',best_params['bootstrap'])
	print('')


#************ TO BE USED WHEN RESULTS OF PARAMETER TUNING ARE SAVED 
	best_model = RandomForestClassifier(n_estimators=n_est_final,criterion=best_params['criterion'],max_features=best_params['max_features'], bootstrap = best_params['bootstrap']) # comment when real running
	best_model.fit(X_train,y_train)#FITTING THE BEST MODEL

	# estimator_plot = best_model.estimators_[10]
	# dot_data = export_graphviz(estimator_plot, 
 #                out_file=None, 
 #                feature_names = X_train.columns,
 #                class_names = y_labels,
 #                rounded = True, proportion = False, 
 #                precision = 2, filled = True)
	# #call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
	# graph = pydotplus.graph_from_dot_data(dot_data)
	# graph.write_png('tree.png')
	#check_call(['dot','-Tpng','InputFile.dot','-o','OutputFile.png'])

	feat_importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
	Tools.plot_feature_impo(feat_importances)
	print(best_model.score(X_test,y_test))


	#prediction on the test
	y_predicted = best_model.predict(X_test)
	Tools.confusion_matrix(y_test,y_predicted,y_labels)
	
