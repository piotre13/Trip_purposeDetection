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

class Classifiers():
	def __init__(self, X, y, RS):
		self.X = X
		self.y = y
		self.RS  = RS
		self.C_range = np.logspace(-2, 10, 13)
		self.gamma_range = np.logspace(-9, 3, 13)

	def SVM_classifier(self, c, gamma, labels):
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=self.RS)
		svm = SVC(
			C=c, 
			gamma=gamma,
			kernel='rbf', 
			decision_function_shape='ovo',
			random_state=self.RS)
		svm = svm.fit(X_train, y_train)
		y_predicted = svm.predict(X_test)
		# tr_res = round(metrics.accuracy_score(y_train, svm.predict(X_train))*100,2)
		# print("The SVM Train accuracy is:", tr_res, "[%]")
		self.cm = confusion_matrix(y_test, y_predicted, normalize='true')
		self.cm = pd.DataFrame(self.cm, index=labels, columns=labels)
		res = round(accuracy_score(y_test, y_predicted)*100,2)
		print("The SVM Test accuracy is:", res, "[%]")
		return res


	def grid_search(self):
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=self.RS)
		results, times = [], []
		cnt,best=0,[0, None, None]
		tot = len(self.C_range)*len(self.gamma_range)
		for c in self.C_range:
			tmp,tmp2=[],[]
			for gamma in self.gamma_range:
				cnt+=1
				t = time()
				clf = SVC(C=c, gamma=gamma, kernel='rbf',decision_function_shape='ovo',random_state=self.RS)
				clf = clf.fit(X_train, y_train)
				y_predicted = clf.predict(X_test)
				res = round(accuracy_score(y_test, y_predicted)*100,2)
				if res > best[0]:
					best = [res, c, gamma]
				t, t_0 = timer(str(cnt)+'/'+str(tot)+': SVM with C=%.2f & gamma=%f: result=%.2f'%(c,gamma,res), t, report_time=True)
				tmp.append(res)
				tmp2.append(t_0)
			times.append(tmp2)
			results.append(tmp)
		plt.figure()
		x_tick, y_tick = [], []
		[x_tick.append('{:.1e}'.format(x)) for x in self.C_range]
		[y_tick.append('{:.1e}'.format(x)) for x in self.gamma_range]
		hm = sn.heatmap(results, annot=True, xticklabels=x_tick, yticklabels=y_tick)
		np.save('results.npy', results)
		np.save('times.npy', times)
		plt.tight_layout()
		plt.show()
		hm.figure.savefig('grid_search.png')
		return best[1], best[2]

	def confusion_plot(self):
		plt.figure(figsize=(10,10))
		sn.heatmap(self.cm, annot=True, fmt='.2f', cmap='coolwarm')
		# plt.xlabel('Predicted')
		# plt.ylabel('Truth')
		plt.tight_layout()
		plt.savefig('confusion_plot.png')
		plt.show()

	def harry_plotter(self):
		results = np.load('results.npy')
		times = np.load('times.npy')
		colors = []
		[colors.append('#%06X' % randint(0, 0xFFFFFF)) for x in range(2*len(self.C_range))]
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		# ax1.set_title(title)
		pprint(results)
		for i in range(len(self.C_range)):
			print(results[:,i])
			ax2.plot(self.C_range, results[i,:], linestyle='--', marker='*', label=self.C_range[i], color=colors[i])
			ax1.plot(self.C_range, times[i,:], linestyle='--', marker='*', color=colors[i+len(self.C_range)])
		ax1.set_xlabel('C parameter')
		ax2.set_ylabel('Accuracy [%]', color=colors[i])
		ax1.set_ylabel('Execution Time [s]', color=colors[i+len(self.C_range)])
		plt.legend()
		plt.show()

		fig, ax3 = plt.subplots()
		ax4 = ax3.twinx()
		for j in range(len(self.gamma_range)):
			ax4.plot(self.gamma_range, results[j,:], linestyle='--', marker='*',label=self.gamma_range[j], color=colors[j])
			ax3.plot(self.gamma_range, times[j,:], linestyle='--', marker='*', color=colors[j+len(self.C_range)])
		ax4.set_xlabel('Gamma parameter')
		ax4.set_ylabel('Accuracy [%]', color=colors[j])
		ax3.set_ylabel('Execution Time [s]',color=colors[j+len(self.C_range)])
		plt.legend()
		plt.show()
		# fig.savefig(title+'.png')
		# for j in range(len(self.gamma_range)):

if __name__ == '__main__':
	def timer(name, prev_time, report_time=False):
		curr_time = (time()-prev_time)
		exec_time = str(timedelta(seconds=curr_time)).split(".")[0]
		print(name, "executed in:", exec_time)
		print('****************************************************************\n')
		if report_time:
			return time(), curr_time
		else:
			return time()

	t = time()
	print("Started at:", str(timedelta(seconds=t+7200)).split(".")[0].split(",")[1])

	data_path = 'final_Dataset_manual.xlsx'
	territorial_path ='FINAL_territorial_new.xlsx'
	user_path = 'UserDataset.xlsx'
	
	DM = DatasetManager(data_path,territorial_path,user_path)
	t = timer("Initialization", t)

	df, cat_labels = DM.df_extractor()
	DM.hist_plot(df)
	# df = DM.df_reader('df.xlsx')
	t = timer("Extraction of dataframe", t)
	
	X, y, ct = DM.pre_processing(df, binary_class=False, save_file=True)
	# X = DM.remove_cols(X, ['o_hour', 'alpha_category', 'alpha_a_time'])
	t = timer("Pre-processing", t)

	C = Classifiers(X, y, 666)
	# # c_best, gamma_best = C.grid_search()
	# # print(c_best, gamma_best)
	# # C.harry_plotter()
	# #C=100.00 & gamma=0.0046
	C.SVM_classifier(c=100, gamma=0.01, labels=cat_labels)
	# # C.SVM_classifier(c=c_best, gamma=gamma_best, labels=cat_labels)
	# t = timer("SVM classifier", t)
	C.confusion_plot()
	# C.harry_plotter(X, y, 'SVM_new')

	# BINARY
	"""
	X1, y1, ct = DM.pre_processing(df, binary_class=True, save_file=True)
	# X = DM.remove_cols(X, ['o_hour', 'alpha_category', 'alpha_a_time'])
	t = timer("Pre-processing for binary class", t)

	C = Classifiers(X1, y1, 666)
	C.SVM_classifier(c=100, gamma=0.01, labels=cat_labels)
	t = timer("SVM classifier (binary)", t)
	"""
	print("Ended at:", str(timedelta(seconds=time()+7200)).split(".")[0].split(",")[1])