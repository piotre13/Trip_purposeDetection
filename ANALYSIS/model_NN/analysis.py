import json
from pathlib import Path
import numpy as np
import pandas as pd
from pprint import pprint
import datetime
from pytz import timezone
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import random as random_tf
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import keras.backend as k
from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import LSTM, Dense, GRU, Concatenate, Dropout, LeakyReLU, PReLU, ReLU
from keras.engine.input_layer import Input
from keras.utils import plot_model
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import random
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
import traceback
import scipy.stats as stats

np.random.seed(69)
random.seed(69)
random_tf.set_seed(69)

class NeuralNetwork:
	def __init__(self):
		trips = 'data/trips_dataset.xlsx'
		users = 'data/users.xlsx'
		terr = 'data/territorial.xlsx'
		to_label = 'data/ToLabel.xlsx'

		self.categories = {
						'shopping': 4, 'health': 6, 'police': 11, 'helth': 6, 'home': 0, 'NONE': 15, 
						'work': 1, 'entertainment': 8, 'commuting': 9, 'recreation': 7, 'admin_chores': 10, 
						'nan': 13, 'admni_chores': 10, 'religious': 12, 'education': 3, 'eating': 2, 'travel': 5
						}

		self.territorial_features = ['INCOME', 'home', 'work', 'eating', 'entertainment', 'recreation', 
			'shopping', 'travel', 'admin_chores', 'religious', 'health', 'police', 'education', 'P_TOT', 'age 25-39', 'age 40-64', 'age >65', 'age 10-24', 'P47', 'P48', 'P49','P61','P62']
		self.trip_features = ['activity_time', 'category', 'mode', 'o_datetime', 'd_datetime']
		
		self.occup = {
			'student':0,
			'worker':1,
			'retired':2
		}

		data, target = self.preparation(trips, terr, users, changes=0, binary=0)
		self.categorical_model(data, target, train=1)
		self.ssl(to_label)

	def binarization_apply(self, row):
		bin_cat = ['home', 'work', 'education']
		if row['category_label'] in bin_cat:
			row['category'] = 0
		else:
			row['category'] = 1

		return row

	def preparation(self, trips, territorial, users, changes=0, binary=0):
		if changes:
			trips = pd.read_excel(trips, index_col=0)
			users = pd.read_excel(users, index_col=4)
			users = users.drop(columns=['Unnamed: 0'])
			terr = pd.read_excel(territorial, index_col=1)
			terr = terr.drop(columns=['Unnamed: 0'])
			terr = terr.fillna(terr.mean())

			df = []
			test = []
			modes = trips['mode'].unique()
			modes = dict({(modes[i], i) for i in range(len(modes))})

			for row, col in trips.iterrows():
				user_id = col['user_id']
				tmp_obj = {}
				d_census_id = col['d_census_id']
				try:
					territorial_info = terr.loc[int(d_census_id)]
					user_info = users.loc[user_id]
					
					# for i in self.territorial_features:
					for i in terr.columns:
						tmp_obj[i] = territorial_info[i]
					for i in self.trip_features:
						tmp_obj[i] = col[i]
					for i in user_info.index:
						if i == 'Row':
							continue
						tmp_obj[i] = user_info[i]
					format_date = "%Y-%m-%d %H:%M:%S"
					o_d = datetime.datetime.strptime(tmp_obj['o_datetime'], format_date) + datetime.timedelta(hours=1)
					d_d = datetime.datetime.strptime(tmp_obj['d_datetime'], format_date) + datetime.timedelta(hours=1)
					tmp_obj['o_datetime'] = self.date_to_cat(o_d)
					tmp_obj['d_datetime'] = self.date_to_cat(d_d)
					tmp_obj['mode'] = modes[tmp_obj['mode']]
					if tmp_obj['category'] == 'helth':
						tmp_obj['category'] = 'health'
					elif tmp_obj['category'] == 'admni_chores':
						tmp_obj['category'] = 'admin_chores'
					tmp_obj['category_label'] = tmp_obj['category']
					tmp_obj['category'] = self.categories[tmp_obj['category']]
					tmp_obj['activity_time'] = self.activity_to_cat(tmp_obj['activity_time'])
					tmp_obj['occupation'] = self.occup[tmp_obj['occupation']]

					if tmp_obj['category_label'] == 'nan' or tmp_obj['category_label'] == 'NONE':
						test.append(tmp_obj)
					else:
						df.append(tmp_obj)
				except:
					# pass
					traceback.print_exc()

			self.df = pd.DataFrame(df)
			self.test = pd.DataFrame(test)

			self.df.to_excel('data/CompleteDataframe_AllTerritorial.xlsx', index=False)
			self.test.to_excel('data/ToLabel_AllTerritorial.xlsx', index=False)
		else:
			self.df = pd.read_excel('data/df.xlsx')
			self.to_label = pd.read_excel('data/ToLabel_AllTerritorial.xlsx')
			print('Dataset Loaded')

		self.df = self.df.sample(frac=1)
		lb = ['eating', 'entertainment', 'shopping', 'commuting', 'recreation', 
		'health', 'travel', 'home', 'work', 'education', 'religious', 'police', 'admin_chores']
		
		# for row, col in self.df.iterrows():
		# 	if col['category_label'] not in lb:
		# 		self.df = self.df.drop(row)

		if not binary:
			self.labels = {
					'shopping': 3, 'health': 5, 'home': 10, 
					'work': 9, 'entertainment': 1, 'commuting': 0, 'recreation': 7, 
					'education': 2, 'eating': 4, 'travel': 6, 'admin_chores':8, 'police':12, 'religious':11}
		else:
			self.df = self.df.apply(self.binarization_apply, axis=1)
			self.labels = {'systematic (home,work,education)':0, 'non-systematic':1}
		
		target = preprocessing.OneHotEncoder().fit_transform(self.df['category'].values.reshape(-1,1))
		df_train = self.df.drop(columns=['category','category_label'])
		self.ct = ColumnTransformer(
			[
				('oh', preprocessing.OneHotEncoder(), ['activity_time', 'mode', 'd_datetime', 'o_datetime', 'occupation', 'gender', 'bin_weekday', 'bin_category']),
				('qt', preprocessing.QuantileTransformer(output_distribution='normal'), 
					['home', 'work', 'eating', 'entertainment', 'recreation', 'shopping', 
						'travel', 'admin_chores', 'religious', 'health', 'police', 'education', 'age',
						# 'P_TOT', 
						# 'MALE_TOT','FEM_TOT',
						'age 25-39', 'age 40-64', 'age >65', 'age 10-24', 
						'P47', 'P48', 'P49','P61','P62',
						# 'INCOME'
						]),
				# ('mm', preprocessing.MinMaxScaler(), ['P61'])
			],
			# remainder='passthrough'
			)
		df_train = df_train.fillna(df_train.mean())
		self.sc_fit = self.ct.fit(df_train)
		data = self.sc_fit.transform(df_train)
		return data, target

	def top_k(self, ytrue, ypred):
		return keras.metrics.top_k_categorical_accuracy(ytrue, ypred, k=2)

	def categorical_model(self, data, target, model_name='model_layers2.h5', n=512, epochs=50, train=1):
		xtrain, xval, ytrain, yval = model_selection.train_test_split(data, target, test_size=0.2, random_state=69)
		if not train:
			try: 
				model = load_model(model_name)
				self.model = model
				print('Loaded')
			except:
				print('Load a Model')
				exit()
		else:
			model = Sequential()
			model.add(Dense(n, input_shape=(data.shape[1],), activation='relu'))
			model.add(Dropout(0.5))
			# model.add(ReLU())
			
			model.add(Dense(n//2, activation='relu'))
			model.add(Dropout(0.5))
			# model.add(ReLU())

			model.add(Dense(len(self.labels), activation='softmax'))
			model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True), 
				metrics=['accuracy', self.top_k])
			print(model.summary())
			history = model.fit(xtrain,ytrain,
				validation_data=(xval, yval),
				epochs=epochs,
				verbose=2,
				callbacks=[keras.callbacks.callbacks.ModelCheckpoint(filepath='models/model.h5',
									save_best_only=True)])

			keras.utils.plot_model(model, to_file='plots/model.png', show_shapes=True, dpi=300)
			model.save('model_nonsyt.h5')
			self.model = model
			self.plot_metrics(history, 'train')
			self.plot_metrics(history, 'validation')

			self.plot_confusion(xtrain, ytrain)
			self.plot_confusion(xval, yval, dataset='Validation')


	def plot_confusion(self, data_to_evaluate, target, dataset='Train'):
		y_pred = self.model.predict_classes(data_to_evaluate)
		target = np.argmax(target, axis=1)
		cm = confusion_matrix(y_true=target, y_pred=y_pred)
		cm = pd.DataFrame(cm, index=[i for i in self.labels.keys()], columns=[i for i in self.labels.keys()])
		plt.figure(figsize=(10,10))
		sns.heatmap(cm, annot=True, cmap='coolwarm')
		if len(self.labels) > 2:
			plt.savefig(f'plots/{dataset}ConfusionMatrix.png')
		else:
			plt.savefig(f'plots/{dataset}ConfusionMatrix_binary.png')

		cm = confusion_matrix(y_true=target, y_pred=y_pred, normalize='true')
		cm = pd.DataFrame(cm, index=[i for i in self.labels.keys()], columns=[i for i in self.labels.keys()])
		plt.figure(figsize=(10,10))
		sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='.2f')
		if len(self.labels) > 2:
			plt.savefig(f'plots/{dataset}ConfusionMatrixNormalized.png')
		else:
			plt.savefig(f'plots/{dataset}ConfusionMatrixNormalized_binary.png')

	def plot_metrics(self, history, metric):
		if metric == 'train':
			metric_loss = 'loss'
			metric_acc = 'accuracy'
			if len(self.labels) > 2:
				metric_top = 'top_k'
		else:
			metric_loss = 'val_loss'
			metric_acc = 'val_accuracy'
			if len(self.labels) > 2:
				metric_top = 'val_top_k'

		import matplotlib
		from matplotlib import rc
		rc('mathtext', default='regular')
		# matplotlib.rcParams.update({'font.size':14})

		fig, ax1 = plt.subplots(figsize=(15,6))
		color = 'tab:red'
		ax1.set_xlabel('Epochs', fontsize=16)
		ax1.set_ylabel('Categorical Loss', color=color, fontsize=16)
		ax1.plot(history.history[metric_loss], label=metric_loss, color=color)
		ax1.legend(fontsize=14)
		ax1.grid()
		ax1.tick_params(axis='y', labelcolor=color)

		ax2 = ax1.twinx()
		color = 'tab:blue'
		ax2.set_ylabel('Accuracy', color=color, fontsize=16)
		ax2.plot(history.history[metric_acc], label=metric_acc, color=color)
		if len(self.labels) > 2:
			ax2.plot(history.history[metric_top], label=metric_top, color='dodgerblue')
		ax2.tick_params(axis='y', labelcolor=color)
		ax2.legend(fontsize=14)
		fig.tight_layout()
		if len(self.labels) > 2:
			plt.savefig(f'plots/{metric_loss}_{metric_acc}.png')
		else:
			plt.savefig(f'plots/{metric_loss}_{metric_acc}_binary.png')


	def open_json(self, file):
		with open(file) as f:
			obj = json.loads(f.read())

		return obj

	def date_to_cat(self, date, step=3):
		hour = date.hour
		converter = {}
		for i in range(0,24,step):
			k = i//step
			v = [i+j for j in range(step)]
			if hour in v:
				return k

	def activity_to_cat(self, activity):
		activity = activity/60
		if activity < 10:
			return 0
		elif 10 < activity < 20:
			return 1
		elif 20 < activity < 30:
			return 2
		elif 30 < activity < 60:
			return 3
		elif 60 < activity < 120:
			return 4
		elif 120 < activity < 240:
			return 5
		elif 240 < activity < 360:
			return 6
		elif 360 < activity < 60*8:
			return 7
		elif 60*8 < activity < 60*12:
			return 8
		elif activity > 12*60:
			return 9
			

	def labelization(self, to_label):
		to_label_df = pd.read_excel(to_label)
		to_label = to_label_df.drop(columns=['category_label', 'category'])

		labels_dict = dict({(self.labels[i], i) for i in self.labels.keys()})
		
		new_sc = self.ct.fit(to_label)
		data = new_sc.transform(to_label)
		predictions = self.model.predict(data)

		new_labels = []
		new_categories = []
		for p in predictions:
			v = np.argmax(p)
			label = labels_dict[v]
			new_categories.append(v)
			new_labels.append(label)		

		to_label_df['category'] = new_categories
		to_label_df['category_label'] = new_labels
		self.labeled_df = to_label_df

	def ssl(self, to_label):
		self.labelization(to_label)
		self.total_df = self.df.append(self.labeled_df)

		df = self.total_df.sample(frac=1)
		target = preprocessing.OneHotEncoder().fit_transform(df['category'].values.reshape(-1,1))
		train = df.drop(columns=['category_label', 'category'])

		scaler = self.ct.fit(train)
		data = scaler.transform(train)

		self.categorical_model(data, target)



