import numpy as np
import pandas as pd
from RandomF import Tools
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.ensemble import RandomForestClassifier
import pickle


#EXTRACTING AND DIVIDING THE DATASET
data_path = 'df.xlsx'
data = pd.read_excel(data_path)
X_train,X_test,y_train,y_test = train_test_split(data.drop(['category','alpha_category','bin_category'],axis='columns'),data['alpha_category'], test_size=0.2)
	

#LOADING THE BEST CONFIGURATIONS AND FIND THE BEST NUM OF ESTIMATORS
random_results = np.load('results.npy',allow_pickle='TRUE').item()
best_pos = list(random_results['rank_test_score']).index(min(random_results['rank_test_score'])) 
best_params = random_results['params'][best_pos]
results_grid = np.load('grid_results_2.npy',allow_pickle='TRUE').item()
n_estimators = np.arange(1,500,1)
Tools = Tools()
n_est_final = Tools.report_final(results_grid, n_estimators)


#TRAINING THE MODEL
best_model = RandomForestClassifier(n_estimators=n_est_final,criterion=best_params['criterion'],max_features=best_params['max_features'], bootstrap = best_params['bootstrap'])
best_model.fit(X_train,y_train)
#print(best_model.score(X_test,y_test))

#SAVING THE MODEL FOR REUSE
filename = 'finalized_RandomF.sav'
pickle.dump(best_model, open(filename, 'wb'))

#LOADING MODEL AND TEST IT
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)