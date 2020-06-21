from FirebaseClass import Firebase
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from scipy.stats import planck

# def download_trips(reference):
#     fb = Firebase()
#     fb.authenticate()
#     # data, keys = fb.download('trips')
#     query = fb.db.child("users").order_by_key().equal_to(reference).get().val()
#         # .equal_to(reference).get().val()
#     # return data, keys
#     print(query)
#     # for i in query:
#     #     print(i)


def handle_data(fb, df, territorial_info, territoral_features, columns, th=10):
    trips, _ = fb.download("trips")
    trips = trips[-th:]
    new_data = []
    cnt = 0
    for t in trips:
        cnt+=1
        d_key = t['D']['census_id']
        u = t['user_id']
        new_row = {}
        try:
            for i in territoral_features:
                new_row[i] = territorial_info.loc[d_key, i]
        except:
            prob = ['home', 'work', 'eating', 'entertainment', 'recreation', 'shopping', 'travel', 'admin_chores',
                    'religious',
                    'health', 'police', 'education']
            order = ['work', 'home', 'travel', 'education', 'entertainment', 'recreation', 'eating', 'shopping',
                     'admin_chores', 'religious', 'police',
                     'health']
            for i in territoral_features:
                if i in prob:
                    lambda_ = 0.45 + np.random.uniform(low=0.01, high=0.2)
                    value = planck.pmf(order.index(i), lambda_)
                    new_row[i] = value
                else:
                    value = territorial_info[i].mean()
                    new_row[i] = value

        user = fb.db.child("users").order_by_key().equal_to(u).get().val()
        user = user[u]
        # user = ex_u[0][1]
        new_row['activity_time'] = t['activity_time']
        new_row['mode'] = t['mode']
        new_row['age'] = int(user['age'])
        new_row['gender'] = user['gender']
        new_row['occupation'] = user['occupation']
        new_row['d_hour'] = t['d_hour']
        new_row['bin_weekday'] = t['is_week']
        new_row['alpha_category'] = t['category'].lower()
        new_data.append(new_row)
    x = pd.DataFrame(new_data, columns=columns)
    df = df.append(x, ignore_index=True)

    return df

def retrain_model(fb, df, territorial_info, territorial_features, columns, th=1):
    columns.append('alpha_category')
    # df = pd.read_excel('data/df.xlsx', index_col=0)
    new_df = handle_data(fb, df, territorial_info, territorial_features, columns, th=th)
    new_df = new_df.fillna(-1)
    new_df.to_excel('data/df.xlsx', columns=columns)

    X_train, X_test, y_train, y_test = train_test_split(
        new_df.drop(columns=['alpha_category']), new_df['alpha_category'],
        test_size=0.2)
    best_model = RandomForestClassifier(n_estimators=475, criterion='entropy',
                                        max_features='sqrt', bootstrap=True)
    best_model.fit(X_train, y_train)
    filename = 'data/rf.sav'
    print(best_model.score(X_test, y_test))
    pickle.dump(best_model, open(filename, 'wb'))


if __name__ == '__main__':
    ex_d = [
        {
            'D': {'census_id': 1043, 'lat': 45.0745039, 'lng': 7.6944195, 'name': 'Le Panche Cocktail Bar'},
            'O': {'census_id': 73, 'lat': 45.0703393, 'lng': 7.686864, 'name': 'Torino'},
            'activity_time': 4,
            'category': 'Eating',
            'd_hour': 2,
            'is_week': 1,
            'mode': 4,
            'user_id': '127081263'},
        {
            'D': {'census_id': 3213, 'lat': 45.064949, 'lng': 7.669199000000001,
                  'name': "GAM - Galleria Civica d'Arte Moderna e Contemporanea"},
            'O': {'census_id': 73, 'lat': 45.0703393, 'lng': 7.686864, 'name': 'Torino'},
            'activity_time': 4,
            'category': 'Entertainment',
            'd_hour': 1,
            'is_week': 1,
            'mode': 2,
            'user_id': '127081263'},
        {
            'D': {'census_id': 773, 'lat': 45.066396, 'lng': 7.660298000000001,
                  'name': 'OGR - Officine Grandi Riparazioni'},
            'O': {'census_id': 73, 'lat': 45.0703393, 'lng': 7.686864, 'name': 'Torino'},
            'activity_time': 5,
            'category': 'Eating',
            'd_hour': 2,
            'is_week': 1,
            'mode': 3,
            'user_id': '127081263'
        }]

    ex_k = ['-M80m2dv5756QYUivRT9', '-M80mP0KNpu9pRJfOykQ', '-M80n5KmOWqmhXs-D6OT']

    ex_u = [('127081263', {'age': 24, 'gender': 0, 'occupation': 0})]

    df = pd.read_excel('data/df.xlsx')
    # df = 0
    territoral_features = [
        'P_TOT', 'MALE_TOT', 'FEM_TOT', 'age 0-9', 'age 10-24', 'age 25-39', 'age 40-64', 'age >65', 'male 0-9',
        'male 10-24', 'male 25-39',
        'male 40-64', 'male >65', 'P47', 'P48', 'P49', 'P50', 'P51', 'P52', 'P61', 'P62', 'P128', 'P130', 'P131',
        'P135',
        'P137', 'P138',
        'ST1', 'ST3', 'ST4', 'ST5', 'ST9', 'ST10', 'ST11', 'ST12', 'ST13', 'PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6',
        'PF7',
        'PF8', 'PF9', 'INCOMEinco', 'home', 'work', 'eating', 'entertainment', 'recreation', 'shopping', 'travel',
        'admin_chores', 'religious',
        'health', 'police', 'education'
    ]
    columns = ['activity_time', 'category', 'mode', 'P_TOT', 'MALE_TOT', 'FEM_TOT',
               'age 0-9', 'age 10-24', 'age 25-39', 'age 40-64', 'age >65', 'male 0-9',
               'male 10-24', 'male 25-39', 'male 40-64', 'male >65', 'P47', 'P48',
               'P49', 'P50', 'P51', 'P52', 'P61', 'P62', 'P128', 'P130', 'P131',
               'P135', 'P137', 'P138', 'ST1', 'ST3', 'ST4', 'ST5', 'ST9', 'ST10',
               'ST11', 'ST12', 'ST13', 'PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6', 'PF7',
               'PF8', 'PF9', 'INCOMEinco', 'home', 'work', 'eating', 'entertainment',
               'recreation', 'shopping', 'travel', 'admin_chores', 'religious',
               'health', 'police', 'education', 'age', 'gender', 'occupation',
               'd_hour', 'bin_weekday', 'bin_category', 'alpha_category']

    territorial_info = pd.read_excel('data/FINAL_territorial.xlsx', index_col=1).drop(columns=['Unnamed: 0'])

    retrain_model(df, territorial_info, territoral_features, columns)
