import pandas as pd
import numpy as np

import pickle

dataset = pd.read_csv('adult.csv')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dataset['income'] = le.fit_transform(dataset['income'])

dataset = dataset.replace('?', np.nan)

columns_with_nan = ['workclass', 'occupation', 'native.country']

for col in columns_with_nan:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

dataset['marital.status'] = dataset['marital.status'].map(
                            {'Married-civ-spouse' : 1,
                             'Never-married' : 2,
                             'Divorced' : 3,
                             'Separated' : 4,
                             'Widowed' : 5,
                             'Married-spouse-absent' : 6,
                             'Married-AF-spouse' : 7
                             })

for col in dataset.columns:
    if dataset[col].dtypes == 'object':
        encoder = LabelEncoder()
        dataset[col] = encoder.fit_transform(dataset[col])

X = dataset.drop('income', axis=1)
Y = dataset['income']

X = X.drop(['workclass', 'education', 'race', 'sex',
            'capital.loss', 'native.country', 'fnlwgt', 'relationship',
            'capital.gain'], axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(X) 

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)

ros.fit(X, Y)

X_resampled, Y_resampled = ros.fit_resample(X, Y)

X = X_resampled
Y = Y_resampled

from sklearn.ensemble import RandomForestClassifier
ran_for = RandomForestClassifier(max_depth=102, n_estimators=40, random_state=42)

ran_for.fit(X, Y)

pickle.dump(ran_for, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

prediction = model.predict(scaler.transform(np.array([[20, 10, 3, 5, 40]])))






