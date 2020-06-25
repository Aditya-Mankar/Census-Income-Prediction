import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

dataset = pd.read_csv('adult.csv')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dataset['income'] = le.fit_transform(dataset['income'])

dataset = dataset.replace('?', np.nan)

columns_with_nan = ['workclass', 'occupation', 'native.country']

for col in columns_with_nan:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method == 'POST':
        marital_name = request.form['Marital Status']
        
        marital_value = 0
        
        if marital_name == 'Married-civ-spouse':
            marital_value = 1
        elif marital_name == 'Never-married':
            marital_value = 2
        elif marital_name == 'Divorced':
            marital_value = 3
        elif marital_name == 'Separated':
            marital_value = 4
        elif marital_name == 'Widowed':
            marital_value = 5
        elif marital_name == 'Married-spouse-absent':
            marital_value = 6
        elif marital_name == 'Married-AF-spouse':
            marital_value = 7
    
    
        age_value = request.form['Age']
        edu_num_value = request.form['Years of Education']
        occupation_value = request.form['Occupation Code']
        hours_value = request.form['Hours of work per week']

    
    features = [age_value, edu_num_value, marital_value, 
                occupation_value, hours_value]
    
    int_features = [int(x) for x in features]
    final_features = [np.array(int_features)]
    prediction = model.predict(scaler.transform(final_features))
    
    if prediction == 1:
        output = "Income is more than 50K"
    elif prediction == 0:
        output = "Income is less than 50K"
        
    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)


