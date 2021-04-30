# Miscellineous
import os
from datetime import datetime
from flask import Flask,request, url_for, redirect, render_template, jsonify
import requests

# Package Imports
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import requests
import json
import pickle
import os

os.getcwd()



api = Flask(__name__)

@api.route("/")
def hello():
    return "Hello Financial Paradise"



@api.route("/prediction", methods=["POST"])
def get_prediction():
    request_json     = request.get_json()
    prediction = compute_prediction(request_json)
    return jsonify(prediction)

def compute_prediction(request):
    try:
        pre_request = preprocessing(request)
        print("Preproccessed data", pre_request)
        prediction = predict(pre_request)  # only one sample
        print("Prediction data", prediction)
    except Exception as e:
        return {"status": "Error", "message": str(e)}

    return prediction

# Define a risk assessment function
def risk_assessor(data, a):
    client_infor = data.loc[[a]].values   #Subset a specific client infor, *a* represent SK_ID_CURR
    print('client_infor', client_infor)
    #Loading the Model
    model_two = 'XGBoost.sav'
    model = pickle.load(open(model_two, 'rb'))
    prob = model.predict_proba(client_infor).tolist()[0]    #predict a client's probability of defaulting
    p = prob[1]
    client_data = {}
    client_data['client_id'] = a
    client_data['client_prediction'] = p
    print("pred", p)
    return p


def predict(request):
    data = request
    print('PREDICT data', data)
    clients = {}
    client_id = 1001
    scores_list = {}
    for row in data.head().itertuples():
        print(row.Index)
        score = risk_assessor(data, row.Index)
        if row.Index in scores_list:
            scores_list[row.Index].append(score)
        else:
            scores_list[row.Index] = [score]
    print('scores_list', scores_list)
    return scores_list

def preprocessing(request):
    input_data = request
    app_train = pd.DataFrame(input_data['clients'])
    df = app_train.to_excel('10101.xlsx', index = "Client Account No.")
    app_train = pd.read_excel('10101.xlsx', index_col = "Client Account No.")
    app_train.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
    app_train.drop(["a"], axis=1, inplace=True)

    print('app_train', app_train)
    variables  = ['Final branch', 'Sales Details', 'Gender Revised', 'Marital Status', 'HOUSE', 'Loan Type', 'Fund',
                'Loan Purpose', 'Client Type','Client Classification', 'Currency', 'target', 'Highest Sales','Lowest Sales',
                'Age', 'principal_amount']


    # Subset the data
    app_train = app_train.loc[:, variables]
    
    # Replace the N/a class with class 'missing'
    app_train['Sales Details'] = np.where(app_train['Sales Details'].isnull(), 'no saledetails', app_train['Sales Details'])
    app_train['HOUSE'] = np.where(app_train['HOUSE'].isnull(), 'not specified', app_train['HOUSE'])
    app_train['Client Type'] = np.where(app_train['Client Type'].isnull(), 'not specified', app_train['Client Type'])
    app_train['Marital Status'] = np.where(app_train['Marital Status'].isnull(), 'not specified', app_train['Marital Status'])
    app_train['Gender Revised'] = np.where(app_train['Gender Revised'].isnull(), 'not specified', app_train['Gender Revised'])
    app_train['Client Classification'] = np.where(app_train['Client Classification'].isnull(),
                                                'not specified', app_train['Client Classification'])


    # Subset numerical data
    numerics = ['int16','int32','int64','float16','float32','float64']
    numerical_vars = list(app_train.select_dtypes(include=numerics).columns)
    numerical_data = app_train[numerical_vars]

    # Fill in missing values
    numerical_data = numerical_data.fillna(numerical_data.mean())

    # Subset categorical data
    cates = ['object']
    cate_vars = list(app_train.select_dtypes(include=cates).columns)
    categorical_data = app_train[cate_vars]
    categorical_data = categorical_data.astype(str)
    categorical_data.shape

    # Instantiate label encoder
    le = LabelEncoder()
    categorical_data = categorical_data.apply(lambda col: le.fit_transform(col).astype(str))

    # Concat the data
    clean_data = pd.concat([categorical_data, numerical_data], axis = 1)
    clean_data.shape

    # Prepare test data for individual predictions
    test_data = clean_data.drop(['target'], axis = 1)
    return test_data

if __name__ == '__main__':
    api.run(debug=True, host='0.0.0.0')