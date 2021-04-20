
# Package Imports
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing

# Miscellineous
import os

os.getcwd()

#Loading the Model
model_two = 'XGBoost.sav'
model = pickle.load(open(model_two, 'rb'))

# Read in data
app_train = pd.read_excel('final data sales details.xlsx', index_col = 'Client Account No.')

# Variables to be considered by the model
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
cate_vars = list(app_data.select_dtypes(include=cates).columns)
categorical_data = app_data[cate_vars]
categorical_data = categorical_data.astype(str)
categorical_data.shape

# Instantiate label encoder
le = preprocessing.LabelEncoder()
categorical_data = categorical_data.apply(lambda col: le.fit_transform(col).astype(str))

# Concat the data
clean_data = pd.concat([categorical_data, numerical_data], axis = 1)
clean_data.shape

# Prepare test data for individual predictions
test_data = clean_data.drop(['target'], axis = 1)


# Define a risk assessment function
def risk_assessor(a):
    client_infor = test_data.loc[[a]].values   #Subset a specific client infor, *a* represent SK_ID_CURR
    prob = model.predict_proba(client_infor).tolist()[0]    #predict a client's probability of defaulting
    p = prob[1]
    if p >= 0.67:
        print('Client with ID # {} has a high risk of defaulting the loan'.format(a))
    elif p > 0.33:
        print('Client with ID # {} has a moderate risk of defaulting the loan'.format(a))
    else:
        print('Client with ID # {} has a low risk of defaulting the loan'.format(a))


risk_assessor(465)




