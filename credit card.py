# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 13:32:23 2021

@author: vamsi
"""


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

data = pd.read_excel(r'default_of_credit_card_clients_0 (1).xlsx')

data

# provding column names to the data frame

data.columns = ['ID' , 'LIMIT_BAL' , 'SEX' , 'EDUCATION' , 'MARRIAGE' , 'AGE' ,  'PAY_1' , 'PAY_2' , 'PAY_3' , 'PAY_4' , 'PAY_5' ,  'PAY_6' ,
               'BILL_AMT1' ,'BILL_AMT2' ,'BILL_AMT3' ,'BILL_AMT4' ,'BILL_AMT5' ,'BILL_AMT6' , 'PAY_AMT1', 'PAY_AMT2' , 'PAY_AMT3' , 'PAY_AMT4' , 'PAY_AMT5' , 'PAY_AMT6' , 'default payment']

df = data.drop (0 , axis=0) 
df

# will import test train split to predict the default payment using the above variables
df = df.apply(pd.to_numeric)

from sklearn.model_selection import train_test_split
X_1 = df[[ 'PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6' ]]
y_1 = df['default payment']
X_train,X_test , y_train ,  y_test  = train_test_split (X_1 , y_1 , test_size = 0.2)

print(model.predict([[0,1,0,0,1,0]]))

model_credit = LogisticRegression(penalty='l2',  C=10 , solver = 'lbfgs' )

model_credit.fit(X_train,y_train.values.ravel())

y_pred = model_credit.predict(X_test)
y_pred

metrics.classification_report(y_test, y_pred)
metrics.confusion_matrix(y_test, y_pred)

import pickle

pickle.dump(model_credit, open("Model_credit.pkl", "wb"))

model = pickle.load(open("Model_credit.pkl", "rb"))

