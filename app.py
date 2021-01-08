# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 21:57:07 2021

@author: vamsi
"""


import flask
from flask import Flask, request , jsonify, render_template
#import jinja2
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
filename = 'Model_credit.pkl'
model_credit = pickle.load(open(filename, 'rb'))

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    pred = model_credit.predict(final_features)
    
    return render_template('after.html',data=pred)



if __name__ == "__main__":
    app.run(debug=True)
