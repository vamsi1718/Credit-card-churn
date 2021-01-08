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
filename = 'Model_GYM.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def man():
    return render_template('home.html')