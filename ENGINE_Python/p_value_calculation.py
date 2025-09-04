from typing import BinaryIO

import numpy as np
import pandas as pd
import json
import random
import os
from glob import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
import pickle

def permutation_p(var, r2_score):
    filename = next((f for f in os.listdir('rfrpermtests') if var in f))
    df = pd.read_csv('rfrpermtests/' + filename)
    numAbove = len(df[df['R2'] > r2_score])
    p_val = numAbove/1000
    return p_val

measures = ['Imag_Coherence']
bands = ['Alpha']

p_all = {}

with open("initialregression/r2_all.json", "r") as file:
    r2_all = json.load(file)

for measure in measures:
    for band in bands:
        interest = band + '_' + measure
        print(interest)

        rsq = r2_all[interest]

        p_all[interest] = permutation_p(interest, rsq)


with open("pval_alpha.json", "w") as file:
    json.dump(p_all, file)