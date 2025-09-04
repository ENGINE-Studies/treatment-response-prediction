from typing import BinaryIO

import numpy as np
import pandas as pd
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

excludedIds = [3011, 3026, 3034, 3043, 3045]
measures = [ 'Imag_Coherence$']
bands = ['Alpha']


# Set parameters
n_jobs = 1  # Number of parallel jobs for processing

redcap_data = pd.read_csv('/Users/ls1002/Documents/Coding/ENGINE/data/clinicaloutcomes.csv')
eeg = pd.read_csv('/Users/ls1002/Documents/Coding/ENGINE/data/eeg_superfile_7.21.25.csv')


redcap_rel_ids = redcap_data[~redcap_data['record_id'].isin(excludedIds)]
eeg_rel_ids = eeg[~eeg['record_id'].isin(excludedIds)].reset_index(drop = True)

folds = redcap_rel_ids.shape[0]

eeg_only_data = eeg_rel_ids.drop(columns=['record_id'])
#Define X and y for model
y = redcap_rel_ids['cdrs_total']

for band in measures:

    type_only = eeg_only_data.filter(regex = band)
    X = type_only

    mse_list = []
    mae_list = []
    r2_list = []

    for i in range(1000):
        random_seed = i
        np.random.seed(random_seed)
        random.seed(random_seed)

        estimators = [
            ('imputer', SimpleImputer(strategy='median')),  # Missing values will be replaced by the median
            ('selector', SelectKBest(mutual_info_regression)),  # Select the best features
            ('regressor', RandomForestRegressor(random_state=i))
        ]

        pipe = Pipeline(estimators)
        outer_cv_rf = KFold(n_splits=folds, shuffle=True, random_state=i)

        random.Random(i).shuffle(y)
        preds_rf = cross_val_predict(pipe, X, y, cv=outer_cv_rf, n_jobs=n_jobs, method = 'predict')

        mse = mean_squared_error(y, preds_rf)
        mae = mean_absolute_error(y, preds_rf)
        r2 = r2_score(y, preds_rf)


        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)

    df = pd.DataFrame({
        "MSE": mse_list,
        "MAE": mae_list,
        "R2": r2_list
    })

    fileoutname = 'rfrpermtests/' + band + '_permutationtest.csv'

    df.to_csv(fileoutname, index = False)

  



