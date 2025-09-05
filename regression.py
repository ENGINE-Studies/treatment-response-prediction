from typing import BinaryIO

import numpy as np
import pandas as pd
import random
import json
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
measures = ['(?<!Imag_)Coherence$', 'Imag_Coherence$', 'WPLI$', 'WPPC$']
bands = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']

random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)

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

estimators = [
    ('imputer', SimpleImputer(strategy='median')),  # Missing values will be replaced by the median
    ('selector', SelectKBest(mutual_info_regression)),  # Select the best features
    ('regressor', RandomForestRegressor(random_state=0))
]

pipe = Pipeline(estimators)

# set parameters for nested grid search - these params are for my "inner cross validation", hyperparameter turning 
params = {
    "selector__k": [5, 10, 20, 40, 60, 80, 120, 160],
    "regressor__n_estimators": [500, 1000],
    "regressor__max_features": ['sqrt', 'log2']
}

#because my n splits = n samples, this is effectively leave one out CV ; "outer cross validation"
outer_cv_rf = KFold(n_splits=folds, shuffle=True, random_state=random_seed)

predicted_values = pd.DataFrame({})
mse_all = {}
mae_all = {}
r2_all = {}

for measure in measures:
    for band in bands:
        interest = band + '_' + measure
        print(interest)
        if interest == "Alpha_Abs_Power$":
            continue
        elif interest == "Beta_Abs_Power$":
            continue        

        type_only = eeg_only_data.filter(regex = interest)
        X = type_only
        print("number of features:", X.shape[1])

        #hyperparameter tuning
        inner_cv = KFold(n_splits=min(5, folds-1), shuffle=True, random_state=random_seed)
        grid = GridSearchCV(pipe, param_grid=params, cv=inner_cv, n_jobs=n_jobs, scoring='r2')  # Optimize for R²

        #loocv predictions
        preds_rf = cross_val_predict(grid, X, y, cv=outer_cv_rf, n_jobs=n_jobs, method = 'predict')

        preds = pd.DataFrame(preds_rf, columns=['predicted'])
        preds.to_csv('initialregression/'+ interest + '_predictHYPERPARAMETEROPTIMIZED.csv', index=False)

        mse = mean_squared_error(y, preds_rf)
        mae = mean_absolute_error(y, preds_rf)
        r2 = r2_score(y, preds_rf)

        print(mse)
        print(mae)
        print(r2)
        predicted_values[interest] = preds_rf


        mse_all[interest] = mse
        mae_all[interest] = mae
        r2_all[interest] = r2

        grid.fit(X, y)  # fit on full data to create final model
        final_model = grid.best_estimator_  # best pipeline with tuned hyperparameters

        with open(f'initialregression/{interest}_grid.pkl', 'wb') as f:
            pickle.dump(grid, f)  # Save the entire GridSearchCV object



with open('initialregression/'+ "mse_allnew_measureonlyHYPERPARAMETEROPTIMIZED.json", "w") as file:
    json.dump(mse_all, file)

with open('initialregression/' + "mae_allnew_measureonlyHYPERPARAMETEROPTIMIZED.json", "w") as file:
    json.dump(mae_all, file)

with open('initialregression/'+"r2_allnew_measureonlyHYPERPARAMETEROPTIMIZED.json", "w") as file:
    json.dump(r2_all, file)

predicted_values.to_csv('initialregression/'+'predicted_valuesnew_measureonlyHYPERPARAMETEROPTIMIZED.csv')