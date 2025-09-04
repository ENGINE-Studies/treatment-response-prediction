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
measures = ['Abs_Power$', 'Rel_Power$', 'Osc_Power$', 'FOOOF_1_45$', 'FOOOF_1_45_offset$',
            'FOOOF_30_45$', 'FOOOF_30_35_offset$']
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

params = {
    "selector__k": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31, 41, 51, 61, 71, 81, 91],
    "regressor__n_estimators": [500, 1000],
    "regressor__max_features": ['sqrt', 'log2']
}

outer_cv_rf = KFold(n_splits=folds, shuffle=True, random_state=0)

predicted_values = pd.DataFrame({})
mse_all = {}
mae_all = {}
r2_all = {}

for measure in measures:
    print(measure)
    if measure == '(?<!Imag_)Coherence$':
        interest = 'Coherence$'
    else:
        interest = measure
    type_only = eeg_only_data.filter(regex = measure)
    X = type_only
    print("number of features:", X.shape[1])
    preds_rf = cross_val_predict(pipe, X, y, cv=outer_cv_rf, n_jobs=n_jobs, method = 'predict')

    preds = pd.DataFrame(preds_rf, columns=['predicted'])
    preds.to_csv('initialregression/'+ interest + '_predict.csv', index=False)


    mse = mean_squared_error(y, preds_rf)
    mae = mean_absolute_error(y, preds_rf)
    r2 = r2_score(y, preds_rf)

    predicted_values[measure] = preds_rf
    mse_all[measure] = mse
    mae_all[measure] = mae
    r2_all[measure] = r2

    rfr_interpretation = GridSearchCV(estimator=pipe, param_grid = params, refit=True)
    interpretation_model = rfr_interpretation.fit(X, y)

    filef = 'initialregression/' + interest + '_model_regressor.pkl'
    with open(filef, 'wb') as f:
        pickle.dump(rfr_interpretation, f)
    best_estimator = interpretation_model.best_estimator_.named_steps['regressor'].feature_importances_
    selected_features = interpretation_model.best_estimator_.named_steps.selector.get_support()
    feature_names = X.columns[selected_features]
    list_feature_importance = list(zip(feature_names, best_estimator))
    results = pd.DataFrame(list_feature_importance,
                          columns=['Feature_Name', 'Importance_Score'])
    results.to_csv('initialregression/'+ interest + '_interpretationmodel_features.csv')



with open('initialregression/'+ "mse_onlymeasure.json", "w") as file:
    json.dump(mse_all, file)

with open('initialregression/' + "mae_onlymeasure.json", "w") as file:
    json.dump(mae_all, file)

with open('initialregression/'+"r2_onlymeasure.json", "w") as file:
    json.dump(r2_all, file)

predicted_values.to_csv('initialregression/'+'predicted_values_onlymeasure.csv')