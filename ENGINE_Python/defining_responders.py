from typing import BinaryIO

import numpy as np
import pandas as pd
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

from classifier_treatmentresponse import baseline

excludedIds = [3011, 3026, 3034, 3043, 3045]
measures = ['Coherence', 'Imag_Coherence', 'WPLI', 'WPPC', 'Abs_Power', 'Rel_Power', 'Osc_Power', 'FOOOF_1_45', 'FOOOF_1_45_offset',
            'FOOOF_30_45', 'FOOOF_30_45_offset']
bands = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']


# Set parameters
#n_jobs = 3  # Number of parallel jobs for processing

# Read datasets
outcome_data = pd.read_csv("/Users/laurensidelinger/PycharmProjects/ENGINE/data/clinicaloutcomes.csv")
baseline_data = pd.read_csv("data/clinicalbaseline.csv")


outcome_rel_ids = outcome_data[~outcome_data['record_id'].isin(excludedIds)]
baseline_rel_ids = baseline_data[~baseline_data['record_id'].isin(excludedIds)].reset_index(drop = True)

baseline_cdrs = baseline_rel_ids['cdrs_total']
outcome_cdrs = outcome_rel_ids['cdrs_total']

cdrs_diff = outcome_cdrs - baseline_cdrs
percent_change = cdrs_diff/baseline_cdrs

d = {'record_id': outcome_rel_ids['record_id'], 'cdrs_diff': cdrs_diff, 'percent_change': percent_change,
     'responder_status': percent_change <= -.5}
df = pd.DataFrame(data = d)
print(df)

df.to_csv('data/responder_status.csv')
#outcome_rel_ids['cdrs_percent_change'] = percent_change
#outcome_rel_ids['responder_status'] = percent_change <= -.5

#print(outcome_rel_ids['responder_status'])