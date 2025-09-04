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


excludedIds = [3011, 3026, 3034, 3043, 3045]
measures = ['Coherence', 'Imag_Coherence', 'WPLI', 'WPPC', 'Abs_Power', 'Rel_Power', 'Osc_Power', 'FOOOF_1_45', 'FOOOF_1_45_offset',
            'FOOOF_30_45', 'FOOOF_30_35_offset']
bands = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']


# Set parameters
n_jobs = 3  # Number of parallel jobs for processing

# Read datasets
redcap_data = pd.read_csv("/Users/ls1002/Documents/Coding/ENGINE/data/clinicaloutcomes.csv")
eeg_connectivity_input = pd.read_csv("/Users/ls1002/Documents/Coding/ENGINE/data/master_connectivity_FULL__nonZ_18-Nov-2024.csv")
#eeg_poweretc = pd.read_csv("/Users/laurensidelinger/PycharmProjects/ENGINE/data/poweretc_nov17_engine_baseline_mdd.csv")

eeg_connectivity_input['Subject'] = [(sub.replace("sub-", "")) for sub in eeg_connectivity_input['Subject']]
#eeg_poweretc['Subject'] = [(sub.replace("sub-", "")) for sub in eeg_poweretc['Subject']]


#reformatting eeg connectivity
eeg_connectivity_melted = pd.melt(eeg_connectivity_input, id_vars =['Subject', 'Channel_1', 'Channel_2', 'FreqBand'], var_name = 'Metric', value_name = 'Value')

eeg_connectivity_renamed = eeg_connectivity_melted.rename(columns={"Subject": "record_id"})

eeg_connectivity_renamed['combocol'] = eeg_connectivity_renamed['Channel_1'] + '_' + \
    eeg_connectivity_renamed['Channel_2'] + '_' + eeg_connectivity_renamed['FreqBand'] + '_' + eeg_connectivity_renamed['Metric']

#eeg_connectivity_renamed.to_csv('eeg_connectivity_reformated_7.21.25.csv', index = False)

df_unique_pairs = eeg_connectivity_renamed.copy()

# Create sorted channel pair columns
df_unique_pairs['chan_min'] = df_unique_pairs[['Channel_1', 'Channel_2']].min(axis=1)
df_unique_pairs['chan_max'] = df_unique_pairs[['Channel_1', 'Channel_2']].max(axis=1)

# Drop duplicates based on sorted channel pairs plus other grouping columns
df_unique_pairs = df_unique_pairs.drop_duplicates(
    subset=['record_id', 'FreqBand', 'Metric', 'chan_min', 'chan_max']
)

# (Optional) Drop helper columns
df_unique_pairs = df_unique_pairs.drop(columns=['chan_min', 'chan_max'])

eeg_connectivity = df_unique_pairs.pivot(index = 'record_id', columns = 'combocol', values = 'Value').reset_index()

eeg_connectivity.to_csv('eeg_connectivity_reformated_7.21.25.csv', index = False)


#cleaning eeg power
#eeg_power_melted = pd.melt(eeg_poweretc, id_vars =['Subject', 'Channel', 'FreqBand'], var_name = 'Metric', value_name = 'Value')

#eeg_power_renamed = eeg_power_melted.rename(columns={"Subject": "record_id"})

#eeg_power_renamed['combocol'] = eeg_power_renamed['Channel'] + '_' + eeg_power_renamed['FreqBand'] + '_' + eeg_power_renamed['Metric']

#eeg_power = eeg_power_renamed.pivot(index = 'record_id', columns = 'combocol', values = 'Value').reset_index()

#eeg_power.to_csv('data/eeg_power_reformated_1.14.25.csv', index = False)

eeg_power = pd.read_csv('/Users/ls1002/Documents/Coding/ENGINE/data/eeg_power_reformated_1.14.25.csv')

#join together eeg dataframes

eeg_superfile = pd.concat([eeg_connectivity, eeg_power], axis = 1)
eeg_superfile.to_csv('eeg_superfile_7.21.25.csv', index = False)
#eeg_connectivity['comboname'] = eeg_connectivity['Subject'] + '_' + eeg_connectivity['Channel_1'] + '_' + \
 #   eeg_connectivity['Channel_2'] + '_' + eeg_connectivity['FreqBand']


#widened_eeg_power = eeg_poweretc.pivot(index = 'Subject', )


#eeg_connectivity['Subject'] = [int(sub.replace("sub-", "")) for sub in eeg_connectivity['Subject']]
#eeg_data = eeg_connectivity.rename(columns={"Subject": "record_id"})

#edcap_rel_ids = redcap_data[~redcap_data['record_id'].isin(excludedIds)]
#eeg_rel_ids = eeg_data[~eeg_data['record_id'].isin(excludedIds)].reset_index(drop = True)



#eeg_only_data = eeg_rel_ids.drop(columns=['record_id'])

#coherence_only = eeg_only_data.filter(like = 'Gamma_WPPC')
