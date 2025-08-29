import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import __version__ as sklearn_version
from packaging import version
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import joblib
from sklearn.metrics import r2_score, mean_squared_error

def norm_col(array):
	min_by_col = np.min(array, axis = 0) 
	array = np.subtract(array, min_by_col)
	max_by_col = np.max(array, axis = 0)
	return np.divide(array, max_by_col)

data_path = r'F:\arbor_ubuntu\synth_data_one_morph_more_stims_v2.csv'
#data_path = r'F:\arbor_ubuntu\all_arbor_synth_data_again.csv'
morph_features = [
    'mean_diameter',
    'max_branch_order',
    'max_euclidean_distance',
	'max_path_distance',
	'num_outer_bifurcations',
	'num_branches',
	'num_nodes',
	'num_tips',
	'total_length',
	'total_surface_area',
	'total_volume'
	]

cell_props = [
	#section_name
	'soma_Ra',
	'soma_g_pas',
	'soma_e_pas',
	'axon_g_pas',
	'axon_e_pas',
	'apic_g_pas',
	'apic_e_pas',
	'dend_g_pas',
	'dend_e_pas',
	'soma_cm',
	'axon_cm',
	'axon_Ra',
	'apic_cm',
	'apic_Ra',
	'dend_cm',
	'dend_Ra',
	'axon_gbar_NaV',
	'axon_gbar_K_T',
	'axon_gbar_Kd',
	'axon_gbar_Kv2like',
	'axon_gbar_Kv3_1',
	'axon_gbar_SK',
	'axon_gbar_Ca_HVA',
	'axon_gbar_Ca_LVA',
	'axon_gamma_CaDynamics',
	'axon_decay_CaDynamics',
	'soma_gbar_NaV',
	'soma_gbar_SK',
	'soma_gbar_Kv3_1',
	'soma_gbar_Ca_HVA',
	'soma_gbar_Ca_LVA',
	'soma_gamma_CaDynamics',
	'soma_decay_CaDynamics',
	'soma_gbar_Ih',
	'apic_gbar_NaV',
	'apic_gbar_Kv3_1',
	'apic_gbar_Im_v2',
	'apic_gbar_Ih',
	'dend_gbar_NaV',
	'dend_gbar_Kv3_1',
	'dend_gbar_Im_v2',
	'dend_gbar_Ih'
	]

pos_ephys_properties = ['steady_state_voltage', 'steady_state_voltage_stimend',
				 'time_to_first_spike', 'time_to_last_spike',
				 'spike_count', 'AP_height', 'AP_width',
				 'AHP_depth', 'AP_amplitude_from_voltagebase', 'mean_frequency', 'voltage_base', 
				 'AHP1_depth_from_peak', 'AHP2_depth_from_peak', 'AP_amplitude', 
				 'AP_peak_downstroke', 'AP_peak_upstroke', 'AP_rise_rate_change',
				 'decay_time_constant_after_stim', 'time_constant']

neg_ephys_properties = ['sag_ratio1', 'sag_time_constant', 'steady_state_voltage_stimend']

ephys_feat = []
few_currents_to_test = [-100, 100, 200]
currents_to_test = [-200, -100, -50, 50, 100, 150, 200]

for c in currents_to_test:
	props = pos_ephys_properties
	if c < 0:
		props = neg_ephys_properties

	for prop in props:
		name = f'{c}_{prop}'
		ephys_feat.append(name)


df = pd.read_csv(data_path).dropna()
df = df[df['200_spike_count'] > 0]

for col in ephys_feat + cell_props:
	df[col] = norm_col(df[col].to_numpy(dtype = np.float64))

df.to_csv(r'F:\arbor_ubuntu\synth_data_one_morph_more_stims_v2_normalized_filtered.csv')