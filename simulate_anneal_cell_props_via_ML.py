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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def check_spiking(net, cell_properties_vector):
	net.eval()
	with torch.no_grad():
	    out = net(torch.from_numpy(cell_properties_vector).float().unsqueeze(0).to(device)).cpu().numpy()[0]
	    return np.where(out == np.max(out))[0][0]

def norm_col(array):
	min_by_col = np.min(array, axis = 0) 
	array = np.subtract(array, min_by_col)
	max_by_col = np.max(array, axis = 0)
	return np.divide(array, max_by_col)

data_path = r'F:\arbor_ubuntu\synth_data_one_morph_more_stims_v2_normalized_filtered.csv'
normalized_cell_data = pd.read_csv(data_path).dropna()

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

pos_ephys_properties = ['AP_peak_upstroke', 'AP_peak_downstroke',
				 'AP_amplitude', 'AHP2_depth_from_peak',
				 'AP_width', 'AP_height', 'time_to_first_spike',
				 'steady_state_voltage', 'steady_state_voltage_stimend']

neg_ephys_properties = ['sag_ratio1', 'steady_state_voltage_stimend']

ephys_feat = []
pos_currents_to_test = [50, 100, 150, 200]
currents_to_test = [-200, -100, -50, 50, 100, 150, 200]
model_dict = {}

#load all models
for current in pos_currents_to_test:
	model_dict[current] = torch.load(f"F:\\arbor_ubuntu\\synth_data_single_morph_predictors\\subset\\{current}_spike_present.pth", weights_only=False)
	ephys_feat.append(f'{current}_spike_count')

test_idx = 20
test_target_vector = np.asarray(normalized_cell_data[ephys_feat].iloc[test_idx].to_numpy() > 0, dtype = np.float64())
test_cell_properties_vector = normalized_cell_data[cell_props].iloc[test_idx].to_numpy()
print(test_target_vector)

num_trials = 100
min_error_found = 1.0
best_candidate_cell_properties_vector = np.random.rand(len(cell_props))

for trial in range(num_trials):
	candidate_target_vector = np.zeros(len(pos_currents_to_test))
	candidate_cell_properties_vector = np.random.rand(len(cell_props))
	for idx, current in enumerate(pos_currents_to_test):
		net = model_dict[current]
		candidate_target_vector[idx] = check_spiking(net, candidate_cell_properties_vector)
	mse = mean_squared_error(candidate_target_vector, test_target_vector)
	if mse < min_error_found:
		min_error_found = mse
		best_candidate_cell_properties_vector = candidate_cell_properties_vector

print(min_error_found)
print(best_candidate_cell_properties_vector)
print(test_cell_properties_vector)
print(mean_squared_error(best_candidate_cell_properties_vector, test_cell_properties_vector))