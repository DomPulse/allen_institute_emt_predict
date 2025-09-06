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

def numerical_model(net, cell_properties_vector):
	net.eval()
	with torch.no_grad():
		out = net(torch.from_numpy(cell_properties_vector).float().unsqueeze(0).to(device)).cpu().numpy()[0]
		return out[0]

def norm_col(array):
	min_by_col = np.min(array, axis = 0) 
	array = np.subtract(array, min_by_col)
	max_by_col = np.max(array, axis = 0)
	return np.divide(array, max_by_col)

def calc_mse(candidate_cell_properties_vector):
	for idx, name in enumerate(current_named_ephys_feat):
		net = model_dict[name]
		if 'spike' in name:
			candidate_target_vector[idx] = check_spiking(net, candidate_cell_properties_vector)
		else:
			candidate_target_vector[idx] = numerical_model(net, candidate_cell_properties_vector)
	return mean_squared_error(goal_target_vector, candidate_target_vector)

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

neg_ephys_properties = ['sag_ratio1', 'steady_state_voltage_stimend']
pos_ephys_properties = ['AP_peak_upstroke', 'AP_peak_downstroke',
				 'voltage_base', 'AHP1_depth_from_peak',
				 'AP_amplitude_from_voltagebase', 'AHP_depth', 'AP_width',
				 'AP_height', 'steady_state_voltage']

current_named_ephys_feat = []
neg_currents_to_test = [-200, -100, -50]
pos_currents_to_test = [50, 100, 150, 200]
spiking_pos_currents = []
model_dict = {}

goal_idx = 20

for current in pos_currents_to_test:
	name = f'{current}_spike_present'
	model_dict[name] = torch.load(f"F:\\arbor_ubuntu\\synth_data_single_morph_predictors\\subset\\{name}.pth", weights_only=False)
	current_named_ephys_feat.append(name)
	normalized_cell_data[name] = np.asarray(normalized_cell_data[f'{current}_spike_count'].to_numpy() > 0, dtype = np.float64())
	goal_spike_present_at_current = normalized_cell_data[name].iloc[goal_idx]
	if goal_spike_present_at_current > 0:
		spiking_pos_currents.append(current)

#load all models
for current in neg_currents_to_test:
	for ephys_feat in neg_ephys_properties:
		name = f'{current}_{ephys_feat}'
		model_dict[name] = torch.load(f"F:\\arbor_ubuntu\\synth_data_single_morph_predictors\\subset\\{name}.pth", weights_only=False)
		current_named_ephys_feat.append(name)

for current in spiking_pos_currents:
	for ephys_feat in pos_ephys_properties:
		name = f'{current}_{ephys_feat}'
		model_dict[name] = torch.load(f"F:\\arbor_ubuntu\\synth_data_single_morph_predictors\\subset\\{name}.pth", weights_only=False)
		current_named_ephys_feat.append(name)

goal_target_vector = normalized_cell_data[current_named_ephys_feat].iloc[goal_idx].to_numpy(dtype = np.float64)
goal_cell_properties_vector = normalized_cell_data[cell_props].iloc[goal_idx].to_numpy(dtype = np.float64)

cell_prop_test_idx = 0
goal_cell_prop_value = normalized_cell_data[cell_props[cell_prop_test_idx]].iloc[goal_idx]
print(goal_cell_prop_value)
best_cand = np.zeros(len(cell_props))
num_points_to_test = 20
num_trials_per_test = 10000
cell_prop_test_values = np.linspace(0, 1, num_points_to_test)
mean_error = np.zeros(num_points_to_test)
all_tested_errors = []
for gaming, prop in enumerate(cell_prop_test_values):
	for t in range(num_trials_per_test):
		candidate_target_vector = np.zeros(len(current_named_ephys_feat))
		candidate_cell_properties_vector = np.random.rand(len(cell_props))
		candidate_cell_properties_vector[cell_prop_test_idx] = prop
		mse = calc_mse(candidate_cell_properties_vector)
		all_tested_errors.append(mse)
		if mse == np.min(all_tested_errors):
			best_cand = candidate_cell_properties_vector
		mean_error[gaming] += mse/num_trials_per_test
	print(gaming, mean_error[gaming])

plt.plot(cell_prop_test_values, mean_error)
plt.plot([goal_cell_prop_value, goal_cell_prop_value], [-10, 10], color = 'black', ls = '--')
plt.ylim(0.9*np.min(mean_error), 1.1*np.max(mean_error))
plt.xlabel('tested cellular property value')
plt.ylabel('mean error of predicted ephys')
plt.show()

print(calc_mse(goal_cell_properties_vector), np.min(all_tested_errors))

plt.scatter(goal_cell_properties_vector, candidate_cell_properties_vector)
plt.ylabel('best candidate properties')
plt.xlabel('true cellular properties')
plt.show()
