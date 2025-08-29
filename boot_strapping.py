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

data_path = r'F:\arbor_ubuntu\synth_data_one_morph_more_stims_v2_normalized_filtered.csv'
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

possible_predicts = [
	#section_name
	'soma_g_pas',
	'soma_e_pas',
	'apic_g_pas',
	'apic_e_pas',
	'soma_cm',
	'apic_cm',
	'apic_Ra',
	'soma_gbar_NaV',
	'soma_gbar_SK',
	'soma_gbar_Kv3_1',
	'soma_gbar_Ca_LVA',
	'soma_gamma_CaDynamics',
	'soma_decay_CaDynamics',
	'apic_gbar_NaV',
	'apic_gbar_Im_v2',
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current device: {device}')
hid_size = 256
drop_frac = 0.1
net = nn.Sequential(
	nn.Linear(len(possible_predicts + ephys_feat) - 1, hid_size),
	nn.Tanh(),
	nn.Dropout(drop_frac),	
	
	nn.Linear(hid_size, hid_size),
	nn.Tanh(),
	nn.Dropout(drop_frac),
	
	
	nn.Linear(hid_size, 1)
).to(device)

feature_act_dict = {}
for feature in possible_predicts:
	bleh = f'{feature}_true'
	blah = f'{feature}_predict'
	feature_act_dict[bleh] = []
	feature_act_dict[blah] = []
	
for test_idx in range(100):
	#test_idx = 1
	true_output = df.iloc[test_idx].loc[possible_predicts].to_numpy(dtype = np.float64)
	predicted_output = np.random.rand(len(possible_predicts))
	fixed_input_ephys_input = df.iloc[test_idx].loc[ephys_feat].to_numpy(dtype = np.float64)
	
	#fixed_input_ephys_input = np.random.rand(len(ephys_feat)) 
	#this will be taken it out, i need to save and train on normalized data
	
	net.eval()
	num_iters = 100
	for i in range(num_iters):
		#print(predicted_output)
		for index, output_feature in enumerate(possible_predicts):
			state_dict = torch.load(f"F:\\arbor_ubuntu\\synth_data_single_morph_predictors\\subset\\{output_feature}.pth", map_location=device)
			net.load_state_dict(state_dict)
			all_except_testing = np.delete(predicted_output, [index])
			concat_input = np.concat((fixed_input_ephys_input, all_except_testing))
			with torch.no_grad():
				predicted_output[index] = np.clip(net(
					torch.from_numpy(concat_input).float().unsqueeze(0).to(device)
				).cpu().numpy().item(), 0, 1)
	
	print(test_idx)
	for j, feature in enumerate(possible_predicts):
		#print(f'{feature}, true:{true_output[j]}, predicted:{predicted_output[j]}')
		bleh = f'{feature}_true'
		blah = f'{feature}_predict'
		feature_act_dict[bleh].append(true_output[j])
		feature_act_dict[blah].append(predicted_output[j])
		
for feature in possible_predicts:
	bleh = f'{feature}_true'
	blah = f'{feature}_predict'

	plt.show()
	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.scatter(feature_act_dict[blah], feature_act_dict[bleh])
	test_r2 = r2_score(feature_act_dict[bleh], feature_act_dict[blah])
	plt.title(f'{feature}: r^2 = {test_r2:.2f}')
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlabel('model prediction')
	plt.ylabel('true value')