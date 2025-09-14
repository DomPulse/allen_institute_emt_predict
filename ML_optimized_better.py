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
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#chat's handy work
def update_value(data, section=None, name=None, mechanism=None, new_value=None):
	"""
	Update values in the JSON structure.
	
	Args:
		data (dict): Parsed JSON.
		section (str | None): Section (e.g., "soma", "axon"). None if not applicable.
		name (str | None): Parameter name (e.g., "g_pas", "Ra").
		mechanism (str | None): Mechanism name, or None/"" for missing.
		new_value (float | str): New value to assign.
	"""
	# Case 1: update passive Ra
	if name == "ra" and "passive" in data:
		if "ra" in data["passive"][0]:
			data["passive"][0]["ra"] = float(new_value)
			return True
	
	# Case 2: update genome entries
	if "genome" in data:
		mech = mechanism if mechanism is not None else ""
		for entry in data["genome"]:
			if (entry["section"] == section 
				and entry["name"] == name 
				and entry["mechanism"] == mech):
				entry["value"] = str(new_value)
				return True
	
	return False

def parse_param_key(key, mechanisms=None):
	"""
	Split a key like 'soma_g_pas' or 'soma_decay_CaDynamics' 
	into section, name, mechanism.
	
	Args:
		key (str): The key string (e.g., 'soma_g_pas').
		mechanisms (list[str] | None): List of known mechanisms 
			(e.g., ['NaV','CaDynamics','Kv3_1']). Defaults to a common set.
	
	Returns:
		tuple: (section, name, mechanism)
	"""
	if mechanisms is None:
		mechanisms = [
			"NaV", "K_T", "Kd", "Kv2like", "Kv3_1", "SK",
			"Ca_HVA", "Ca_LVA", "CaDynamics", "Ih", "Im_v2"
		]
	
	section, rest = key.split("_", 1)
	
	# check if rest ends with a mechanism
	for mech in mechanisms:
		if rest.endswith(mech):
			name = rest[: -len(mech)]
			return section, name.rstrip("_"), mech
	
	# otherwise no mechanism
	return section, rest, ""

#back to me

def denormalize_value(normal_val, minimum = 0, maximum = 1):
	return normal_val*maximum + minimum

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

def norm_col(array, max_by_col = None, min_by_col = None):
	if min_by_col is None:
		min_by_col = np.min(array, axis = 0) 
	array = np.subtract(array, min_by_col)
	if max_by_col is None:
		max_by_col = np.max(array, axis = 0)
	return np.divide(array, max_by_col)

def calc_mse(candidate_cell_properties_vector, goal_target_vector):
	candidate_target_vector = np.zeros(len(goal_target_vector))
	for idx, name in enumerate(current_named_ephys_feat):
		net = model_dict[name]
		if '_spike_present' in name:
			candidate_target_vector[idx] = check_spiking(net, candidate_cell_properties_vector)
		else:
			candidate_target_vector[idx] = numerical_model(net, candidate_cell_properties_vector)
	return mean_squared_error(goal_target_vector, candidate_target_vector), candidate_target_vector

data_path = r'F:\arbor_ubuntu\synth_data_one_morph_more_stims_v2.csv'
minimax_path = r'F:\arbor_ubuntu\synth_data_one_morph_more_stims_v2_minimax.csv'
cell_data = pd.read_csv(data_path).dropna()
normal_minimax = pd.read_csv(minimax_path)

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
	'dend_gbar_Ih',
	'soma_ra'
	]

neg_ephys_properties = ['sag_ratio1', 'steady_state_voltage_stimend']
pos_ephys_properties = ['AP_peak_upstroke', 'AP_peak_downstroke',
				 'voltage_base', 'AHP2_depth_from_peak',
				 'AP_amplitude_from_voltagebase', 'AP_width',
				 'AP_height', 'AP_amplitude', 'steady_state_voltage', 'time_to_first_spike', 'spike_count', 'time_to_last_spike']

current_named_ephys_feat = []
neg_currents_to_test = [-200, -100, -50]
pos_currents_to_test = [50, 100, 150, 200]
spiking_pos_currents = []
model_dict = {}

goal_idx = 19

for current in pos_currents_to_test:
	name = f'{current}_spike_present'
	model_dict[name] = torch.load(f"F:\\arbor_ubuntu\\synth_data_single_morph_predictors\\subset\\{name}_again.pth", weights_only=False)
	current_named_ephys_feat.append(name)
	cell_data[name] = np.asarray(cell_data[f'{current}_spike_count'].to_numpy() > 0, dtype = np.float64())
	goal_spike_present_at_current = cell_data[name].iloc[goal_idx]
	if goal_spike_present_at_current > 0:
		spiking_pos_currents.append(current)

#load all models
for current in neg_currents_to_test:
	for ephys_feat in neg_ephys_properties:
		name = f'{current}_{ephys_feat}'
		model_dict[name] = torch.load(f"F:\\arbor_ubuntu\\synth_data_single_morph_predictors\\subset\\{name}_again.pth", weights_only=False)
		current_named_ephys_feat.append(name)

for current in spiking_pos_currents:
	for ephys_feat in pos_ephys_properties:
		name = f'{current}_{ephys_feat}'
		model_dict[name] = torch.load(f"F:\\arbor_ubuntu\\synth_data_single_morph_predictors\\subset\\{name}_again.pth", weights_only=False)
		current_named_ephys_feat.append(name)

goal_target_vector = np.zeros(len(current_named_ephys_feat))
goal_cell_properties_vector = np.zeros(len(cell_props))

for idx, prop in enumerate(current_named_ephys_feat):
	goal_target_vector[idx] = cell_data.loc[goal_idx, prop]
	if '_spike_present' not in prop:
		this_prop_min = normal_minimax[normal_minimax['cols'] == prop]['min'].to_list()[0] #this can't be right but eh
		this_prop_max = normal_minimax[normal_minimax['cols'] == prop]['max'].to_list()[0] #this can't be right but eh
		goal_target_vector[idx] = norm_col(goal_target_vector[idx], this_prop_max, this_prop_min)

for idx, prop in enumerate(cell_props):
	goal_cell_properties_vector[idx] = cell_data.loc[goal_idx, prop]
	this_prop_min = normal_minimax[normal_minimax['cols'] == prop]['min'].to_list()[0] #this can't be right but eh
	this_prop_max = normal_minimax[normal_minimax['cols'] == prop]['max'].to_list()[0] #this can't be right but eh
	goal_cell_properties_vector[idx] = norm_col(goal_cell_properties_vector[idx], this_prop_max, this_prop_min)

print(goal_cell_properties_vector)
print(goal_target_vector)


best_cand = np.zeros(len(cell_props))
num_trials_per_test = 50000
all_tested_errors = []
for t in range(num_trials_per_test):
	candidate_cell_properties_vector = np.random.rand(len(cell_props))
	mse, _ = calc_mse(candidate_cell_properties_vector, goal_target_vector)
	all_tested_errors.append(mse)
	if mse == np.min(all_tested_errors):
		best_cand = candidate_cell_properties_vector

print('true:', goal_cell_properties_vector)
print('best cand:', best_cand)

print(calc_mse(goal_cell_properties_vector, goal_target_vector))

plt.scatter(goal_cell_properties_vector, best_cand)
plt.ylabel('best candidate properties')
plt.xlabel('true cellular properties')
plt.show()

plt.scatter(goal_target_vector, calc_mse(best_cand, goal_target_vector)[1])
plt.ylabel('best candidate predicted ephys')
plt.xlabel('true ephys')
plt.show()


plt.scatter(goal_target_vector, calc_mse(goal_cell_properties_vector, goal_target_vector)[1])
plt.ylabel('ephys predict from true cell prop')
plt.xlabel('true ephys')
plt.show()

json_path = rf'F:\arbor_ubuntu\10k_randomized_jsons\random_genome_{goal_idx+1}.json'
with open(json_path, "r") as f:
	example_json = json.load(f)
	
for idx, prop in enumerate(cell_props):
	this_prop_min = normal_minimax[normal_minimax['cols'] == prop]['min'].to_list()[0] #this can't be right but eh
	this_prop_max = normal_minimax[normal_minimax['cols'] == prop]['max'].to_list()[0] #this can't be right but eh
	section, name, mechanism = parse_param_key(prop)
	update_value(example_json, section, name, mechanism, denormalize_value(best_cand[idx], this_prop_min, this_prop_max))


# Save back to file
with open("cell_params_updated.json", "w") as f:
	json.dump(example_json, f, indent=4)


