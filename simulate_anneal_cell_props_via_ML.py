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
currents_to_test = [-200, -100, -50, 50, 100, 150, 200]
model_dict = {}

#load all models
for current in neg_currents_to_test:
	for ephys_feat in neg_ephys_properties:
		name = f'{current}_{ephys_feat}'
		model_dict[name] = torch.load(f"F:\\arbor_ubuntu\\synth_data_single_morph_predictors\\subset\\{name}.pth", weights_only=False)
		current_named_ephys_feat.append(name)

for current in pos_currents_to_test:
	for ephys_feat in pos_ephys_properties:
		name = f'{current}_{ephys_feat}'
		model_dict[name] = torch.load(f"F:\\arbor_ubuntu\\synth_data_single_morph_predictors\\subset\\{name}.pth", weights_only=False)
		current_named_ephys_feat.append(name)

for current in pos_currents_to_test:
	name = f'{current}_spike_present'
	model_dict[name] = torch.load(f"F:\\arbor_ubuntu\\synth_data_single_morph_predictors\\subset\\{name}.pth", weights_only=False)
	current_named_ephys_feat.append(name)
	normalized_cell_data[name] = np.asarray(normalized_cell_data[f'{current}_spike_count'].to_numpy() > 0, dtype = np.float64())
	

test_idx = 20
test_target_vector = normalized_cell_data[current_named_ephys_feat].iloc[test_idx].to_numpy(dtype = np.float64)
test_cell_properties_vector = normalized_cell_data[cell_props].iloc[test_idx].to_numpy(dtype = np.float64)
print(test_target_vector)

# --- Hyperparameters ---
population_size = 200	   # number of candidates in each generation
num_generations = 1000	  # number of generations to run
mutation_rate = 0.2		# probability of mutating a gene
mutation_strength = 0.1	# size of mutation (Gaussian noise)
elitism = 20			# keep top N from each generation

# --- Helper: evaluate fitness (lower MSE = better) ---
def evaluate_candidate(candidate):
	target_vector = np.zeros(len(current_named_ephys_feat))
	for idx, model_name in enumerate(current_named_ephys_feat):
		net = model_dict[model_name]
		if 'spike' in model_name:
			target_vector[idx] = check_spiking(net, candidate)
		else:
			target_vector[idx] = numerical_model(net, candidate)
	return mean_squared_error(target_vector, test_target_vector)

# --- Initialize population ---
population = np.random.rand(population_size, len(cell_props))
fitness = np.array([evaluate_candidate(ind) for ind in population])

best_idx = np.argmin(fitness)
best_candidate = population[best_idx].copy()
best_error = fitness[best_idx]

# --- Evolution loop ---
for gen in range(num_generations):
	# sort by fitness (lower is better)
	sorted_idx = np.argsort(fitness)
	population = population[sorted_idx]
	fitness = fitness[sorted_idx]

	# keep elites
	new_population = [population[i].copy() for i in range(elitism)]

	# fill rest of population
	while len(new_population) < population_size:
		# --- selection: tournament of 2 ---
		parents_idx = np.random.choice(population_size // 2, 2, replace=False)
		parent1, parent2 = population[parents_idx]

		# --- crossover: average ---
		child = (parent1 + parent2) / 2.0

		# --- mutation ---
		if np.random.rand() < mutation_rate:
			child += mutation_strength * np.random.randn(len(cell_props))

		# clip to [0,1]
		child = np.clip(child, 0.0, 1.0)
		new_population.append(child)

	# replace old population
	population = np.array(new_population)
	fitness = np.array([evaluate_candidate(ind) for ind in population])

	# track best
	gen_best_idx = np.argmin(fitness)
	if fitness[gen_best_idx] < best_error:
		best_error = fitness[gen_best_idx]
		best_candidate = population[gen_best_idx].copy()

	if gen % 10 == 0:
		print(f"Gen {gen:04d}: Best error {best_error:.4f}")

# --- Results ---
print("Best error found:", best_error)
print("Best candidate:", best_candidate)
print("Ground truth:", test_cell_properties_vector)
print("MSE vs test props:", mean_squared_error(best_candidate, test_cell_properties_vector))

plt.scatter(test_cell_properties_vector, best_candidate)
plt.show()