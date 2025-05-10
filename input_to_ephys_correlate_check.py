import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import load_patch_clamp_data as lpcd
import classify_patch_clamp as cpc
import gc
from sklearn.feature_selection import mutual_info_regression as mir

counts_path = r"F:\Big_MET_data\trans_data_cpm\20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
just_trans_path = 'F:\Big_MET_data\metadata_pluse_trans.csv'
mopho_trans_path = 'F:\Big_MET_data\combo_morph_trans.csv'
ephys_path = r'F:\Big_MET_data\derived_ephys'
morpho_trans_data = pd.read_csv(mopho_trans_path)
currents_to_check = [-0.11, -0.09, -0.07, -0.05, -0.03, 0.05, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25]

def norm_col(array):
	min_by_col = np.min(array, axis = 0) 
	array = np.subtract(array, min_by_col)
	max_by_col = np.max(array, axis = 0)
	return np.divide(array, max_by_col)

def norm_row(array):
	min_by_row = np.min(array, axis = 1) 
	min_by_row = min_by_row.reshape(-1,1)
	array = np.subtract(array, min_by_row)
	max_by_row = np.max(array, axis = 1)
	max_by_row = max_by_row.reshape(-1,1)
	return np.divide(array, max_by_row)

counts_df = pd.read_csv(counts_path)
col_names = list(counts_df)
gene_names = list(counts_df[col_names[0]]) #needed for labeling input features :)
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
	'total_volume']
efel_features_col_names = [
	'steady_state_voltage', 'steady_state_voltage_stimend',
	'time_to_first_spike', 'time_to_last_spike',
	'spike_count', 'AP_height', 'AP_width',
	'sag_amplitude', 'sag_ratio1', 'sag_time_constant']

del col_names, counts_df
gc.collect()

for stim_current in currents_to_check:
	
	this_current_ground_truth = []
	this_current_precited = []
	
	mean_volt_changes = []
	valid_inputs = []
	
	#for data_idx in range(len(morpho_trans_data['cell_id'].to_list())):
	for data_idx in range(20):
		#print(morpho_trans_data['ephys_path'].loc[data_idx], morpho_trans_data['cell_id'].loc[data_idx]) 
		
		cell_id = morpho_trans_data['cell_id'].loc[data_idx]
		this_cell_ephys = pd.read_csv(f'{ephys_path}\{cell_id}.csv')

		this_current_ephys = this_cell_ephys[np.isclose(this_cell_ephys['current_second_edge'], stim_current, atol = 0.0001)]
		
		steady_state_voltages = this_current_ephys['steady_state_voltage'].to_numpy()
		steady_state_voltages_stimend = this_current_ephys['steady_state_voltage_stimend'].to_numpy()
			
		voltage_changes = steady_state_voltages_stimend - steady_state_voltages
				
		
		if len(voltage_changes) > 0:
			mean_volt_changes.append(np.mean(voltage_changes))
			this_input_data = morpho_trans_data.iloc[data_idx, 5:].to_numpy().astype(np.float64)
			#print(this_input_data)
			valid_inputs.append(list(this_input_data))
			
	mean_volt_changes = np.asarray(mean_volt_changes)		
	valid_inputs = np.asarray(valid_inputs)
	
	#we want to normalize the morphology data between all cells
	#but we want to compare relative levels of gene expression within a given cell
	#so we normalize based on the column (all cells) for the morphology and row for the transcriptome
	morph_end = len(morph_features)
	valid_inputs[:, :morph_end] = norm_col(valid_inputs[:, :morph_end])
	valid_inputs[:, morph_end:] = norm_row(valid_inputs[:, morph_end:])
		
	mutual_info = mir(valid_inputs, mean_volt_changes, n_jobs=10)
	print(mutual_info.shape)
	