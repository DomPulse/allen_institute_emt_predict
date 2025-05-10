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
full_save_path = r'F:\Big_MET_data\feature_extraction'
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
features_to_ignore_zeros = [
	'time_to_first_spike', 'time_to_last_spike',
	'AP_height', 'AP_width',
	'sag_amplitude', 'sag_ratio1', 'sag_time_constant']

del col_names, counts_df
gc.collect()

for stim_current in currents_to_check:
		
	mutual_info_by_feat_dict = {}
	mutual_info_by_feat_dict['input_features'] = morph_features + gene_names
	
	for output_feature_name in efel_features_col_names:
		
		mean_output_feature = []
		valid_inputs = []
				
		for data_idx in range(len(morpho_trans_data['cell_id'].to_list())):
		#for data_idx in range(20):
			#print(morpho_trans_data['ephys_path'].loc[data_idx], morpho_trans_data['cell_id'].loc[data_idx]) 
			
			cell_id = morpho_trans_data['cell_id'].loc[data_idx]
			
			try:
				this_cell_ephys = pd.read_csv(f'{ephys_path}\{cell_id}.csv')
			except:
				pass
				#print(f'{cell_id} data not found!')
	
			this_current_ephys = this_cell_ephys[np.isclose(this_cell_ephys['current_second_edge'], stim_current, atol = 0.0001)]
			
			#we want to ignore things with value 0 for certain features
			#like we don't want to inclde time to first spike when there were no spikes, ya dig?
			if output_feature_name in features_to_ignore_zeros:
				output_features = this_current_ephys[this_current_ephys[output_feature_name] != 0][output_feature_name].to_numpy()
			else:
				output_features = this_current_ephys[output_feature_name].to_numpy()
				
			if len(output_features) > 0:
				mean_output_feature.append(np.mean(output_features))
				this_input_data = morpho_trans_data.iloc[data_idx, 5:].to_numpy().astype(np.float64)
				valid_inputs.append(list(this_input_data))
			
		mean_output_feature = np.asarray(mean_output_feature)		
		valid_inputs = np.asarray(valid_inputs)
		
		#if there are too few examples the mutual info can't run
		#shouldn't be a problem when running with all the data but it is a good check
		if mean_output_feature.shape[0] > 3:		
		
			#we want to normalize the morphology data between all cells
			#but we want to compare relative levels of gene expression within a given cell
			#so we normalize based on the column (all cells) for the morphology and row for the transcriptome
			morph_end = len(morph_features)
			valid_inputs[:, :morph_end] = norm_col(valid_inputs[:, :morph_end])
			valid_inputs[:, morph_end:] = norm_row(valid_inputs[:, morph_end:])
				
			mutual_info = mir(valid_inputs, mean_output_feature, n_jobs=10)
			mutual_info_by_feat_dict[output_feature_name] = mutual_info
			print(f'{output_feature_name} processed successfully')
			del mutual_info

		del mean_output_feature, valid_inputs, this_current_ephys, this_cell_ephys
		gc.collect()
		
	mutual_info_by_feat_df = pd.DataFrame.from_dict(mutual_info_by_feat_dict)
	mutual_info_by_feat_df.to_csv(os.path.join(full_save_path, f'{stim_current}_mutual_info.csv'))
	print(f'{stim_current} saved successfully')
	del mutual_info_by_feat_df, mutual_info_by_feat_dict
	gc.collect()
