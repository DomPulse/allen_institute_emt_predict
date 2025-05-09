import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import load_patch_clamp_data as lpcd
import classify_patch_clamp as cpc
import gc

just_trans_path = 'F:\Big_MET_data\metadata_pluse_trans.csv'
mopho_trans_path = 'F:\Big_MET_data\combo_morph_trans.csv'
ephys_path = r'F:\Big_MET_data\derived_ephys'
morpho_trans_data = pd.read_csv(mopho_trans_path)

currents_to_check = [-0.11, -0.09, -0.07, -0.05, -0.03, 0.05, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25]

for stim_current in currents_to_check:
	
	this_current_ground_truth = []
	this_current_precited = []
	
	mean_volt_changes = []
	valid_inputs = []
	
	for data_idx in range(5):
		#print(morpho_trans_data['ephys_path'].loc[data_idx], morpho_trans_data['cell_id'].loc[data_idx]) 
		
		cell_id = morpho_trans_data['cell_id'].loc[data_idx]
		this_cell_ephys = pd.read_csv(f'{ephys_path}\{cell_id}.csv')

		this_current_ephys = this_cell_ephys[np.isclose(this_cell_ephys['current_second_edge'], stim_current, atol = 0.0001)]
		
		steady_state_voltages = this_current_ephys['steady_state_voltage'].to_numpy()
		steady_state_voltages_stimend = this_current_ephys['steady_state_voltage_stimend'].to_numpy()
			
		voltage_changes = steady_state_voltages_stimend - steady_state_voltages
				
		
		if len(voltage_changes) > 0:
			print(len(voltage_changes))
			mean_volt_changes.append(np.mean(voltage_changes))
			this_input_data = morpho_trans_data.iloc[data_idx, 5:].to_numpy().astype(np.float64)
			#print(this_input_data)
			valid_inputs.append(list(this_input_data))
			
	mean_volt_changes = np.asarray(mean_volt_changes)		
	valid_inputs = np.asarray(valid_inputs)
	
	#print(valid_inputs.shape, mean_volt_changes.shape)
	print(stim_current, mean_volt_changes)
			

