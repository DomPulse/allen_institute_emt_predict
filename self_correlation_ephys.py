'''
I should really start putting these at the start of all my things
I want to place an upper bound on what my model accuracy could be
so, I am going to look at half the ephys features for a given current stimulus and see how well I can predict the other half
and take like an r^2 value or something, maybe some nice scatter plots
'''

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
		
	for data_idx in range(5):
		#print(morpho_trans_data['ephys_path'].loc[data_idx], morpho_trans_data['cell_id'].loc[data_idx]) 
		
		cell_id = morpho_trans_data['cell_id'].loc[data_idx]
		this_cell_ephys = pd.read_csv(f'{ephys_path}\{cell_id}.csv')

		this_current_ephys = this_cell_ephys[np.isclose(this_cell_ephys['current_second_edge'], stim_current, atol = 0.0001)]
		
		steady_state_voltages = this_current_ephys['steady_state_voltage'].to_numpy()
		steady_state_voltages_stimend = this_current_ephys['steady_state_voltage_stimend'].to_numpy()
			
		voltage_changes = steady_state_voltages_stimend - steady_state_voltages
		np.random.shuffle(voltage_changes) #don't want to bias it by looking at the same session over and over
		
		if voltage_changes is not None:
		
			measure_frac_idx = int(0.8*len(voltage_changes))
			measure_mean = np.mean(voltage_changes[:measure_frac_idx])
			measure_std = np.std(voltage_changes[:measure_frac_idx])
			
			pred_frac_true_values = voltage_changes[measure_frac_idx:]
			pred_frac_pred_values = np.random.normal(measure_mean, measure_std, len(pred_frac_true_values))
			
			this_current_precited += list(pred_frac_pred_values)
			this_current_ground_truth += list(pred_frac_true_values)
			

	plt.scatter(this_current_precited, this_current_ground_truth)
	plt.title(f'Current Stimulus Magnitude {stim_current} (nA)')
	plt.xlabel('predicted voltage change (mV)')
	plt.ylabel('true voltage change (mV)')
	plt.show()
		
		
