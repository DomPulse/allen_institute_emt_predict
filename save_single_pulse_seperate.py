import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import load_patch_clamp_data as lpcd
import classify_patch_clamp as cpc
import gc

just_trans_path = 'F:\Big_MET_data\metadata_pluse_trans.csv'
mopho_trans_path = 'F:\Big_MET_data\combo_morph_trans.csv'
full_save_path = r'F:\Big_MET_data\derived_ephys'
morpho_trans_data = pd.read_csv(mopho_trans_path)

efel_features_col_names = ['steady_state_voltage', 'steady_state_voltage_stimend',
				 'time_to_first_spike', 'time_to_last_spike',
				 'spike_count', 'AP_height', 'AP_width',
				  'sag_amplitude', 'sag_ratio1', 'sag_time_constant']

derived_current_params_col_names = ['second_time_start', 'second_time_end', 'current_second_edge', 'end_time']

for data_idx in range(5):
	print(morpho_trans_data['ephys_path'].loc[data_idx], morpho_trans_data['cell_id'].loc[data_idx]) 
	
	single_ephys_file_path = morpho_trans_data['ephys_path'].loc[data_idx]
	cell_id = morpho_trans_data['cell_id'].loc[data_idx]

	all_times, all_currents, all_volts, orginal_lengths = lpcd.give_me_the_stuff(single_ephys_file_path)
	
	ephys_entry_data = []
	
	for i in range(len(all_times)):
		stop_at = orginal_lengths[i]-1
				
		try:
			chunks, is_bad_data, diff_current, derived_current_params, derived_efeatures = cpc.i_like_em_chunky(stop_at, all_currents[i], all_times[i], all_volts[i])
			if not is_bad_data:
				#print(folder_path, i, derived_efeatures)
			
				ephys_entry_data.append([i] + derived_current_params + derived_efeatures)
		except:
			pass
	
	pd.DataFrame(ephys_entry_data, columns = ['entry_idx'] + derived_current_params_col_names + efel_features_col_names).to_csv(os.path.join(full_save_path, f'{cell_id}.csv'))
	print(f'saved {cell_id}')
	del ephys_entry_data
	gc.collect()