import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc
import efel
import json
from neuron_morphology.swc_io import morphology_from_swc
from neuron_morphology.feature_extractor.data import Data
from neuron_morphology.features.branching.bifurcations import num_outer_bifurcations
from neuron_morphology.feature_extractor.utilities import unnest
from neuron_morphology.feature_extractor.feature_extractor import FeatureExtractor
from neuron_morphology.features.intrinsic import max_branch_order, num_tips, num_branches, num_nodes
from neuron_morphology.features.size import max_euclidean_distance, mean_diameter, total_length, total_surface_area, total_volume
from neuron_morphology.features.path import max_path_distance



def my_feat_pics(local_test_data):

	
	features = [
	    mean_diameter,
	    max_branch_order,
	    max_euclidean_distance,
		max_path_distance,
		num_outer_bifurcations,
		num_branches,
		num_nodes,
		num_tips,
		total_length,
		total_surface_area,
		total_volume
	]	

	results = (
	    FeatureExtractor()
	    .register_features(features)
	    .extract(local_test_data)
	    .results
	)
	
	
	return unnest(results)

def list_all_files(root_folder):
	session_search = []
	session_to_path = {}
	for dirpath, dirnames, filenames in os.walk(root_folder):
		for filename in filenames:
			post_ses = filename.split("ses-")[-1]
			#print(post_ses.split("_")[0])
			pre_other_stuff = post_ses.split("_")[0]
			try:
				session_search.append(int(pre_other_stuff))
				session_to_path[int(pre_other_stuff)] = f'{dirpath}\\{filename}'
			except:
				pass
	return session_search, session_to_path

def main():
	
	save_path = r"F:\arbor_ubuntu"
	morph_path = r"F:\arbor_ubuntu\10k_mouse_pyr_morph"
	dir_list = os.listdir(morph_path)
	
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

	pos_ephys_properties = ['AHP_depth', 'AP_amplitude_from_voltagebase',
					 'AP_width', 'mean_frequency', 'steady_state_voltage_stimend',
					 'time_to_first_spike', 'voltage_base',
					 'AHP1_depth_from_peak', 'AHP2_depth_from_peak', 'AP_amplitude', 
				     'AP_peak_downstroke', 'AP_peak_upstroke', 'AP_rise_rate_change',
				     'decay_time_constant_after_stim', 'time_constant', 
				     'activation_time_constant', 'deactivation_time_constant', 'inactivation_time_constant']
	
	neg_ephys_properties = ['sag_ratio1', 'sag_time_constant', 'steady_state_voltage_stimend']
	
	currents_to_test = [-100, 100, 200]
	ephys_cols = []
	for c in currents_to_test:
		props = pos_ephys_properties
		if c < 0:
			props = neg_ephys_properties

		for prop in props:
			name = f'{c}_{prop}'
			ephys_cols.append(name)
	
	morph_features_df = pd.DataFrame(columns = ['folder', 'file_name', 'count'] + morph_features + ephys_cols + cell_props)
	print(morph_features_df)	
	
	total_count = 0
	for folder in dir_list:
		folder_path = f'{morph_path}\{folder}\CNG version'
		file_list = os.listdir(folder_path)
		for file in file_list:
			try:
				test_data = Data(morphology_from_swc(f'{folder_path}\{file}'))
				#morph = morphology_from_swc(r'F:\arbor_ubuntu\swc_test_files\example_morphology.swc')
				#test_data = Data(morph)
				features = my_feat_pics(test_data)
				new_row_dict = {}
				new_row_dict['folder'] = folder
				new_row_dict['file_name'] = file
				new_row_dict['count'] = total_count
				for key, val in features.items():
					new_row_dict[key] = [val]
				new_row_df = pd.DataFrame(new_row_dict)
				morph_features_df = pd.concat([morph_features_df, new_row_df], ignore_index=True)
			except:
				pass
			total_count += 1
		if total_count > 1000:
			break

	morph_features_df.to_csv(f'{save_path}//morphology_features.csv')
	del morph_features_df
	gc.collect()
		
if __name__ == "__main__":
	main()
	 