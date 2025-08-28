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

#i'll be real, chatGPT wrote the next 2 functions, i don't care enough about parsing jsons to do that
def _maybe_float(x):
	try:
		return float(x)
	except (TypeError, ValueError):
		return x

def cell_prop_stuff(json_path, append_dict):
	"""
	Parse an Allen/BBP-style json and add:
	  - passive soma Ra  -> key: 'soma_ra'
	  - genome params	-> key: f"{section}_{name}" (e.g., 'soma_g_pas')

	Parameters
	----------
	json_path : str
		Path to the JSON file.
	append_dict : dict
		Dict to populate (mutated in-place and also returned).

	Returns
	-------
	dict
		The same dict with added keys/values.
	"""
	with open(json_path, 'r') as f:
		data = json.load(f)
	
	# soma_ra from "passive"
	if "passive" in data and data["passive"]:
		soma_ra = data["passive"][0].get("ra", None)
		if soma_ra is not None:
			append_dict['soma_ra'] = _maybe_float(soma_ra)

	# genome: section + name
	for entry in data.get("genome", []):
		section = entry.get("section", "unknown")
		name = entry.get("name", "unknown")
		value = entry.get("value", None)

		key = f"{section}_{name}"
		# if a duplicate key somehow appears, last one wins; customize if needed
		append_dict[key] = _maybe_float(value)
		#print(key, value)

	return append_dict

def efel_stuff(ephys_df, append_dict, ephys_features, prefix, start_time, end_time, crop_end):
	subset_df = ephys_df[(ephys_df['t/ms'] >= start_time) & (ephys_df['t/ms'] <= crop_end)].reset_index(drop=True)
	trace1 = {}

	# Set the 'T' (=time) key of the trace
	trace1['T'] = subset_df['t/ms']

	# Set the 'V' (=voltage) key of the trace
	trace1['V'] = subset_df['U/mV']

	# Set the 'stim_start' (time at which a stimulus starts, in ms)
	# key of the trace
	# Warning: this need to be a list (with one element)
	trace1['stim_start'] = [start_time]

	# Set the 'stim_end' (time at which a stimulus end) key of the trace
	# Warning: this need to be a list (with one element)
	trace1['stim_end'] = [end_time]

	# Multiple traces can be passed to the eFEL at the same time, so the
	# argument should be a list
	traces = [trace1]

	# set the threshold for spike detection to -20 mV
	efel.set_setting('Threshold', -20)

	# Now we pass 'traces' to the efel and ask it to calculate the feature
	# values
	traces_results = efel.get_feature_values(traces, ephys_features)
	
	# The return value is a list of trace_results, every trace_results
	# corresponds to one trace in the 'traces' list above (in same order)
	for trace_results in traces_results:
		# trace_result is a dictionary, with as keys the requested features
		for feature_name, feature_values in trace_results.items():
			true_name = f'{prefix}_{feature_name}'
			if feature_values is not None:
				true_value = np.mean(feature_values)
			else:
				true_value = 0
			append_dict[true_name] = true_value
	return append_dict

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
	morph_folder = r"F:\arbor_ubuntu\10k_mouse_pyr_morph"
	ephys_folder = r"F:\arbor_ubuntu\MORE_MOUSE_BITES"
	cell_prop_folder = r"F:\arbor_ubuntu\10k_randomized_jsons"
	dir_list = os.listdir(morph_folder)
	
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

	pos_ephys_properties = ['steady_state_voltage', 'steady_state_voltage_stimend',
					 'time_to_first_spike', 'time_to_last_spike',
					 'spike_count', 'AP_height', 'AP_width',
					 'AHP_depth', 'AP_amplitude_from_voltagebase', 'mean_frequency', 'voltage_base', 
					 'AHP1_depth_from_peak', 'AHP2_depth_from_peak', 'AP_amplitude', 
					 'AP_peak_downstroke', 'AP_peak_upstroke', 'AP_rise_rate_change',
					 'decay_time_constant_after_stim', 'time_constant']
	
	neg_ephys_properties = ['sag_ratio1', 'sag_time_constant', 'steady_state_voltage_stimend']
	
	currents_to_test = [-200, -100, -50, 50, 100, 150, 200]
	ephys_cols = []
	for c in currents_to_test:
		props = pos_ephys_properties
		if c < 0:
			props = neg_ephys_properties

		for prop in props:
			name = f'{c}_{prop}'
			ephys_cols.append(name)
	
	morph_features_df = pd.DataFrame(columns = ['folder', 'file_name', 'count'] + ephys_cols + cell_props)
	print(morph_features_df)	
	
	total_count = 0
	for total_count in range(10000):

		try:

			new_row_dict = {}
			new_row_dict['folder'] = 'kimura'
			new_row_dict['file_name'] = 'P7-control-for-TRAK2-MBD-14-003.CNG.swc'
			new_row_dict['count'] = total_count
			
			'''
			test_data = Data(morphology_from_swc(f'{folder_path}\{file}'))
			features = my_feat_pics(test_data)
			for key, val in features.items():
				new_row_dict[key] = [val]
			'''
			
			ephys_path = f"{ephys_folder}\\kimura_P7-control-for-TRAK2-MBD-14-003.CNG.swc_{total_count}.csv"
			#ephys_path = r'F:\arbor_ubuntu\ephys_single_morph\anton_KO-1-DIV-TTb.CNG.swc_8103.csv'
			raw_ephys_trace = pd.read_csv(ephys_path)
			new_row_dict = efel_stuff(raw_ephys_trace, new_row_dict, neg_ephys_properties, -200, 500, 1700, 2200)
			new_row_dict = efel_stuff(raw_ephys_trace, new_row_dict, neg_ephys_properties, -100, 2200, 3400, 3900)
			new_row_dict = efel_stuff(raw_ephys_trace, new_row_dict, neg_ephys_properties, -50, 3900, 5100, 5600)
			new_row_dict = efel_stuff(raw_ephys_trace, new_row_dict, pos_ephys_properties, 50, 5600, 6800, 7300)
			new_row_dict = efel_stuff(raw_ephys_trace, new_row_dict, pos_ephys_properties, 100, 7300, 8500, 9000)
			new_row_dict = efel_stuff(raw_ephys_trace, new_row_dict, pos_ephys_properties, 150, 9000, 10200, 10700)
			new_row_dict = efel_stuff(raw_ephys_trace, new_row_dict, pos_ephys_properties, 200, 10700, 11900, 12500)
			
			silly_buffer = 'random_genome' #no clue why this is necessary
			json_genome_path = f"{cell_prop_folder}\{silly_buffer}_{total_count}.json"
			new_row_dict = cell_prop_stuff(json_genome_path, new_row_dict)
			#print(new_row_dict)
			new_row_df = pd.DataFrame([new_row_dict])
			#print('gaming?')
			morph_features_df = pd.concat([morph_features_df, new_row_df], ignore_index=True)
			print(ephys_path)
		except:
			pass

	morph_features_df.to_csv(f'{save_path}//synth_data_one_morph_more_stims_v2.csv')
	#morph_features_df.to_csv(f'{save_path}//morph_and_genome.csv')
	del morph_features_df
	gc.collect()
		
if __name__ == "__main__":
	main()
	 