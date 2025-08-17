import numpy as np
import matplotlib.pyplot as plt
import efel
import pandas as pd
import parallel_neuron_model_functional as pnmf

currents_to_test = [-0.1, -0.05, 0.05, 0.1, 0.15]
pos_ephys_properties = ['AHP_depth', 'AP_amplitude_from_voltagebase',
				 'AP_width', 'mean_frequency', 'steady_state_voltage_stimend',
				 'time_to_first_spike', 'voltage_base',
				 'AHP1_depth_from_peak', 'AHP2_depth_from_peak', 'AP_amplitude', 
			     'AP_peak_downstroke', 'AP_peak_upstroke', 'AP_rise_rate_change',
			     'decay_time_constant_after_stim', 'time_constant', 
			     'activation_time_constant', 'deactivation_time_constant', 'inactivation_time_constant']
neg_ephys_properties = ['sag_ratio1', 'sag_time_constant', 'steady_state_voltage_stimend']

cond_names = ['Na_g', 'CaT_g', 'CaS_g',	'A_g',	'KCa_g', 'Kd_g', 'H_g', 'Leak_g']

sim_length = 3000
stim_start = 1000
stim_end = 2000
time_step = 0.05
times = np.linspace(0, sim_length, int(sim_length/time_step))
glob_num_neurons = 5000

glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps = pnmf.initialize_neurons(num_neurons = glob_num_neurons)
glob_Vs, glob_all_params, glob_V_mem, glob_Ca2 = pnmf.the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps, time_step=time_step, sim_length=sim_length)

all_keys = []

for thing in cond_names + ['area', 'cap']:
	all_keys.append(thing)

for current in currents_to_test:
	ephys_list = pos_ephys_properties
	if current < 0:
		ephys_list = neg_ephys_properties
		
	for thing in ephys_list:
		thing_name = f'{current}_{thing}'
		all_keys.append(thing_name)


gaming_df = pd.DataFrame(index=range(glob_num_neurons), columns=all_keys)

for i in range(glob_num_neurons):
	for index, thing in enumerate(cond_names):
		
		gaming_df.loc[i, thing] = glob_all_params[i, index, 0]
		gaming_df.loc[i, 'area'] = glob_areas[i]
		gaming_df.loc[i, 'cap'] = glob_caps[i]
		
for current in currents_to_test:
	glob_curr_prim = pnmf.current_primitive(len(times), 'square', 0, current, stim_start/sim_length, stim_end/sim_length)
	glob_Vs, glob_all_params, glob_pos_stim_V_mem, glob_pos_stim_Ca2 = pnmf.the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps, I_ext = glob_curr_prim, time_step=time_step, sim_length=sim_length)
	
	ephys_list = pos_ephys_properties
	if current < 0:
		ephys_list = neg_ephys_properties
	
	for i in range(glob_num_neurons):
	
		trace1 = {}
	
		trace1['T'] = times
		trace1['V'] = glob_Vs[i]
	
		trace1['stim_start'] = [stim_start]
		trace1['stim_end'] = [stim_end]
	
		traces = [trace1]
	
		efel.set_setting('Threshold', -20)
	
		traces_results = efel.get_feature_values(traces, ephys_list)
	
		# The return value is a list of trace_results, every trace_results
		# corresponds to one trace in the 'traces' list above (in same order)
		for trace_results in traces_results:
			# trace_result is a dictionary, with as keys the requested features
			for feature_name, feature_values in trace_results.items():
				key_name = f'{current}_{feature_name}'
				
				if feature_values is not None:
					this_value = np.mean(feature_values)
				else:
					this_value = 0
				gaming_df.loc[i, key_name] = this_value
		

gaming_df.to_csv('multi_stim_synth_data.csv')