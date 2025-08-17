import numpy as np
import matplotlib.pyplot as plt
import efel
import pandas as pd
import parallel_neuron_model_functional as pnmf

pos_ephys_properties = ['AHP_depth', 'AP_amplitude_from_voltagebase',
				 'AP_width', 'mean_frequency', 
				 'time_to_first_spike', 'voltage_base',
				 'AHP1_depth_from_peak', 'AHP2_depth_from_peak', 'AP_amplitude', 
			     'AP_peak_downstroke', 'AP_peak_upstroke', 'AP_rise_rate_change',
			     'decay_time_constant_after_stim', 'time_constant', 
			     'activation_time_constant', 'deactivation_time_constant', 'inactivation_time_constant']
neg_ephys_properties = ['sag_ratio1', 'sag_time_constant']
both_ephys = ['steady_state_voltage_stimend']

cond_names = ['Na_g', 'CaT_g', 'CaS_g',	'A_g',	'KCa_g', 'Kd_g', 'H_g', 'Leak_g']

sim_length = 3000
stim_start = 1000
stim_end = 2000
time_step = 0.05
times = np.linspace(0, sim_length, int(sim_length/time_step))
glob_num_neurons = 10000
glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps = pnmf.initialize_neurons(num_neurons = glob_num_neurons)
	
glob_pos_curr_prim = pnmf.current_primitive(len(times), 'square', 0, 0.15, stim_start/sim_length, stim_end/sim_length)
glob_neg_curr_prim = pnmf.current_primitive(len(times), 'square', 0, -0.05, stim_start/sim_length, stim_end/sim_length)
glob_Vs, glob_all_params, glob_V_mem, glob_Ca2 = pnmf.the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps, time_step=time_step, sim_length=sim_length)
glob_pos_stim_Vs, glob_all_params, glob_pos_stim_V_mem, glob_pos_stim_Ca2 = pnmf.the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps, I_ext = glob_pos_curr_prim, time_step=time_step, sim_length=sim_length)
glob_neg_stim_Vs, glob_all_params, glob_neg_stim_V_mem, glob_neg_stim_Ca2 = pnmf.the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps, I_ext = glob_neg_curr_prim, time_step=time_step, sim_length=sim_length)

all_keys = []
gaming_dict = {}
for thing in cond_names + neg_ephys_properties + pos_ephys_properties + ['area', 'cap']:
	gaming_dict[thing] = []
	all_keys.append(thing)
for thing in both_ephys:
	pos_name = f'pos_{thing}'
	neg_name = f'neg_{thing}'
	gaming_dict[pos_name] = []
	gaming_dict[neg_name] = []
	all_keys.append(pos_name)
	all_keys.append(neg_name)
	
for i in range(glob_num_neurons):
	'''
	plt.title(i)
	plt.plot(times, glob_Vs[i])
	plt.show()
	'''
	
	data_dict = {}
	for index, value in enumerate(cond_names):
		data_dict[value] = glob_all_params[i, index, 0]
	
	data_dict['area'] = glob_areas[i]
	data_dict['cap'] = glob_caps[i]
	
	
	trace1 = {}

	trace1['T'] = times
	trace1['V'] = glob_pos_stim_Vs[i]

	trace1['stim_start'] = [stim_start]
	trace1['stim_end'] = [stim_end]

	traces = [trace1]

	efel.set_setting('Threshold', -20)

	traces_results = efel.get_feature_values(traces, pos_ephys_properties + both_ephys)

	# The return value is a list of trace_results, every trace_results
	# corresponds to one trace in the 'traces' list above (in same order)
	for trace_results in traces_results:
		# trace_result is a dictionary, with as keys the requested features
		for feature_name, feature_values in trace_results.items():
			if feature_name in both_ephys:
				key_name = f'pos_{feature_name}'
			else:
				key_name = f'{feature_name}'
			
			if feature_values is not None:
				this_value = np.mean(feature_values)
			else:
				this_value = 0
			data_dict[key_name] = this_value
	
	trace1 = {}

	trace1['T'] = times
	trace1['V'] = glob_neg_stim_Vs[i]

	trace1['stim_start'] = [stim_start]
	trace1['stim_end'] = [stim_end]

	traces = [trace1]

	efel.set_setting('Threshold', -20)

	traces_results = efel.get_feature_values(traces, neg_ephys_properties + both_ephys)

	# The return value is a list of trace_results, every trace_results
	# corresponds to one trace in the 'traces' list above (in same order)
	for trace_results in traces_results:
		# trace_result is a dictionary, with as keys the requested features
		for feature_name, feature_values in trace_results.items():
			if feature_name in both_ephys:
				key_name = f'neg_{feature_name}'
			else:
				key_name = f'{feature_name}'
			
			if feature_values is not None:
				this_value = np.mean(feature_values)
			else:
				this_value = 0
			data_dict[key_name] = this_value

	if data_dict['time_to_first_spike'] >= 0:
		for thing in all_keys:
			gaming_dict[thing].append(data_dict[thing]) 

print(gaming_dict)
gaming_df = pd.DataFrame.from_dict(gaming_dict)
gaming_df.to_csv('even_more_dual_stim_proxy_synth_data.csv')