import numpy as np
import matplotlib.pyplot as plt
import efel
import pandas as pd
import parallel_neuron_model_functional as pnmf

efel_features = ['steady_state_voltage', 'steady_state_voltage_stimend',
				 'time_to_first_spike', 'time_to_last_spike',
				 'spike_count', 'AP_height', 'AP_width']
cond_names = ['Na_g', 'CaT_g', 'CaS_g',	'A_g',	'KCa_g', 'Kd_g', 'H_g', 'Leak_g']

sim_length = 3000
stim_start = 1000
stim_end = 2000
time_step = 0.05
times = np.linspace(0, sim_length, int(sim_length/time_step))
glob_num_neurons = 5000
glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps = pnmf.initialize_neurons(num_neurons = glob_num_neurons)
	
glob_curr_prim = pnmf.current_primitive(len(times), 'square', 0, 0.15, stim_start/sim_length, stim_end/sim_length)
glob_Vs, glob_all_params, glob_V_mem, glob_Ca2 = pnmf.the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps, time_step=time_step, sim_length=sim_length)
glob_Vs, glob_all_params, glob_V_mem, glob_Ca2 = pnmf.the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps, I_ext = glob_curr_prim, time_step=time_step, sim_length=sim_length)

gaming_dict = {}
for thing in cond_names+efel_features:
	gaming_dict[thing] = []
	
for i in range(glob_num_neurons):
	'''
	plt.title(i)
	plt.plot(times, glob_Vs[i])
	plt.show()
	'''
	
	data_dict = {}
	for index, value in enumerate(cond_names):
		data_dict[value] = glob_all_params[i, index, 0]
	
	trace1 = {}

	trace1['T'] = times
	trace1['V'] = glob_Vs[i]

	trace1['stim_start'] = [stim_start]
	trace1['stim_end'] = [stim_end]

	traces = [trace1]

	efel.set_setting('Threshold', -20)

	traces_results = efel.get_feature_values(traces,
										   efel_features)

	# The return value is a list of trace_results, every trace_results
	# corresponds to one trace in the 'traces' list above (in same order)
	for trace_results in traces_results:
		# trace_result is a dictionary, with as keys the requested features
		for feature_name, feature_values in trace_results.items():
			if feature_values is not None:
				data_dict[feature_name] = np.mean(feature_values)
			else:
				data_dict[feature_name] = 0

	if data_dict['time_to_first_spike'] >= 0:
		for thing in cond_names+efel_features:
			gaming_dict[thing].append(data_dict[thing]) 

print(gaming_dict)
gaming_df = pd.DataFrame.from_dict(gaming_dict)
gaming_df.to_csv('gamer_moment.csv')