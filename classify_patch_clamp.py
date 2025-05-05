import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import load_patch_clamp_data as lpcd
import gc
import efel

efel_features = ['voltage_base', 'time_to_first_spike', 'time_to_last_spike',
				  'sag_amplitude', 'sag_ratio1', 'sag_time_constant',
				  'minimum_voltage', 'maximum_voltage', 'spike_count']

def i_like_em_chunky(stop_at, raw_current, the_times, raw_voltage):
	
	curr_diff = np.abs(np.roll(raw_current, -1) - raw_current)[:stop_at]
		
	threshed = curr_diff > 0.025
	num_edge = np.sum(threshed) #it's like 2x number of chunks but sometimes there are rising/falling edges that aren't part of a whole chunk
	if int(num_edge) != 4:
		#only take square pulses
		return None
	
	t = 0
	chunky = []
	max_t = len(curr_diff) - 2 #i'm just ignoring shit suitably near the end
	while t < max_t:
		this_chunk = []
		if threshed[t] == 1:
			#print(raw_current[t+1])
			this_chunk = [t]
			while threshed[t + 1] != 1 and t + 1 < max_t:
				if curr_diff[t + 1] > 0:
					return None
				t += 1
				
			this_chunk.append(t)
			chunky.append(this_chunk)

			t += 1 #skips after end of chunk is found
				
		t += 1
	
			
		
	#print(np.mean(derivs_in_chunks), np.mean(np.abs(derivs_in_chunks)), np.std(derivs_in_chunks))
	chunky = np.asarray(chunky)
	has_priming = np.isclose(raw_current[chunky[0, 0] + 1], 0.05)
	
	if not has_priming:
		return None
	
	integrated_current = np.sum(raw_current)
	if np.abs(integrated_current) < 1000:
		return None
	
	second_time_start_idx = chunky[1, 0] + 1
	second_time_end_idx = chunky[1, 1] + 1
	second_time_start = the_times[second_time_start_idx]
	second_time_end = the_times[second_time_end_idx]
	current_second_edge = raw_current[second_time_start_idx]

	derived_current_params = [second_time_start, second_time_end, current_second_edge, the_times[stop_at - 1]]
	derived_efeatures = []
	
	try:
		trace1 = {}
		trace1['T'] = the_times[:stop_at]
		trace1['V'] = raw_voltage[:stop_at]
		trace1['stim_start'] = [second_time_start]
		trace1['stim_end'] = [second_time_end]
		traces = [trace1]
		efel.set_setting('Threshold', -20)
		traces_results = efel.get_feature_values(traces, efel_features)

		for trace_results in traces_results:
			for feature_name, feature_values in trace_results.items():
		
				if feature_values is None:
					this_feat_val = 0
				else:
					this_feat_val = feature_values[0]
				derived_efeatures.append(this_feat_val)
		
	except:
		return None

	
	return chunky, False, curr_diff, derived_current_params, derived_efeatures

def main():
	folder_path = 'F:\\Big_MET_data\\fine_and_dandi_ephys\\000020\\sub-673134052\\'
	
	print(folder_path)
	
	files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
	all_times, all_currents, all_volts, orginal_lengths = lpcd.give_me_the_stuff(folder_path, files)
	
	for i in range(len(all_times)):
		stop_at = orginal_lengths[i]-2
		
		try:
			chunks, is_bad_data, diff_current, derived_current_params, derived_efeatures = i_like_em_chunky(stop_at, all_currents[i], all_times[i], all_volts[i])
			
			print(derived_efeatures)
		
			fig, axs = plt.subplots(2)
			axs[0].set_xlabel('Time (ms)')
			axs[0].set_ylabel(f"{derived_current_params[0]} {i} failed: {is_bad_data}")
			axs[0].plot(all_times[i, :stop_at], diff_current > 0.025, color = [0, 0, 1])
			for chunk in chunks:
				axs[0].axvspan(all_times[i, chunk[0]], all_times[i, chunk[1]], alpha=0.5, color='red')
			
			axs[1].set_xlabel('Time (ms)')
			axs[1].set_ylabel('Current (nA)')
			axs[1].plot(all_times[i, :stop_at], all_currents[i, :stop_at], color = [1, 0, 0])
			
			plt.show()
		
		except:
			pass


if __name__ == "__main__":
	main()
	gc.collect()
