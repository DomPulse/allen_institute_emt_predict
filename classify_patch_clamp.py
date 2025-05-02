import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import load_patch_clamp_data as lpcd
import gc
import efel

efel_features = ['voltage_base', 'time_to_first_spike', 'time_to_last_spike',
				  'sag_amplitude', 'sag_ratio1', 'sag_time_constant',
				  'minimum_voltage', 'AP1_width', 'APlast_width', 
				  'deactivation_time_constant', 'activation_time_constant', 'inactivation_time_constant']

def i_like_em_chunky(stop_at, raw_current, the_times, raw_voltage):
	trace1 = {}
	trace1['T'] = the_times[:stop_at]
	trace1['V'] = raw_voltage[:stop_at]
	trace1['stim_start'] = [0]
	trace1['stim_end'] = [stop_at]
	traces = [trace1]
	efel.set_setting('Threshold', -20)
	traces_results = efel.get_feature_values(traces, ['time_to_first_spike'])
	for trace_results in traces_results:
		for feature_name, feature_values in trace_results.items():
			print(feature_values)
	
	
	curr_diff = np.abs(np.roll(raw_current, -1) - raw_current)[:stop_at]
	
	current_type = 'no class'
	failed = False
	threshed = curr_diff > 0.025
	num_edge = np.sum(threshed) #it's like 2x number of chunks but sometimes there are rising/falling edges that aren't part of a whole chunk
	if num_edge > 20 or num_edge < 1:
		#if there are more than 4 pulses (3 real and 1 prime) I DON'T want it
		#hell I'm feeling generous, keep some of the step up functions ;)
		failed = True
	
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
					failed = True
				t += 1
				
			this_chunk.append(t)
			chunky.append(this_chunk)

			t += 1 #skips after end of chunk is found
				
		t += 1
	
	if int(num_edge) == 2:
		current_type = 'ramp'
		#then it's probably either ramp or sine
		#we hate sine >:(
		if np.max(raw_current) > 0.005 and np.min(raw_current) < -0.005:
			#the sine wave olscillates so this will probably cut it out but won't cut out most of the negative square waves bc those will have 2 chunks <3
			failed = True
			current_type = 'sine'
			
	elif int(num_edge) == 4:
		current_type = 'single pulse'
	
	elif int(num_edge) == 8:
		current_type = 'three pulse'
	else:
		current_type = 'step up'
		
		
	#print(np.mean(derivs_in_chunks), np.mean(np.abs(derivs_in_chunks)), np.std(derivs_in_chunks))
	chunky = np.asarray(chunky)
	has_priming = np.isclose(raw_current[chunky[0, 0] + 1], 0.05)
	if not has_priming:
		failed = True
		
	if len(chunky) > 1:
		second_time_idx = chunky[1, 0] + 1
		last_time_idx = chunky[-1, 0] + 1
		second_time = the_times[second_time_idx]
		last_time = the_times[last_time_idx]
		current_second_edge = raw_current[second_time_idx]
		current_last_edge = raw_current[last_time_idx]
		
	else:
		second_time = 0
		last_time = 0
		current_second_edge = 0
		current_last_edge = 0
	integrated_current = np.sum(raw_current)
	derived_current_params = [current_type, second_time, last_time, current_second_edge, current_last_edge, integrated_current, np.min(raw_current[:stop_at]), np.max(raw_current[:stop_at]), num_edge, the_times[stop_at - 1]]
	
	print(derived_current_params)
	
	return chunky, failed, derived_current_params, curr_diff

def main():
	folder_path = 'F:\\Big_MET_data\\fine_and_dandi_ephys\\000020\\sub-673134052\\'
	
	print(folder_path)
	
	files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
	all_times, all_currents, all_volts, orginal_lengths = lpcd.give_me_the_stuff(folder_path, files)
	
	for i in range(len(all_times)):
		stop_at = orginal_lengths[i]-2
		
		chunks, is_bad_data, derived_current_params, diff_current = i_like_em_chunky(stop_at, all_currents[i], all_times[i], all_volts[i])
	
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
		
		print()


if __name__ == "__main__":
	main()
	gc.collect()
