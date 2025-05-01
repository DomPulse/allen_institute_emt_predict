import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import load_patch_clamp_data as lpcd
import gc

def i_like_em_chunky(time_series, raw_current):
	current_type = 'no class'
	failed = False
	threshed = time_series > 0.025
	kinda_num_chunks = np.sum(threshed) #it's like 2x number of chunks but sometimes there are rising/falling edges that aren't part of a whole chunk
	if kinda_num_chunks > 20 or kinda_num_chunks < 1:
		#if there are more than 4 pulses (3 real and 1 prime) I DON'T want it
		#hell I'm feeling generous, keep some of the step up functions ;)
		failed = True
	
	t = 0
	chunky = []
	pseudo_deriv = np.roll(raw_current, 50) - raw_current
	while t < len(time_series):
		this_chunk = []
		if threshed[t] == 1:
			this_chunk = [t]
			while threshed[t + 1] != 1 and t + 1 < len(time_series) - 1:
				if time_series[t + 1] > 0:
					failed = True
				t += 1
				
			this_chunk.append(t)
			chunky.append(this_chunk)

			t += 1 #skips after end of chunk is found
				
		t += 1
	
	if int(kinda_num_chunks) == 2:
		current_type = 'ramp'
		#then it's probably either ramp or sine
		#we hate sine >:(
		if np.max(raw_current) > 0.005 and np.min(raw_current) < -0.005:
			#the sine wave olscillates so this will probably cut it out but won't cut out most of the negative square waves bc those will have 2 chunks <3
			failed = True
			current_type = 'sine'
			
	elif int(kinda_num_chunks) == 4:
		current_type = 'single pulse'
	
	elif int(kinda_num_chunks) == 8:
		current_type = 'three pulse'
	else:
		current_type = 'step up'
		
		
	#print(np.mean(derivs_in_chunks), np.mean(np.abs(derivs_in_chunks)), np.std(derivs_in_chunks))
		
	return chunky, failed, current_type

def main():
	folder_path = 'F:\\Big_MET_data\\fine_and_dandi_ephys\\000020\\sub-673134052\\'
	
	print(folder_path)
	
	files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
	all_times, all_currents, all_volts, orginal_lengths = lpcd.give_me_the_stuff(folder_path, files)
	diff_current = np.abs(np.roll(all_currents, -1, axis = 1) - all_currents)
	
	for i in range(len(all_times)):
		stop_at = orginal_lengths[i]-1
		
		chunks, is_bad_data, current_type = i_like_em_chunky(diff_current[i, :stop_at], all_currents[i, :stop_at])
	
		fig, axs = plt.subplots(2)
		axs[0].set_xlabel('Time (ms)')
		axs[0].set_ylabel(f"{current_type} {i} failed: {is_bad_data}")
		axs[0].plot(all_times[i, :stop_at], diff_current[i, :stop_at] > 0.025, color = [0, 0, 1])
		for chunk in chunks:
			axs[0].axvspan(all_times[i, chunk[0]], all_times[i, chunk[1]], alpha=0.5, color='red')
		
		axs[1].set_xlabel('Time (ms)')
		axs[1].set_ylabel('Current (nA)')
		axs[1].plot(all_times[i, :stop_at], all_currents[i, :stop_at], color = [1, 0, 0])
		
		plt.show()


if __name__ == "__main__":
	main()
	gc.collect()
