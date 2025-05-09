from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import numpy as np
from pynwb.icephys import CurrentClampSeries, VoltageClampSeries
from math import log10, floor
import os

#never mentioned this anywhere but 
#https://dandiarchive.org/dandiset/000020/draft <- data download
#https://www.youtube.com/watch?v=WLJivIn_Tls&ab_channel=NeurodataWithoutBorders <- useful tutorial 

#chatGPT is bad at the actual NWB processing but great at this
def round_to_sigfig(value, sigfig=1):
	if value == 0:
		return 0.0
	exponent = floor(log10(abs(value)))
	factor = 10 ** (exponent - sigfig + 1)
	return round(value / factor) * factor

SI_PREFIXES = {
	-24: 'y',   # yocto
	-21: 'z',   # zepto
	-18: 'a',   # atto
	-15: 'f',   # femto
	-12: 'p',   # pico
	-9:  'n',   # nano
	-6:  'µ',   # micro
	-3:  'm',   # milli
	-2:  'c',   # centi
	-1:  'd',   # deci
	 0:  '',	# base unit
	 1:  'da',  # deca
	 2:  'h',   # hecto
	 3:  'k',   # kilo
	 6:  'M',   # mega
	 9:  'G',   # giga
	12:  'T',   # tera
	15:  'P',   # peta
	18:  'E',   # exa
	21:  'Z',   # zetta
	24:  'Y',   # yotta
}

def get_si_prefix(value):
	if value == 0:
		return '', 0
	exponent = int(floor(log10(abs(value))))
	si_exponent = (exponent // 3) * 3
	prefix = SI_PREFIXES.get(si_exponent, f"e{si_exponent}")
	return prefix

def crop_on_voltage_flatline(voltage, current, time=None, zero_thresh=1e-3, min_flat_length=100):
	"""
	Crops voltage, current, and optionally time arrays when voltage flattens to ~0.
	
	Args:
		voltage (np.ndarray): Voltage recording (e.g., in mV)
		current (np.ndarray): Current recording (e.g., in pA)
		time (np.ndarray or None): Optional time array
		zero_thresh (float): Threshold to consider "close to 0" (default: 1e-3)
		min_flat_length (int): Number of consecutive near-zero samples to trigger cropping

	Returns:
		Cropped (voltage, current, [time]) tuple
	"""
	voltage = np.asarray(voltage)
	current = np.asarray(current)
	near_zero = np.abs(voltage) < zero_thresh

	# Find where the voltage becomes flat zero-like for a sustained period
	count = 0
	for i, val in enumerate(near_zero):
		if val:
			count += 1
			if count >= min_flat_length:
				cutoff = i - min_flat_length + 1
				break
		else:
			count = 0
	else:
		# No cutoff found, return original
		if time is not None:
			return voltage, current, time
		return voltage, current

	# Slice arrays
	if time is not None:
		return voltage[:cutoff], current[:cutoff], time[:cutoff]
	else:
		return voltage[:cutoff], current[:cutoff]

def pad_to_max_length(arrays, pad_value=0):
	original_lengths = np.array([len(sublist) for sublist in arrays])
	max_length = original_lengths.max()
	
	num_arrays = len(arrays)
	padded = np.zeros((num_arrays, max_length))
	for i in range(num_arrays):
		padded[i, :original_lengths[i]] = arrays[i]
	
	return np.array(padded), original_lengths

#ok end of chatGPT code

# Load your NWB file

def give_me_the_stuff(file_of_interest):
	stim_res_combo = []
	all_times = []
	all_currents = []
	all_volts = []
	with NWBHDF5IO(file_of_interest, 'r') as io:
		nwbfile = io.read()
	
		stims = nwbfile.stimulus
		stim_names = []
		for i in stims:
			#print('stim', i, str(type(stims[i])) == "<class 'pynwb.icephys.CurrentClampStimulusSeries'>")
			stim_names.append(i)
		#print(stim_names)
		
		response = nwbfile.acquisition
		response_names = []
		for i in response:
			#print('res', i, str(type(response[i])) == "<class 'pynwb.icephys.CurrentClampSeries'>")
			response_names.append(i)
		#print(stim_names)
		
		
		for index in range(len(stim_names)): 
			
			this_stim = nwbfile.get_stimulus(stim_names[index])
			this_response = nwbfile.get_acquisition(response_names[index])
			
			stim_units = this_stim.conversion
			response_units = this_response.unit
						  
			if this_stim.unit == 'amperes' and this_response.unit == 'volts' and len(this_stim.data[()]) == len(this_response.data[()]):
				
				#print(this_stim.gain, this_response.gain)
				stim_mag = round_to_sigfig(this_stim.conversion)
				#print(stim_names[index], this_stim.comments, this_stim.control_description)
				res_mag = round_to_sigfig(this_response.conversion)
	
				longest_time = 1000*len(this_stim.data[()])/this_stim.rate #the maximum time in milliseconds
				times = np.linspace(0, longest_time, len(this_stim.data[()]))
				
				volts, current, times = crop_on_voltage_flatline(this_response.data[()], this_stim.data[()], times)
				
				stim_res_combo.append([times, current*stim_mag/(1E-9), volts*res_mag/(1E-3)])
				#stim_res_combo.append([times, current*stim_mag/(this_stim.gain), volts*res_mag/(this_response.gain)])
				all_times.append(times)
				all_currents.append(current*stim_mag/(1E-9))
				all_volts.append(volts*res_mag/(1E-3))

	
	all_times, original_lengths = pad_to_max_length(all_times)
	all_currents, original_lengths = pad_to_max_length(all_currents)
	all_volts, original_lengths = pad_to_max_length(all_volts)
					
	return all_times, all_currents, all_volts, original_lengths

'''
def main():
	folder_path = "D:\\Neuro_Sci\\morph_ephys_trans_stuff\\fine_and_dandi_ephys\\000020\\sub-610663891\\"
	
	files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
	
	all_times, all_currents, all_volts, orginal_lengths = give_me_the_stuff(folder_path, files)
	
	end_stops = []
	right_answers = []
	for i in range(len(all_times)):
		stop_at = orginal_lengths[i]-1
	
		end_stops.append(stop_at)
	
		fig, ax1 = plt.subplots(figsize=(10, 5))
		ax1.set_xlabel('Time (ms)')
		ax1.set_ylabel('Voltage (mV)')
		ax1.plot(all_times[i, :stop_at], all_volts[i, :stop_at], label="real voltage", color = [0, 0, 1])
		ax1.tick_params(axis='y')
		
		ax2 = ax1.twinx()
		ax2.set_ylabel('Current (nA)')
		ax2.plot(all_times[i, :stop_at], all_currents[i, :stop_at], label="injected current", color = [1, 0, 0])
		ax2.tick_params(axis='y')
		
		fig.tight_layout()
		plt.title("Voltage and Current Over Time")
		plt.show()
		
if __name__ == "__main__":
	main()
	gc.collect()
'''