import numpy as np
import matplotlib.pyplot as plt
import load_patch_clamp_data as lpcd
import parallel_neuron_model_functional as pnmf
import os
import scipy.interpolate as interp
from scipy.interpolate import interp1d
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pickle

#taken from this bad boi https://journals.physiology.org/doi/epdf/10.1152/jn.00641.2003
#they explicity set the membrane surface area which I should change to reflect morphology when I integrate with the Allen data
#the only thing they vary is the conductance, I geuss I could vary more but let's copy them first
#all voltages in mV, all times in ms
#currents should be in nA
#https://www.jneurosci.org/content/jneuro/18/7/2309.full.pdf
#aparently same model^ ?

def derived_ephys_props(volts_over_time, kind_of_time_step, expected_mean_volt = -70):
	#we want to look at number of spikes and their width
	#-20 is arbitrariy but consistent and good enough for government work
	times_over_neg_20 = volts_over_time > (-20) 
	to_subtract = np.roll(times_over_neg_20, 1) #shift everything over 1 index with a 0 padding
	
	num_spikes = np.sum(np.abs(np.add(times_over_neg_20, -1*to_subtract)))/2
	
	mean_spike_width = kind_of_time_step*500
	if num_spikes != 0:
		mean_spike_width = kind_of_time_step*np.sum(times_over_neg_20)/num_spikes
		
	#these are normalized to not blow up an error function
	first_spike_time = np.argmax(times_over_neg_20 == 1)/(len(times_over_neg_20))
	last_spike_time = (len(times_over_neg_20) - np.argmax(times_over_neg_20[::-1] == 1))/(len(times_over_neg_20))
	
	#kind of sort of normalized
	peak_volt = -1*np.max(volts_over_time)/expected_mean_volt
	sag_volt = np.min(volts_over_time)/expected_mean_volt
	mean_volt = np.mean(volts_over_time)/expected_mean_volt
	
	return [mean_volt, peak_volt, sag_volt, num_spikes, mean_spike_width, first_spike_time, last_spike_time]

def mutate(baby_param):
	for i in range(8):
		baby_param[i, 0] = np.max([np.random.normal(baby_param[i, 0], 0.01 + np.abs(baby_param[i, 0])/10), 0])
	baby_param[7, 1] = np.random.normal(baby_param[7, 1], np.abs(baby_param[7, 1])/10)
	return baby_param

def death_and_sex(current_params, current_error):
	#current as in this point in time, not current as in flow of ions
	#hey im walking here
	#death and sex? two things im not having ;)
	#yes i am going insane but nobody will read this let me inform you
	max_error = np.max(current_error)
	probs = 1 - np.exp(-1*(current_error - max_error)) #higher probs will have higher chance of being taken, lowest has a 0% chance
	taken = probs > np.random.rand(glob_num_neurons) #are these eligible for being taken to breed?
	golden_ticket = np.random.randint(glob_num_neurons)
	taken[golden_ticket] = 1 #at least one will always be taken, its a near statistical certainty that one will make it be in the case of extreme homogeneity and bad luck i don't want it to crash
	new_params = 0*current_params
	one_indices = np.where(taken == 1)[0]  # Get indices where arr == 1
	for n in range(glob_num_neurons):
		mommy_idx = np.random.choice(one_indices)
		daddy_idx = np.random.choice(one_indices)
		new_params[n] = np.add(current_params[mommy_idx], current_params[daddy_idx])/2
		new_params[n] = mutate(new_params[n])
	
	return new_params

folder_path = "D:\\Neuro_Sci\\morph_ephys_trans_stuff\\fine_and_dandi_ephys\\000020\\sub-610663891\\"

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
all_times, all_currents, all_volts, orginal_lengths = lpcd.give_me_the_stuff(folder_path, files)
all_currents = all_currents*10 #this is complete ad hoc nonsense I don't care if it's wrong, we're 1 step away from numerology

epochs = 1000
glob_num_neurons = 1000
glob_sim_length = np.max(all_times)
glob_time_step = 0.05
ephys_time_step = glob_sim_length/(len(all_times[0]))
glob_sim_steps = int(glob_sim_length/glob_time_step)

end_stops = []
kind_of_resting_potential = []
right_answers = []
for i in range(len(all_times)):
	stop_at = orginal_lengths[i]-1
	if all_currents[i, stop_at - 1] == 0:
		kind_of_resting_potential.append(all_volts[i, stop_at - 1])
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

	right_answers.append(derived_ephys_props(all_volts[i, :stop_at], ephys_time_step))
	

	#print(derived_ephys_props(all_volts[i, :stop_at], ephys_time_step))
end_stops = np.asarray(end_stops)


all_time_max_error = 3000
glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps = pnmf.initialize_neurons(num_neurons = glob_num_neurons)

for e in range(epochs):
	
	train_on = np.random.randint(0, len(all_times)) #picks a random ephys experiment to serve as dataset
	this_right_answer = right_answers[train_on]
	
	stop_at = end_stops[train_on]-1
	glob_sim_length = all_times[train_on, stop_at]
	glob_sim_steps = int(glob_sim_length/glob_time_step)
	
	x_old = np.linspace(0, 1, stop_at)
	x_new = np.linspace(0, 1, glob_sim_steps)
	stim_current = all_currents[train_on, :stop_at]
	exp_volt = all_volts[train_on, :stop_at]
	template_interp_curr = interp1d(x_old, stim_current, fill_value='extrapolate')
	template_interp_volt = interp1d(x_old, exp_volt, fill_value='extrapolate')
	interp_stim_current = template_interp_curr(x_new)
	interp_exp_volt = template_interp_volt(x_new)
	
	
	glob_Vs, glob_all_params, glob_V_mem, glob_Ca2 = pnmf.the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps, sim_length = 450, time_step = glob_time_step) #lets the neurons obtain rest folks
	glob_Vs, glob_all_params, glob_V_mem, glob_Ca2 = pnmf.the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps, I_ext = interp_stim_current, sim_length = glob_sim_length, time_step = glob_time_step)
	
	'''
	plt.plot(interp_stim_current)
	plt.show()
	
	plt.plot(interp_exp_volt)
	plt.show()
	'''
	
	all_errors = np.zeros(glob_num_neurons)
	for n in range(glob_num_neurons):

		this_neuron_answer = derived_ephys_props(glob_Vs[n], glob_time_step)
		
		try:
			mse1 = mean_squared_error(this_right_answer, this_neuron_answer)
			mse2 = mean_squared_error(glob_Vs[n], interp_exp_volt)
			all_errors[n] = mse1 + mse2
		except:
			all_errors[n] = all_time_max_error #high and arbitrary, wish i could do this smarter
			
		
	if np.max(all_errors) > all_time_max_error:
		all_time_max_error = np.max(all_errors)
	
	min_idx = np.argmin(all_errors)
	
	this_end = end_stops[train_on]
	
	plt.plot(glob_Vs[min_idx], label="copy voltage", color = [0, 0, 1])
	plt.plot(interp_exp_volt, label="real voltage", color = [0, 1, 0])
	plt.legend()
	plt.show()
	
	
	mean_mse = np.mean(all_errors)
	print(e, train_on, mean_mse)
	if (e%10 == 0):
		np.save('all_params_'+str(e), glob_all_params)
		np.save('their_mse_'+str(e), all_errors)
	
	glob_all_params = death_and_sex(glob_all_params, all_errors)
	