import numpy as np
import matplotlib.pyplot as plt
import load_patch_clamp_data as lpcd
import os
import scipy.interpolate as interp
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

def m_or_h_infin(V, top, bottom, Ca_adj = False, Ca = 1):
	mult = 1
	if Ca_adj:
		mult = Ca/(Ca + 3)
	denom = 1 + np.exp((V + top)/bottom)
	return mult/denom

def most_taus(V, numer, top, bottom, offset):
	denom = 1 + np.exp((V + top)/bottom)
	return offset + numer/denom

def double_exp_taus(V, numer, top_1, bottom_1, top_2, bottom_2, offset):
	denom = np.exp((V + top_1)/bottom_1) + np.exp((V + top_2)/bottom_2)
	return offset + numer/denom

def tau_h_Na(V):
	first_mult = 0.67/(1+np.exp((V + 62.9)/(-10)))
	second_mult = 1.5 + 1/(1 + np.exp((V + 34.9)/3.6))
	return first_mult*second_mult

def step_m_or_h(j, tau, infin):
	#j can be m or h, tau should be sef explanatory, infin is m or h infinity
	dj = time_step*(infin - j)/tau
	return dj

def step_Ca2(I_CaT, I_CaS, Ca2):
	#just realized they don't number their equations in this paper, weird
	dCa2 = time_step*(-14.96*(I_CaT + I_CaS) - Ca2 + 0.05)/200
	return dCa2

def current_contrib(A, V, E, g, m, p, h = 1):
	#hey some of the ions dont have h modeles in the table 1, gonna just assume constant 1 for those until i see otherwise
	I = A*g*(m**p)*h*(V-E)
	return I

def nernst(Ca2, Ca2_extra = 3):
	#that constant is currnty set for room temperature, maybe it should be something else like idk, body temperature?
	return 25.693*np.log(Ca2_extra/Ca2)

def m_h_stuff(all_params, V_membrane, Ca2, init_m_h = False):
	if init_m_h:
		#setting all the m values
		all_params[:, 0, 3] = m_or_h_infin(V_membrane, 25.5, -5.29)
		all_params[:, 1, 3] = m_or_h_infin(V_membrane, 27.1, -7.2)
		all_params[:, 2, 3] = m_or_h_infin(V_membrane, 33, -8.1)
		all_params[:, 3, 3] = m_or_h_infin(V_membrane, 27.2, -8.7)
		all_params[:, 4, 3] = m_or_h_infin(V_membrane, 28.3, -12.6, True, Ca2)
		all_params[:, 5, 3] = m_or_h_infin(V_membrane, 12.3, -11.8)
		all_params[:, 6, 3] = m_or_h_infin(V_membrane, 70, 6)
		all_params[:, 7, 3] = 1

		#and now all the h values
		all_params[:, 0, 4] = m_or_h_infin(V_membrane, 48.9, 5.18)
		all_params[:, 1, 4] = m_or_h_infin(V_membrane, 32.1, 5.5)
		all_params[:, 2, 4] = m_or_h_infin(V_membrane, 60, 6.2)
		all_params[:, 3, 4] = m_or_h_infin(V_membrane, 56.9, 4.9)
		all_params[:, 4, 4] = 1
		all_params[:, 5, 4] = 1
		all_params[:, 6, 4] = 1
		all_params[:, 7, 4] = 1

	#setting all the m_infinity values
	all_params[:, 0, 5] = m_or_h_infin(V_membrane, 25.5, -5.29)
	all_params[:, 1, 5] = m_or_h_infin(V_membrane, 27.1, -7.2)
	all_params[:, 2, 5] = m_or_h_infin(V_membrane, 33, -8.1)
	all_params[:, 3, 5] = m_or_h_infin(V_membrane, 27.2, -8.7)
	all_params[:, 4, 5] = m_or_h_infin(V_membrane, 28.3, -12.6, True, Ca2)
	all_params[:, 5, 5] = m_or_h_infin(V_membrane, 12.3, -11.8)
	all_params[:, 6, 5] = m_or_h_infin(V_membrane, 70, 6)
	all_params[:, 7, 5] = 1

	#and now all the h_infinity values
	all_params[:, 0, 6] = m_or_h_infin(V_membrane, 48.9, 5.18)
	all_params[:, 1, 6] = m_or_h_infin(V_membrane, 32.1, 5.5)
	all_params[:, 2, 6] = m_or_h_infin(V_membrane, 60, 6.2)
	all_params[:, 3, 6] = m_or_h_infin(V_membrane, 56.9, 4.9)
	all_params[:, 4, 6] = 1
	all_params[:, 5, 6] = 1
	all_params[:, 6, 6] = 1
	all_params[:, 7, 6] = 1

	#setting all the tau_m values
	all_params[:, 0, 7] = most_taus(V_membrane, -1.26, 120, -25, 1.32)
	all_params[:, 1, 7] = most_taus(V_membrane, -21.3, 68.1, -20.5, 21.7)
	all_params[:, 2, 7] = double_exp_taus(V_membrane, 7, 27, 10, 70, -13, 1.4)
	all_params[:, 3, 7] = most_taus(V_membrane, -10.4, 32.9, -15.2, 11.6)
	all_params[:, 4, 7] = most_taus(V_membrane, -75.1, 46, -22.7, 90.3)
	all_params[:, 5, 7] = most_taus(V_membrane, -6.4, 28.3, -19.2, 7.2)
	all_params[:, 6, 7] = most_taus(V_membrane, 1499, 42.2, -8.73, 272)
	all_params[:, 7, 7] = 1

	#and now all the ta_h values
	all_params[:, 0, 8] = tau_h_Na(V_membrane)
	all_params[:, 1, 8] = most_taus(V_membrane, -89.8, 55, -16.9, 105)
	all_params[:, 2, 8] = double_exp_taus(V_membrane, 150, 55, 9, 65, -16, 60)
	all_params[:, 3, 8] = most_taus(V_membrane, -29.2, 38.9, -26.5, 38.6)
	all_params[:, 4, 8] = 1
	all_params[:, 5, 8] = 1
	all_params[:, 6, 8] = 1
	all_params[:, 7, 8] = 1

	CaReverse = nernst(Ca2)
	all_params[:, 1, 1] = CaReverse[:] #hey if anything is wrong check here I'm highly skeptical
	all_params[:, 2, 1] = CaReverse[:] #hey if anything is wrong check here I'm highly skeptical

	return all_params


folder_path = "D:\\Neuro_Sci\\morph_ephys_trans_stuff\\fine_and_dandi_ephys\\000020\\sub-809076486\\"  # change this to your path
folder_path = "D:\\Neuro_Sci\\morph_ephys_trans_stuff\\fine_and_dandi_ephys\\000020\\sub-599387254\\"
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
all_times, all_currents, all_volts, orginal_lengths = lpcd.give_me_the_stuff(folder_path, files)

print('data loaded!')

end_stops = []
kind_of_resting_potential = []
for i in range(len(all_times)):
	stop_at = orginal_lengths[i]-1
	if all_currents[i, stop_at - 1] == 0:
		kind_of_resting_potential.append(all_volts[i, stop_at - 1])
	end_stops.append(stop_at)
	#plt.plot(all_times[i, :stop_at], all_currents[i, :stop_at])
	#plt.show()
end_stops = np.asarray(end_stops)

num_neurons = len(all_times)
A = 6.28E-4 #area in cm^2?
C = 6.28E-1 #capacitence in nF, something tells me they picked this to mostly cancel with area
time_step = 0.05

sim_length = np.max(all_times)
ephys_time_step = sim_length/(len(all_times[0]))
sim_steps = int(sim_length/time_step)
epochs = 9000

areas = np.random.normal(A, A/10)*np.ones(num_neurons)
caps = np.random.normal(C, C/10)*np.ones(num_neurons)

#conductance in mS/cm^2 ?
mult = 1000
Na_g = 100*mult*np.random.rand()
CaT_g = 12.5*mult*np.random.rand()
CaS_g = 10*mult*np.random.rand()
A_g = 50*mult*np.random.rand()
KCa_g = 25*mult*np.random.rand()
Kd_g = 125*mult*np.random.rand()
H_g = 0.5*mult*np.random.rand()
Leak_g = 0.5*mult*np.random.rand()

NaReverse = 50
KCaReverse = -80
KdReverse = -80
AReverse = -80
HReverse = -20

with open('current_best_leak_predict.pkl', 'rb') as f:
    model = pickle.load(f)
conduct_and_rest = np.asarray([Na_g, CaT_g, CaS_g, A_g, KCa_g, Kd_g, H_g, Leak_g, np.mean(kind_of_resting_potential)])
conduct_and_rest = conduct_and_rest.reshape(1, -1)

LeakReverse = model.predict(conduct_and_rest)


base_anneal_params = [Na_g, CaT_g, CaS_g, A_g, KCa_g, Kd_g, H_g, Leak_g, LeakReverse]
anneal_params = np.zeros(9)
for i in range(9):
	anneal_params[i] = base_anneal_params[i]
min_mse = 1000

for e in range(epochs):
	anneal_params[e%9] = np.random.normal(base_anneal_params[e%9], np.abs(base_anneal_params[e%9])/10)
	#volatile initialization
	V_membrane = anneal_params[8]*np.ones(num_neurons)
	Ca2 = 0.05*np.ones(num_neurons)
	CaReverse = nernst(Ca2)*np.ones(num_neurons)
	I_ext = 0.0*np.ones(num_neurons)
	
	#I genuinely don't know if the time constants are meant to be constant, like yeah they should be but they are functions of V so like?
	#I think they do change bc of top of page 3 but like 
	#max conductance, reverse potential, p, m, h, m_infinity, h_infinity, tau_m, tau_h
	
	all_params = np.zeros((num_neurons, 8, 9))
	for n in range(num_neurons):
		all_params[n, :, 0] = anneal_params[:8]
		all_params[n, :, 1] = [NaReverse, CaReverse[n], CaReverse[n], AReverse, KCaReverse, KdReverse, HReverse, anneal_params[8]] #hey if anything is wrong check here I'm highly skeptical
		all_params[n, :, 2] = [3, 3, 3, 3, 4, 4, 1, 1]
	all_params = m_h_stuff(all_params, V_membrane, Ca2, init_m_h = True)
	
	Vs = np.zeros((num_neurons, sim_steps))
	Cas = np.zeros((num_neurons, sim_steps))
	Cas_reverse = np.zeros((num_neurons, sim_steps))
	current_names = ['leak', 'Na', 'CaT', 'CaS', 'A', 'KCa', 'Kd', 'H']
	currents = np.zeros((num_neurons, 8, sim_steps))
	
	for s in range(sim_steps):
		current_time = s*time_step
		x = int(current_time/ephys_time_step)
		
		I_ext[:] = all_currents[:, x]
	
		#print(I_ext)
		
		all_params = m_h_stuff(all_params, V_membrane, Ca2)
		
		for index in range(len(all_params[0])):
			if index < 4:
				all_params[:, index, 4] += time_step*(all_params[:, index, 6] - all_params[:, index, 4])/all_params[:, index, 8]
			all_params[:, index, 3] += time_step*(all_params[:, index, 5] - all_params[:, index, 3])/all_params[:, index, 7]
			
		current_sum = np.zeros((num_neurons, 8))
		for index in range(len(all_params[0])):
			current_sum[:, index] = (current_contrib(areas, V_membrane, all_params[:, index, 1], all_params[:, index, 0], all_params[:, index, 3], all_params[:, index, 2], all_params[:, index, 4]))
		
		dV = time_step*(-1*np.sum(current_sum, axis = 1) + I_ext)/C
		V_membrane += dV
		
		Ca2 += step_Ca2(current_sum[:, 1], current_sum[:, 2], Ca2)
		
		Vs[:, s] = V_membrane[:]
		Cas[:, s] = Ca2[:]
		#currents[:, s] = current_sum
		#Cas_reverse[s] = nernst(Ca2)
		#print(V_membrane, Ca2, dV, all_params[:, 3:5])
	
	all_errors = np.zeros(num_neurons)
	for n in range(num_neurons):
		this_end = end_stops[n]
		
		time = all_times[n, :this_end]
		current = all_currents[n, :this_end]
		real_voltage = all_volts[n, :this_end]
		arr2_interp = interp.interp1d(np.arange(Vs[n].size),Vs[n])
		copy_voltage = arr2_interp(np.linspace(0,Vs[n].size-1, all_times[n].size))[:this_end]	
		
		'''
		fig, ax1 = plt.subplots(figsize=(10, 5))
		
		# Plot voltage
		ax1.set_xlabel('Time (ms)')
		ax1.set_ylabel('Voltage (mV)')
		ax1.plot(time, real_voltage, label="real voltage", color = [0, 0, 1])
		ax1.plot(time, copy_voltage, label="copy voltage", color = [0, 1, 0])
		ax1.tick_params(axis='y')
		
		# Create second y-axis for current
		ax2 = ax1.twinx()
		ax2.set_ylabel('Current (nA)')
		ax2.plot(time, current, label="injected current", color = [1, 0, 0])
		ax2.tick_params(axis='y')
		
		fig.tight_layout()
		plt.title("Voltage and Current Over Time")
		plt.show()
		'''
		
		mse = mean_squared_error(copy_voltage, real_voltage)
		all_errors[n] = mse
	mean_mse = np.mean(all_errors)
	print(e, mean_mse)
	if (mean_mse < min_mse):
		np.save('anneal_params', anneal_params)
		base_anneal_params = anneal_params
		min_mse = mean_mse
	else:
		T = 400*e/epochs
		if np.exp(-mean_mse/T) > np.random.rand():
			base_anneal_params = anneal_params
