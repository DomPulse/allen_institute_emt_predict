import numpy as np
import matplotlib.pyplot as plt

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

def step_Ca2(I_CaT, I_CaS, Ca2):
	#just realized they don't number their equations in this paper, weird
	dCa2 = (-14.96*(I_CaT + I_CaS) - Ca2 + 0.05)/200
	return dCa2

def current_contrib(A, V, E, g, m, p, h = 1):
	#hey some of the ions dont have h modeles in the table 1, gonna just assume constant 1 for those until i see otherwise
	I = A*g*(m**p)*h*(V-E)
	return I

def nernst(nernst_loc_Ca2, Ca2_extra = 3):
	#that constant is currnty set for room temperature, maybe it should be something else like idk, body temperature?
	return 25.693*np.log(Ca2_extra/nernst_loc_Ca2)

def m_h_stuff(mh_loc_all_params, mh_loc_V_mem, mh_loc_Ca2, init_m_h = False):
	if init_m_h:
		#setting all the m values
		mh_loc_all_params[:, 0, 3] = m_or_h_infin(mh_loc_V_mem, 25.5, -5.29)
		mh_loc_all_params[:, 1, 3] = m_or_h_infin(mh_loc_V_mem, 27.1, -7.2)
		mh_loc_all_params[:, 2, 3] = m_or_h_infin(mh_loc_V_mem, 33, -8.1)
		mh_loc_all_params[:, 3, 3] = m_or_h_infin(mh_loc_V_mem, 27.2, -8.7)
		mh_loc_all_params[:, 4, 3] = m_or_h_infin(mh_loc_V_mem, 28.3, -12.6, True, mh_loc_Ca2)
		mh_loc_all_params[:, 5, 3] = m_or_h_infin(mh_loc_V_mem, 12.3, -11.8)
		mh_loc_all_params[:, 6, 3] = m_or_h_infin(mh_loc_V_mem, 70, 6)
		mh_loc_all_params[:, 7, 3] = 1

		#and now all the h values
		mh_loc_all_params[:, 0, 4] = m_or_h_infin(mh_loc_V_mem, 48.9, 5.18)
		mh_loc_all_params[:, 1, 4] = m_or_h_infin(mh_loc_V_mem, 32.1, 5.5)
		mh_loc_all_params[:, 2, 4] = m_or_h_infin(mh_loc_V_mem, 60, 6.2)
		mh_loc_all_params[:, 3, 4] = m_or_h_infin(mh_loc_V_mem, 56.9, 4.9)
		mh_loc_all_params[:, 4, 4] = 1
		mh_loc_all_params[:, 5, 4] = 1
		mh_loc_all_params[:, 6, 4] = 1
		mh_loc_all_params[:, 7, 4] = 1

	#setting all the m_infinity values
	mh_loc_all_params[:, 0, 5] = m_or_h_infin(mh_loc_V_mem, 25.5, -5.29)
	mh_loc_all_params[:, 1, 5] = m_or_h_infin(mh_loc_V_mem, 27.1, -7.2)
	mh_loc_all_params[:, 2, 5] = m_or_h_infin(mh_loc_V_mem, 33, -8.1)
	mh_loc_all_params[:, 3, 5] = m_or_h_infin(mh_loc_V_mem, 27.2, -8.7)
	mh_loc_all_params[:, 4, 5] = m_or_h_infin(mh_loc_V_mem, 28.3, -12.6, True, mh_loc_Ca2)
	mh_loc_all_params[:, 5, 5] = m_or_h_infin(mh_loc_V_mem, 12.3, -11.8)
	mh_loc_all_params[:, 6, 5] = m_or_h_infin(mh_loc_V_mem, 70, 6)
	mh_loc_all_params[:, 7, 5] = 1

	#and now all the h_infinity values
	mh_loc_all_params[:, 0, 6] = m_or_h_infin(mh_loc_V_mem, 48.9, 5.18)
	mh_loc_all_params[:, 1, 6] = m_or_h_infin(mh_loc_V_mem, 32.1, 5.5)
	mh_loc_all_params[:, 2, 6] = m_or_h_infin(mh_loc_V_mem, 60, 6.2)
	mh_loc_all_params[:, 3, 6] = m_or_h_infin(mh_loc_V_mem, 56.9, 4.9)
	mh_loc_all_params[:, 4, 6] = 1
	mh_loc_all_params[:, 5, 6] = 1
	mh_loc_all_params[:, 6, 6] = 1
	mh_loc_all_params[:, 7, 6] = 1

	#setting all the tau_m values
	mh_loc_all_params[:, 0, 7] = most_taus(mh_loc_V_mem, -1.26, 120, -25, 1.32)
	mh_loc_all_params[:, 1, 7] = most_taus(mh_loc_V_mem, -21.3, 68.1, -20.5, 21.7)
	mh_loc_all_params[:, 2, 7] = double_exp_taus(mh_loc_V_mem, 7, 27, 10, 70, -13, 1.4)
	mh_loc_all_params[:, 3, 7] = most_taus(mh_loc_V_mem, -10.4, 32.9, -15.2, 11.6)
	mh_loc_all_params[:, 4, 7] = most_taus(mh_loc_V_mem, -75.1, 46, -22.7, 90.3)
	mh_loc_all_params[:, 5, 7] = most_taus(mh_loc_V_mem, -6.4, 28.3, -19.2, 7.2)
	mh_loc_all_params[:, 6, 7] = most_taus(mh_loc_V_mem, 1499, 42.2, -8.73, 272)
	mh_loc_all_params[:, 7, 7] = 1

	#and now all the ta_h values
	mh_loc_all_params[:, 0, 8] = tau_h_Na(mh_loc_V_mem)
	mh_loc_all_params[:, 1, 8] = most_taus(mh_loc_V_mem, -89.8, 55, -16.9, 105)
	mh_loc_all_params[:, 2, 8] = double_exp_taus(mh_loc_V_mem, 150, 55, 9, 65, -16, 60)
	mh_loc_all_params[:, 3, 8] = most_taus(mh_loc_V_mem, -29.2, 38.9, -26.5, 38.6)
	mh_loc_all_params[:, 4, 8] = 1
	mh_loc_all_params[:, 5, 8] = 1
	mh_loc_all_params[:, 6, 8] = 1
	mh_loc_all_params[:, 7, 8] = 1

	CaReverse = nernst(mh_loc_Ca2)
	mh_loc_all_params[:, 1, 1] = CaReverse[:] #hey if anything is wrong check here I'm highly skeptical
	mh_loc_all_params[:, 2, 1] = CaReverse[:] #hey if anything is wrong check here I'm highly skeptical

	return mh_loc_all_params


def current_primitive(curr_prim_loc_sim_steps, input_type = None, start_current = 0, peak_current = 0, start_time = 0, end_time = 1):
	current = np.zeros(curr_prim_loc_sim_steps)
	if input_type == 'ramp':
		start_time_idx = int(start_time*curr_prim_loc_sim_steps)
		end_time_idx = int(end_time*curr_prim_loc_sim_steps)
		current[start_time_idx:end_time_idx] = np.linspace(start_current, peak_current, end_time_idx-start_time_idx)
	
	if input_type == 'square':
		start_time_idx = int(start_time*curr_prim_loc_sim_steps)
		end_time_idx = int(end_time*curr_prim_loc_sim_steps)
		current[start_time_idx:end_time_idx] = np.ones(end_time_idx-start_time_idx)*peak_current
		
	return current

def the_old_simulator9000(sim_loc_all_params, sim_loc_V_mem, sim_loc_Ca2, sim_loc_areas, sim_loc_caps, sim_length = 3000, time_step = 0.05, I_ext = None):
	sim_loc_all_params = m_h_stuff(sim_loc_all_params, sim_loc_V_mem, sim_loc_Ca2, True)
	
	sim_steps = int(sim_length/time_step)
	if I_ext is None:
		I_ext = np.zeros(sim_steps)
	
	local_num_neurons = len(sim_loc_all_params)
	
	Vs = np.zeros((local_num_neurons, sim_steps))
	Cas = np.zeros((local_num_neurons, sim_steps))
	
	for s in range(sim_steps):
		
		sim_loc_all_params = m_h_stuff(sim_loc_all_params, sim_loc_V_mem, sim_loc_Ca2)
			
		for index in range(len(sim_loc_all_params[0])):
			if index < 4:
				sim_loc_all_params[:, index, 4] += time_step*(sim_loc_all_params[:, index, 6] - sim_loc_all_params[:, index, 4])/sim_loc_all_params[:, index, 8]
			sim_loc_all_params[:, index, 3] += time_step*(sim_loc_all_params[:, index, 5] - sim_loc_all_params[:, index, 3])/sim_loc_all_params[:, index, 7]
			
		current_sum = np.zeros((local_num_neurons, 8))
		for index in range(len(sim_loc_all_params[0])):
			current_sum[:, index] = (current_contrib(sim_loc_areas, sim_loc_V_mem, sim_loc_all_params[:, index, 1], sim_loc_all_params[:, index, 0], sim_loc_all_params[:, index, 3], sim_loc_all_params[:, index, 2], sim_loc_all_params[:, index, 4]))
		
		dV = time_step*(-1*np.sum(current_sum, axis = 1) + I_ext[s])/(sim_loc_caps)
		sim_loc_V_mem += dV
		sim_loc_Ca2 += step_Ca2(current_sum[:, 1], current_sum[:, 2], sim_loc_Ca2)*time_step
		
		Vs[:, s] = sim_loc_V_mem[:]
		#Vs[:, s] = sim_loc_Ca2[:]
	
	return Vs, sim_loc_all_params, sim_loc_V_mem, sim_loc_Ca2

def initialize_neurons(num_neurons = 500, area = 6.28E-4, capacitence = 6.28E-1, min_area = 6.28E-4, min_capacitence = 6.28E-2):
	init_loc_areas = area*np.ones(num_neurons)
	init_loc_caps = capacitence*np.ones(num_neurons)
	
	init_loc_areas = np.random.normal(area, area/3, num_neurons)
	init_loc_caps = np.random.normal(capacitence, capacitence/3, num_neurons)
	
	#max conductance in uS/cm^2 ?
	mult = 1000
	Na_g = 100*mult
	CaT_g = 12.5*mult
	CaS_g = 10*mult
	A_g = 50*mult
	KCa_g = 25*mult
	Kd_g = 125*mult
	H_g = 0.03*mult
	Leak_g = 0.02*mult

	#reverse potentials in mV
	NaReverse = 50
	KCaReverse = -80
	KdReverse = -80
	AReverse = -80
	HReverse = -20
	LeakReverse = -90*np.ones(num_neurons)
	print(LeakReverse)
	
	#volatile initialization
	init_loc_V_mem = -70*np.ones(num_neurons)
	init_loc_Ca2 = 0.05*np.ones(num_neurons)
	init_loc_CaReverse = nernst(init_loc_Ca2)*np.ones(num_neurons)
	
	#max conductance, reverse potential, p, m, h, m_infinity, h_infinity, tau_m, tau_h
	init_loc_all_params = np.zeros((num_neurons, 8, 9))
	for n in range(num_neurons):
		init_loc_all_params[n, :, 0] = [Na_g, CaT_g, CaS_g, A_g, KCa_g, Kd_g, H_g, Leak_g]*np.random.rand(8)
		init_loc_all_params[n, :, 1] = [NaReverse, init_loc_CaReverse[n], init_loc_CaReverse[n], AReverse, KCaReverse, KdReverse, HReverse, LeakReverse[n]] #hey if anything is wrong check here I'm highly skeptical
		init_loc_all_params[n, :, 2] = [3, 3, 3, 3, 4, 4, 1, 1]
		#all_params[n] = mutate(all_params[n])
	init_loc_all_params = m_h_stuff(init_loc_all_params, init_loc_V_mem, init_loc_Ca2, init_m_h = True)
	
	return init_loc_all_params, init_loc_V_mem, init_loc_Ca2, init_loc_areas, init_loc_caps

'''
glob_num_neurons = 10
glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps = initialize_neurons(num_neurons = glob_num_neurons)
	
glob_curr_prim = current_primitive(60000, 'square', 0, 0.15, 0.33, 0.66)
glob_Vs, glob_all_params, glob_V_mem, glob_Ca2 = the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps)
glob_Vs, glob_all_params, glob_V_mem, glob_Ca2 = the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps, I_ext = glob_curr_prim)

for i in range(glob_num_neurons):
	plt.plot(glob_Vs[i])
	plt.show()
	print(glob_Vs[i, 0], glob_Vs[i, -1])
'''