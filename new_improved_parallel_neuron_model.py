import numpy as np
import matplotlib.pyplot as plt

#taken from this bad boi https://journals.physiology.org/doi/epdf/10.1152/jn.00641.2003
#they explicity set the membrane surface area which I should change to reflect morphology when I integrate with the Allen data
#the only thing they vary is the conductance, I geuss I could vary more but let's copy them first
#all voltages in mV, all times in ms
#currents should be in nA
#https://www.jneurosci.org/content/jneuro/18/7/2309.full.pdf
#aparently same model^ ?

#nope now it's this down here v
#https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1002107&type=printable
#I am making the exectutive decision to only do this as a point neuron model, no silly morphology

def slow_K_params(skp_loc_v_mem, skp_loc_Tadj):
	m_infin = 1/(1 + np.exp(-1*(skp_loc_v_mem + 11)/12))
	h_infin = 1/(1 + np.exp(1*(skp_loc_v_mem + 64)/11))
	
	if skp_loc_v_mem < -60:
		tau_m = (1.25 + 175.03*np.exp(0.026*(skp_loc_v_mem + 10)))/skp_loc_Tadj
	else:
		tau_m = (1.25 + 13*np.exp(0.026*(skp_loc_v_mem + 10)))/skp_loc_Tadj
		
	tau_h =  (360 + (1010 + 24*(skp_loc_v_mem + 65))*np.exp(-1*(((skp_loc_v_mem + 85)/48)**2)))
	
	return m_infin, h_infin, tau_m, tau_h

def musc_params(muscp_loc_v_mem, muscp_loc_Tadj):
	a_m = 3.3E-3*np.exp(0.1*(muscp_loc_v_mem + 35))
	b_m = 3.3E-3*np.exp(-0.1*(muscp_loc_v_mem + 35))
	
	m_infin = a_m/(a_m + b_m)
	
	tau_m = 1/(muscp_loc_Tadj*(a_m + b_m))
	
	return m_infin, 1, tau_m, 1

def non_spec_params(nsp_loc_v_mem):
	a_m = 6.43E-3*(nsp_loc_v_mem + 154.9)/(-1 + np.exp(1*(nsp_loc_v_mem + 154.9)/11.9))
	b_m = 1.93E-3*np.exp(nsp_loc_v_mem/33.1)
	
	m_infin = a_m/(a_m + b_m)
	
	tau_m = 1/(a_m + b_m)
	
	return m_infin, 1, tau_m, 1

def Na_pers_params(npp_loc_v_mem, npp_loc_Tadj):
	a_m = 0.182*(npp_loc_v_mem + 38)/(1 - np.exp(-1*(npp_loc_v_mem + 38)/6))
	b_m = -0.124*(npp_loc_v_mem + 38)/(1 - np.exp(1*(npp_loc_v_mem + 38)/6))
	
	a_h = -2.88E-6*(npp_loc_v_mem + 17)/(1 - np.exp(1*(npp_loc_v_mem + 17)/4.63))
	b_h = 6.94E-6*(npp_loc_v_mem + 64.4)/(1 - np.exp(-1*(npp_loc_v_mem + 64.4)/2.63))
	
	m_infin = 1/(1 + np.exp(-1*(npp_loc_v_mem + 52.6)/4.6))
	h_infin = 1/(1 + np.exp(1*(npp_loc_v_mem + 48.8)/10))
	
	tau_m = 6/(npp_loc_Tadj*(a_m + b_m))
	tau_h = 1/(npp_loc_Tadj*(a_h + b_h))
	
	return m_infin, h_infin, tau_m, tau_h

def Na_trans_params(ntp_loc_v_mem, ntp_loc_Tadj):
	a_m = 0.182*(ntp_loc_v_mem + 38)/(1 - np.exp(-1*(ntp_loc_v_mem + 38)/6))
	b_m = -0.124*(ntp_loc_v_mem + 38)/(1 - np.exp(1*(ntp_loc_v_mem + 38)/6))
	
	a_h = -0.015*(ntp_loc_v_mem + 66)/(1 - np.exp(1*(ntp_loc_v_mem + 66)/6))
	b_h = -0.015*(ntp_loc_v_mem + 66)/(-1 - np.exp(1*(ntp_loc_v_mem + 66)/6))
	
	m_infin = a_m/(a_m + b_m)
	h_infin = a_h/(a_h + b_h)
	
	tau_m = 1/(ntp_loc_Tadj*(a_m + b_m))
	tau_h = 1/(ntp_loc_Tadj*(a_h + b_h))
	
	return m_infin, h_infin, tau_m, tau_h

def randomize_coduct(rc_loc_param_limits, rc_loc_num_neurons):
	rc_loc_cond = np.zeros((rc_loc_num_neurons, 11))
	rc_loc_rev = np.random.uniform(rc_loc_param_limits[13, 0], rc_loc_param_limits[13, 1], rc_loc_num_neurons)
	for i in range(11):
		rc_loc_cond[:, i] = np.random.uniform(rc_loc_param_limits[i, 0], rc_loc_param_limits[i, 1], rc_loc_num_neurons)
	
	return rc_loc_cond, rc_loc_rev
	
def define_param_limits():
	
	#conductance units of pS/um^2
	#Tau in ms
	#gamma is unitless?
	
	#I can't find limits for non specific cation (I_H) or Muscarinic? (I_m) so I'm guessing from the last model
	#they belong in slots 2 and 3 respectively
	dpc_loc_param_limits = np.zeros((16, 2))
	
	dpc_loc_param_limits[0, 1] = 40000 #upper limit g_Na transient
	dpc_loc_param_limits[1, 1] = 100 #upper limit g_Na persistent
	
	dpc_loc_param_limits[2, 1] = 3 #upper limit g_H, infered from the older model
	dpc_loc_param_limits[3, 1] = 500 #upper limit g_H, guessed from older model, used g_A 
	
	dpc_loc_param_limits[4, 1] = 1000 #upper limit g_K transient
	dpc_loc_param_limits[5, 1] = 10000 #upper limit g_K persistent
	dpc_loc_param_limits[6, 1] = 20000 #upper limit g_Kv3.1
	dpc_loc_param_limits[7, 1] = 10 #upper limit g_Ca_HVA
	dpc_loc_param_limits[8, 1] = 100 #upper limit g_Ca_LVA
	dpc_loc_param_limits[9, 1] = 100 #upper limit g_SK
	dpc_loc_param_limits[10, :] = [0.2, 0.5] #limits g_leak
	
	dpc_loc_param_limits[11, :] = [0.0005, 0.05] #limits for gamma, some thing in the Calcium change
	dpc_loc_param_limits[12, :] = [20, 1000] #limits for Tau decay
	
	dpc_loc_param_limits[13, :] = [-110, -60] #limits for resting potential, even if this is not a cellular propert (its set by extracellular ion concentration or sm) it still needs to be fit
	dpc_loc_param_limits[14, :] = [6E3, 6E5] #area in um^2
	dpc_loc_param_limits[15, :] = [0.5, 2] #capacitence in uF
	
	return dpc_loc_param_limits	

def step_Ca2(I_CaT, I_CaS, Ca2):
	#just realized they don't number their equations in this paper, weird
	dCa2 = (-14.96*(I_CaT + I_CaS) - Ca2 + 0.05)/200
	return dCa2

def current_contrib(A, V, E, g, m, p, h = 1):
	#hey some of the ions dont have h modeles in the table 1, gonna just assume constant 1 for those until i see otherwise
	I = A*g*(m**p)*h*(V-E)
	return I

def nernst(nernst_local_Ca2, Ca2_extra = 3):
	#that constant is currnty set for room temperature, maybe it should be something else like idk, body temperature?
	return 25.693*np.log(Ca2_extra/nernst_local_Ca2)

def m_h_stuff(mh_local_all_params, mh_local_V_mem, mh_local_Ca2, init_m_h = False):
	if init_m_h:
		#setting all the m values
		mh_local_all_params[:, 0, 3] = m_or_h_infin(mh_local_V_mem, 25.5, -5.29)
		mh_local_all_params[:, 1, 3] = m_or_h_infin(mh_local_V_mem, 27.1, -7.2)
		mh_local_all_params[:, 2, 3] = m_or_h_infin(mh_local_V_mem, 33, -8.1)
		mh_local_all_params[:, 3, 3] = m_or_h_infin(mh_local_V_mem, 27.2, -8.7)
		mh_local_all_params[:, 4, 3] = m_or_h_infin(mh_local_V_mem, 28.3, -12.6, True, mh_local_Ca2)
		mh_local_all_params[:, 5, 3] = m_or_h_infin(mh_local_V_mem, 12.3, -11.8)
		mh_local_all_params[:, 6, 3] = m_or_h_infin(mh_local_V_mem, 70, 6)
		mh_local_all_params[:, 7, 3] = 1

		#and now all the h values
		mh_local_all_params[:, 0, 4] = m_or_h_infin(mh_local_V_mem, 48.9, 5.18)
		mh_local_all_params[:, 1, 4] = m_or_h_infin(mh_local_V_mem, 32.1, 5.5)
		mh_local_all_params[:, 2, 4] = m_or_h_infin(mh_local_V_mem, 60, 6.2)
		mh_local_all_params[:, 3, 4] = m_or_h_infin(mh_local_V_mem, 56.9, 4.9)
		mh_local_all_params[:, 4, 4] = 1
		mh_local_all_params[:, 5, 4] = 1
		mh_local_all_params[:, 6, 4] = 1
		mh_local_all_params[:, 7, 4] = 1

	#setting all the m_infinity values
	mh_local_all_params[:, 0, 5] = m_or_h_infin(mh_local_V_mem, 25.5, -5.29)
	mh_local_all_params[:, 1, 5] = m_or_h_infin(mh_local_V_mem, 27.1, -7.2)
	mh_local_all_params[:, 2, 5] = m_or_h_infin(mh_local_V_mem, 33, -8.1)
	mh_local_all_params[:, 3, 5] = m_or_h_infin(mh_local_V_mem, 27.2, -8.7)
	mh_local_all_params[:, 4, 5] = m_or_h_infin(mh_local_V_mem, 28.3, -12.6, True, mh_local_Ca2)
	mh_local_all_params[:, 5, 5] = m_or_h_infin(mh_local_V_mem, 12.3, -11.8)
	mh_local_all_params[:, 6, 5] = m_or_h_infin(mh_local_V_mem, 70, 6)
	mh_local_all_params[:, 7, 5] = 1

	#and now all the h_infinity values
	mh_local_all_params[:, 0, 6] = m_or_h_infin(mh_local_V_mem, 48.9, 5.18)
	mh_local_all_params[:, 1, 6] = m_or_h_infin(mh_local_V_mem, 32.1, 5.5)
	mh_local_all_params[:, 2, 6] = m_or_h_infin(mh_local_V_mem, 60, 6.2)
	mh_local_all_params[:, 3, 6] = m_or_h_infin(mh_local_V_mem, 56.9, 4.9)
	mh_local_all_params[:, 4, 6] = 1
	mh_local_all_params[:, 5, 6] = 1
	mh_local_all_params[:, 6, 6] = 1
	mh_local_all_params[:, 7, 6] = 1

	#setting all the tau_m values
	mh_local_all_params[:, 0, 7] = most_taus(mh_local_V_mem, -1.26, 120, -25, 1.32)
	mh_local_all_params[:, 1, 7] = most_taus(mh_local_V_mem, -21.3, 68.1, -20.5, 21.7)
	mh_local_all_params[:, 2, 7] = double_exp_taus(mh_local_V_mem, 7, 27, 10, 70, -13, 1.4)
	mh_local_all_params[:, 3, 7] = most_taus(mh_local_V_mem, -10.4, 32.9, -15.2, 11.6)
	mh_local_all_params[:, 4, 7] = most_taus(mh_local_V_mem, -75.1, 46, -22.7, 90.3)
	mh_local_all_params[:, 5, 7] = most_taus(mh_local_V_mem, -6.4, 28.3, -19.2, 7.2)
	mh_local_all_params[:, 6, 7] = most_taus(mh_local_V_mem, 1499, 42.2, -8.73, 272)
	mh_local_all_params[:, 7, 7] = 1

	#and now all the ta_h values
	mh_local_all_params[:, 0, 8] = tau_h_Na(mh_local_V_mem)
	mh_local_all_params[:, 1, 8] = most_taus(mh_local_V_mem, -89.8, 55, -16.9, 105)
	mh_local_all_params[:, 2, 8] = double_exp_taus(mh_local_V_mem, 150, 55, 9, 65, -16, 60)
	mh_local_all_params[:, 3, 8] = most_taus(mh_local_V_mem, -29.2, 38.9, -26.5, 38.6)
	mh_local_all_params[:, 4, 8] = 1
	mh_local_all_params[:, 5, 8] = 1
	mh_local_all_params[:, 6, 8] = 1
	mh_local_all_params[:, 7, 8] = 1

	CaReverse = nernst(mh_local_Ca2)
	mh_local_all_params[:, 1, 1] = CaReverse[:] #hey if anything is wrong check here I'm highly skeptical
	mh_local_all_params[:, 2, 1] = CaReverse[:] #hey if anything is wrong check here I'm highly skeptical

	return mh_local_all_params

def current_primitive(curr_prim_local_sim_steps, input_type = None, start_current = 0, peak_current = 0, start_time = 0, end_time = 1):
	current = np.zeros(curr_prim_local_sim_steps)
	if input_type == 'ramp':
		start_time_idx = int(start_time*curr_prim_local_sim_steps)
		end_time_idx = int(end_time*curr_prim_local_sim_steps)
		current[start_time_idx:end_time_idx] = np.linspace(start_current, peak_current, end_time_idx-start_time_idx)
	
	if input_type == 'square':
		start_time_idx = int(start_time*curr_prim_local_sim_steps)
		end_time_idx = int(end_time*curr_prim_local_sim_steps)
		current[start_time_idx:end_time_idx] = np.ones(end_time_idx-start_time_idx)*peak_current
		
	return current

def the_old_simulator9000(sim_local_all_params, sim_local_V_mem, sim_local_Ca2, sim_local_areas, sim_local_caps, sim_length = 3000, time_step = 0.05, I_ext = None):
	sim_local_all_params = m_h_stuff(sim_local_all_params, sim_local_V_mem, sim_local_Ca2, True)
	
	sim_steps = int(sim_length/time_step)
	if I_ext is None:
		I_ext = np.zeros(sim_steps)
	
	local_num_neurons = len(sim_local_all_params)
	
	Vs = np.zeros((local_num_neurons, sim_steps))
	Cas = np.zeros((local_num_neurons, sim_steps))
	
	for s in range(sim_steps):
		
		sim_local_all_params = m_h_stuff(sim_local_all_params, sim_local_V_mem, sim_local_Ca2)
			
		for index in range(len(sim_local_all_params[0])):
			if index < 4:
				sim_local_all_params[:, index, 4] += time_step*(sim_local_all_params[:, index, 6] - sim_local_all_params[:, index, 4])/sim_local_all_params[:, index, 8]
			sim_local_all_params[:, index, 3] += time_step*(sim_local_all_params[:, index, 5] - sim_local_all_params[:, index, 3])/sim_local_all_params[:, index, 7]
			
		current_sum = np.zeros((local_num_neurons, 8))
		for index in range(len(sim_local_all_params[0])):
			current_sum[:, index] = (current_contrib(sim_local_areas, sim_local_V_mem, sim_local_all_params[:, index, 1], sim_local_all_params[:, index, 0], sim_local_all_params[:, index, 3], sim_local_all_params[:, index, 2], sim_local_all_params[:, index, 4]))
		
		dV = time_step*(-1*np.sum(current_sum, axis = 1) + I_ext[s])/(sim_local_caps)
		sim_local_V_mem += dV
		
		sim_local_Ca2 += step_Ca2(current_sum[:, 1], current_sum[:, 2], sim_local_Ca2)*time_step
		
		Vs[:, s] = sim_local_V_mem[:]
	
	return Vs, sim_local_all_params, sim_local_V_mem, sim_local_Ca2

def initialize_neurons(init_loc_num_neurons = 500):
	Tadj = 2.95 #epirically taken from their paper, i don't care about temperature	
	
	init_loc_param_limits = define_param_limits()
		
	init_local_areas = np.random.uniform(init_loc_param_limits[14, 0], init_loc_param_limits[14, 1], init_loc_num_neurons)
	init_local_caps = np.random.uniform(init_loc_param_limits[15, 0], init_loc_param_limits[15, 1], init_loc_num_neurons)
	
	#reverse potentials in mV
	NaReverse = 50
	KReverse = -85
	HReverse = -45
	LeakReverse = -90
	
	#volatile initialization
	init_local_V_mem = LeakReverse*np.ones(init_loc_num_neurons)
	init_local_Ca2 = 0.05*np.ones(init_loc_num_neurons)
	init_local_CaReverse = nernst(init_local_Ca2)*np.ones(init_loc_num_neurons)
	
	#max conductance, reverse potential, p, m, h, m_infinity, h_infinity, tau_m, tau_h
	init_local_all_params = np.zeros((init_loc_num_neurons, 11, 9))
	init_local_all_params[:, :, 0], LeakReverse = randomize_coduct(init_loc_param_limits, init_loc_num_neurons)
	
	for n in range(init_loc_num_neurons):
		init_local_all_params[n, :, 1] = [NaReverse, NaReverse, HReverse, KReverse, KReverse, KReverse, KReverse, init_local_CaReverse[n], init_local_CaReverse[n], KReverse, LeakReverse[n]] #hey if anything is wrong check here I'm highly skeptical
		init_local_all_params[n, :, 2] = [3, 3, 1, 1, 2, 4, 1, 2, 2, 1, 1]

	init_local_all_params = m_h_stuff(init_local_all_params, init_local_V_mem, init_local_Ca2, init_m_h = True)
	
	return init_local_all_params, init_local_V_mem, init_local_Ca2, init_local_areas, init_local_caps


glob_num_neurons = 10
glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps = initialize_neurons(glob_num_neurons)

'''
glob_curr_prim = current_primitive(60000, 'square', 0, 0.5, 0.5, 0.75)
glob_Vs, glob_all_params, glob_V_mem, glob_Ca2 = the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps)
glob_Vs, glob_all_params, glob_V_mem, glob_Ca2 = the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_areas, glob_caps, I_ext = glob_curr_prim)

plt.plot(glob_Vs[0])
plt.show()
'''