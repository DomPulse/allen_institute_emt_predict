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

def SK_Ca_params(skca_loc_Ca2):
	m_infin = 1/(1 + (4.3E-4/skca_loc_Ca2)**(4.8))	
	
	return m_infin, np.ones(len(m_infin)), np.ones(len(m_infin)), np.ones(len(m_infin))

def Ca_LVA_params(clva_loc_V_mem, clva_loc_Tadj):
	m_infin = 1/(1 + np.exp(-1*(clva_loc_V_mem + 40)/6))
	h_infin = 1/(1 + np.exp(1*(clva_loc_V_mem + 90)/6.4))
	
	tau_m = 5 + 20/(clva_loc_Tadj*(1 + np.exp((clva_loc_V_mem + 35)/5)))
	tau_h = 20 + 50/(clva_loc_Tadj*(1 + np.exp((clva_loc_V_mem + 50)/7)))
	
	return m_infin, h_infin, tau_m, tau_h

def Ca_HVA_params(chva_loc_V_mem):
	a_m = -0.055*(chva_loc_V_mem + 27)/(-1 + np.exp(-1*(chva_loc_V_mem + 27)/3.8))
	b_m = 0.94*np.exp((chva_loc_V_mem + 75)/17)
	
	a_h = 4.57E-4*np.exp(-1*(chva_loc_V_mem + 13)/50)
	b_h = 6.5E-3/(1 + np.exp(-1*(chva_loc_V_mem + 15)/28))
	
	m_infin = a_m/(a_m + b_m)
	h_infin = a_h/(a_h + b_h)
	
	tau_m = 1/(a_m + b_m)
	tau_h = 1/(a_h + b_h)
	
	return m_infin, h_infin, tau_m, tau_h

def Kv31_params(kv31_loc_mem_v, kv31_loc_Tadj):
	m_infin = 1/(1+np.exp(-1*(kv31_loc_mem_v - 18.7)/9.7))
	
	tau_m = 4/(kv31_loc_Tadj*(1+np.exp(-1*(kv31_loc_mem_v + 56.56)/44.14)))
	
	return m_infin, np.ones(len(m_infin)), tau_m, np.ones(len(m_infin))

def fast_K_params(fkp_loc_V_mem, fkp_loc_Tadj):
	m_infin = 1/(1 + np.exp(-1*(fkp_loc_V_mem + 10)/19))
	h_infin = 1/(1 + np.exp(1*(fkp_loc_V_mem + 76)/10))
	
	tau_m = (0.34 + 0.92*np.exp(-1*(((fkp_loc_V_mem + 81)/59)**2)))/fkp_loc_Tadj
	tau_h = (8 + 49*np.exp(-1*(((fkp_loc_V_mem + 83)/23)**2)))/fkp_loc_Tadj
	
	return m_infin, h_infin, tau_m, tau_h

def slow_K_params(skp_loc_V_mem, skp_loc_Tadj):
	m_infin = 1/(1 + np.exp(-1*(skp_loc_V_mem + 11)/12))
	h_infin = 1/(1 + np.exp(1*(skp_loc_V_mem + 64)/11))
	
	mask = (skp_loc_V_mem < -60)
	
	less_60 = mask*((1.25 + 175.03*np.exp(0.026*(skp_loc_V_mem + 10)))/skp_loc_Tadj)
	over60 = (1 - mask)*((1.25 + 13*np.exp(-0.026*(skp_loc_V_mem + 10)))/skp_loc_Tadj)
	
	tau_m = less_60 + over60
		
	tau_h =  (360 + (1010 + 24*(skp_loc_V_mem + 65))*np.exp(-1*(((skp_loc_V_mem + 85)/48)**2)))
	
	return m_infin, h_infin, tau_m, tau_h

def musc_params(muscp_loc_V_mem, muscp_loc_Tadj):
	a_m = 3.3E-3*np.exp(0.1*(muscp_loc_V_mem + 35))
	b_m = 3.3E-3*np.exp(-0.1*(muscp_loc_V_mem + 35))
	
	m_infin = a_m/(a_m + b_m)
	
	tau_m = 1/(muscp_loc_Tadj*(a_m + b_m))
	
	return m_infin, np.ones(len(m_infin)), tau_m, np.ones(len(m_infin))

def non_spec_params(nsp_loc_V_mem):
	a_m = 6.43E-3*(nsp_loc_V_mem + 154.9)/(-1 + np.exp(1*(nsp_loc_V_mem + 154.9)/11.9))
	b_m = 1.93E-3*np.exp(nsp_loc_V_mem/33.1)
	
	m_infin = a_m/(a_m + b_m)
	
	tau_m = 1/(a_m + b_m)
	
	return m_infin, np.ones(len(m_infin)), tau_m, np.ones(len(m_infin))

def Na_pers_params(npp_loc_V_mem, npp_loc_Tadj):
	a_m = 0.182*(npp_loc_V_mem + 38)/(1 - np.exp(-1*(npp_loc_V_mem + 38)/6))
	b_m = -0.124*(npp_loc_V_mem + 38)/(1 - np.exp(1*(npp_loc_V_mem + 38)/6))
	
	a_h = -2.88E-6*(npp_loc_V_mem + 17)/(1 - np.exp(1*(npp_loc_V_mem + 17)/4.63))
	b_h = 6.94E-6*(npp_loc_V_mem + 64.4)/(1 - np.exp(-1*(npp_loc_V_mem + 64.4)/2.63))
	
	m_infin = 1/(1 + np.exp(-1*(npp_loc_V_mem + 52.6)/4.6))
	h_infin = 1/(1 + np.exp(1*(npp_loc_V_mem + 48.8)/10))
	
	tau_m = 6/(npp_loc_Tadj*(a_m + b_m))
	tau_h = 1/(npp_loc_Tadj*(a_h + b_h))
	
	return m_infin, h_infin, tau_m, tau_h

def Na_trans_params(ntp_loc_V_mem, ntp_loc_Tadj):
	a_m = 0.182*(ntp_loc_V_mem + 38)/(1 - np.exp(-1*(ntp_loc_V_mem + 38)/6))
	b_m = -0.124*(ntp_loc_V_mem + 38)/(1 - np.exp(1*(ntp_loc_V_mem + 38)/6))
	
	a_h = -0.015*(ntp_loc_V_mem + 66)/(1 - np.exp(1*(ntp_loc_V_mem + 66)/6))
	b_h = 0.015*(ntp_loc_V_mem + 66)/(1 - np.exp(-1*(ntp_loc_V_mem + 66)/6))
	
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
	
	mult = 1E-1
	dpc_loc_param_limits[0, 1] = 40000*mult #upper limit g_Na transient
	dpc_loc_param_limits[1, 1] = 100*mult #upper limit g_Na persistent
	
	dpc_loc_param_limits[2, 1] = 3*mult #upper limit g_H, infered from the older model
	dpc_loc_param_limits[3, 1] = 500*mult #guessed from older model, used g_A 
	
	dpc_loc_param_limits[4, 1] = 1000*mult #upper limit g_K transient
	dpc_loc_param_limits[5, 1] = 10000*mult #upper limit g_K persistent
	dpc_loc_param_limits[6, 1] = 20000*mult #upper limit g_Kv3.1
	dpc_loc_param_limits[7, 1] = 10*mult #upper limit g_Ca_HVA
	dpc_loc_param_limits[8, 1] = 100*mult #upper limit g_Ca_LVA
	dpc_loc_param_limits[9, 1] = 100*mult #upper limit g_SK
	dpc_loc_param_limits[10, :] = [0.2*mult, 0.5*mult] #limits g_leak
	
	dpc_loc_param_limits[11, :] = [0.0005, 0.05] #limits for gamma, some thing in the Calcium change
	dpc_loc_param_limits[12, :] = [20, 1000] #limits for Tau decay
	
	dpc_loc_param_limits[13, :] = [-90, -70] #limits for resting potential, even if this is not a cellular propert (its set by extracellular ion concentration or sm) it still needs to be fit
	
	dpc_loc_param_limits[15, :] = [6E-1, 6.5E-1] #capacitence in uF
	dpc_loc_param_limits[14, :] = [6E-3, 6.5E-3] #area in um^2?
	
	dpc_loc_param_limits[15, :] = [0.5, 2] #capacitence in uF
	#dpc_loc_param_limits[14, :] = [0.9, 1] #area in um^2?
	
	#dpc_loc_param_limits[15, :] = [6.28E-2, 1] #capacitence in uF?

	
	return dpc_loc_param_limits	

def step_Ca2(sCa2_loc_I_CaHVA, sCa2_loc_I_CaLVA, sCa2_loc_Ca2, sCa2_loc_gamma, sCa2_loc_Tau_decay):
	#if things run ammock, check F
	F = 9.65E-4 #faradays constant (what units??? do dimensional analysis)
	d = 1E6 #"depth of sub-membrane shell for concentration calculations in um"
	#i assum that d is physically unrealistic and F is just in wrong units or something, I don't care
	#debugged via direct comparision to the old Ca2 function
	dCa2 = -1*(sCa2_loc_I_CaHVA + sCa2_loc_I_CaLVA)/(2*F*d*sCa2_loc_gamma) + (1E-4 - sCa2_loc_Ca2)/sCa2_loc_Tau_decay
	return dCa2

def old_step_Ca2(sCa2_loc_I_CaHVA, sCa2_loc_I_CaLVA, sCa2_loc_Ca2, sCa2_loc_gamma, sCa2_loc_Tau_decay):
	#just realized they don't number their equations in this paper, weird
	dCa2 = (-14.96*(sCa2_loc_I_CaHVA + sCa2_loc_I_CaLVA) - sCa2_loc_Ca2 + 0.05)/200
	return dCa2

def nernst(nernst_loc_Ca2, Ca2_extra = 3):
	#that constant is currnty set for room temperature, maybe it should be something else like idk, body temperature?
	return 25.693*np.log(Ca2_extra/nernst_loc_Ca2)

def m_h_stuff(mh_loc_all_params, mh_loc_V_mem, mh_loc_Ca2, mh_loc_Tadj, init_m_h = False):
	mh_loc_num_neurons = len(mh_loc_all_params)
	
	#setting all the m/h_infinity and tau_m/h values
	mh_loc_all_params[:, 0, 5:9] = np.transpose(Na_trans_params(mh_loc_V_mem, mh_loc_Tadj))
	mh_loc_all_params[:, 1, 5:9] = np.transpose(Na_pers_params(mh_loc_V_mem, mh_loc_Tadj))
	mh_loc_all_params[:, 2, 5:9] = np.transpose(non_spec_params(mh_loc_V_mem))
	mh_loc_all_params[:, 3, 5:9] = np.transpose(musc_params(mh_loc_V_mem, mh_loc_Tadj))
	mh_loc_all_params[:, 4, 5:9] = np.transpose(slow_K_params(mh_loc_V_mem, mh_loc_Tadj))
	mh_loc_all_params[:, 5, 5:9] = np.transpose(fast_K_params(mh_loc_V_mem, mh_loc_Tadj))
	mh_loc_all_params[:, 6, 5:9] = np.transpose(Kv31_params(mh_loc_V_mem, mh_loc_Tadj))
	mh_loc_all_params[:, 7, 5:9] = np.transpose(Ca_HVA_params(mh_loc_V_mem))
	mh_loc_all_params[:, 8, 5:9] = np.transpose(Ca_LVA_params(mh_loc_V_mem, mh_loc_Tadj))
	mh_loc_all_params[:, 9, 5:9] = np.transpose(SK_Ca_params(mh_loc_Ca2))
	mh_loc_all_params[:, 10, 5:9] = np.ones((mh_loc_num_neurons, 4))
	
	if init_m_h:
		#setting all the m values
		mh_loc_all_params[:, :, 3:5] = mh_loc_all_params[:, :, 5:7]

	CaReverse = nernst(mh_loc_Ca2)
	mh_loc_all_params[:, 8, 1] = CaReverse[:] #hey if anything is wrong check here I'm highly skeptical
	mh_loc_all_params[:, 7, 1] = CaReverse[:] #hey if anything is wrong check here I'm highly skeptical

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

def current_contrib(A, V, E, g, m, p, h = 1):
	#hey some of the ions dont have h modeles in the table 1, gonna just assume constant 1 for those until i see otherwise
	I = A*g*(m**p)*h*(V-E)
	return I

def the_old_simulator9000(sim_loc_all_params, sim_loc_V_mem, sim_loc_Ca2, sim_loc_passives, sim_length = 3000, time_step = 0.05, I_ext = None):
	sim_loc_all_params = m_h_stuff(sim_loc_all_params, sim_loc_V_mem, sim_loc_Ca2, True)
	
	sim_loc_gamma = sim_loc_passives[0]
	sim_loc_Tau = sim_loc_passives[1]
	sim_loc_areas = sim_loc_passives[2]
	sim_loc_caps = sim_loc_passives[3]
	
	sim_steps = int(sim_length/time_step)
	if I_ext is None:
		I_ext = np.zeros(sim_steps)
	
	local_num_neurons = len(sim_loc_all_params)
	
	Vs = np.zeros((local_num_neurons, sim_steps))
	Cas = np.zeros((local_num_neurons, sim_steps))
	
	for s in range(sim_steps):
		
		sim_loc_all_params = m_h_stuff(sim_loc_all_params, sim_loc_V_mem, sim_loc_Ca2, glob_Tadj)
			
		for index in range(len(sim_loc_all_params[0])):
			sim_loc_all_params[:, index, 4] += time_step*(sim_loc_all_params[:, index, 6] - sim_loc_all_params[:, index, 4])/sim_loc_all_params[:, index, 8]
			sim_loc_all_params[:, index, 3] += time_step*(sim_loc_all_params[:, index, 5] - sim_loc_all_params[:, index, 3])/sim_loc_all_params[:, index, 7]
			
		current_sum = np.zeros((local_num_neurons, 11))
		for index in range(len(sim_loc_all_params[0])):
			current_sum[:, index] = (current_contrib(sim_loc_areas, sim_loc_V_mem, 
												sim_loc_all_params[:, index, 1], sim_loc_all_params[:, index, 0], 
												sim_loc_all_params[:, index, 3], sim_loc_all_params[:, index, 2], 
												sim_loc_all_params[:, index, 4]))
		#print(np.sum(current_sum, axis = 1)[0])
		dV = time_step*(-1*np.sum(current_sum, axis = 1) + I_ext[s])/(sim_loc_caps)
		#print(dV)
		sim_loc_V_mem += dV
		
		sim_loc_Ca2 += step_Ca2(current_sum[:, 7], current_sum[:, 8], sim_loc_Ca2, sim_loc_gamma, sim_loc_Tau)*time_step
		
		Vs[:, s] = sim_loc_V_mem[:]
		#Vs[:, s] = np.sum(current_sum, axis = 1)
	
	return Vs, sim_loc_all_params, sim_loc_V_mem, sim_loc_Ca2

def initialize_neurons(init_loc_num_neurons = 500):
	
	init_loc_param_limits = define_param_limits()
	
	init_loc_passives = np.zeros((4, init_loc_num_neurons))
	
	init_loc_gamma = np.random.uniform(init_loc_param_limits[11, 0], init_loc_param_limits[14, 1], init_loc_num_neurons)
	init_loc_Tau = np.random.uniform(init_loc_param_limits[12, 0], init_loc_param_limits[15, 1], init_loc_num_neurons)
	
	init_loc_areas = np.random.uniform(init_loc_param_limits[14, 0], init_loc_param_limits[14, 1], init_loc_num_neurons)
	init_loc_caps = np.random.uniform(init_loc_param_limits[15, 0], init_loc_param_limits[15, 1], init_loc_num_neurons)
	
	init_loc_passives[0] = init_loc_gamma
	init_loc_passives[1] = init_loc_Tau
	init_loc_passives[2] = init_loc_areas
	init_loc_passives[3] = init_loc_caps
	
	#reverse potentials in mV
	NaReverse = 50
	KReverse = -85
	HReverse = -45
	LeakReverse = -90
	
	#max conductance, reverse potential, p, m, h, m_infinity, h_infinity, tau_m, tau_h
	init_loc_all_params = np.zeros((init_loc_num_neurons, 11, 9))
	init_loc_all_params[:, :, 0], LeakReverse = randomize_coduct(init_loc_param_limits, init_loc_num_neurons)
	
	#volatile initialization
	init_loc_V_mem = -80*np.ones(init_loc_num_neurons)
	init_loc_Ca2 = 0.001*np.ones(init_loc_num_neurons)
	init_loc_CaReverse = nernst(init_loc_Ca2)*np.ones(init_loc_num_neurons)
	print(LeakReverse)
	
	for n in range(init_loc_num_neurons):
		init_loc_all_params[n, :, 1] = [NaReverse, NaReverse, HReverse, KReverse, KReverse, KReverse, KReverse, init_loc_CaReverse[n], init_loc_CaReverse[n], KReverse, LeakReverse[n]] #hey if anything is wrong check here I'm highly skeptical
		init_loc_all_params[n, :, 2] = [3, 3, 1, 1, 2, 4, 1, 2, 2, 1, 1]

	init_loc_all_params = m_h_stuff(init_loc_all_params, init_loc_V_mem, init_loc_Ca2, glob_Tadj, init_m_h = True)
	
	return init_loc_all_params, init_loc_V_mem, init_loc_Ca2, init_loc_passives 


glob_num_neurons = 500
glob_Tadj = 2.95 #epirically taken from their paper, i don't care about temperature
glob_all_params, glob_V_mem, glob_Ca2, glob_passives = initialize_neurons(glob_num_neurons)

glob_curr_prim = current_primitive(60000, 'square', 0, 0.5, 0.33, 0.66)
glob_curr_prim = current_primitive(60000, 'ramp', 0, 0.8, 0.15, 1)
glob_Vs, glob_all_params, glob_V_mem, glob_Ca2 = the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_passives)
glob_Vs, glob_all_params, glob_V_mem, glob_Ca2 = the_old_simulator9000(glob_all_params, glob_V_mem, glob_Ca2, glob_passives, I_ext = glob_curr_prim)

for i in range(glob_num_neurons):
	plt.plot(glob_Vs[i])
	plt.show()
	print(glob_Vs[i, 0], glob_Vs[i, -1])
