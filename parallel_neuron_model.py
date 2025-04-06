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

num_neurons = 500
A = 6.28E-4 #area in cm^2?
C = 6.28E-1 #capacitence in nF, something tells me they picked this to mostly cancel with area
time_step = 0.05
sim_length = 500
sim_steps = int(sim_length/time_step)

areas = np.random.normal(A, A/10, num_neurons)
caps = np.random.normal(C, C/10, num_neurons)

#conductance in mS/cm^2 ?
mult = 1000
Na_g = 200*mult
CaT_g = 12.5*mult
CaS_g = 10*mult
A_g = 50*mult
KCa_g = 25*mult
Kd_g = 125*mult
H_g = 0.5*mult
Leak_g = 0.5*mult

NaReverse = 50
KCaReverse = -80
KdReverse = -80
AReverse = -80
HReverse = -20
LeakReverse = -50

#volatile initialization
V_membrane = -50*np.ones(num_neurons)
Ca2 = 0.05*np.ones(num_neurons)
CaReverse = nernst(Ca2)*np.ones(num_neurons)
I_ext = 5*np.ones(num_neurons)

#I genuinely don't know if the time constants are meant to be constant, like yeah they should be but they are functions of V so like?
#I think they do change bc of top of page 3 but like 
#max conductance, reverse potential, p, m, h, m_infinity, h_infinity, tau_m, tau_h
all_params = np.zeros((num_neurons, 8, 9))
for n in range(num_neurons):
	all_params[n, :, 0] = [Na_g, CaT_g, CaS_g, A_g, KCa_g, Kd_g, H_g, Leak_g]*np.random.rand(8)
	all_params[n, :, 1] = [NaReverse, CaReverse[n], CaReverse[n], AReverse, KCaReverse, KdReverse, HReverse, LeakReverse] #hey if anything is wrong check here I'm highly skeptical
	all_params[n, :, 2] = [3, 3, 3, 3, 4, 4, 1, 1]
all_params = m_h_stuff(all_params, V_membrane, Ca2, init_m_h = True)

Vs = np.zeros((num_neurons, sim_steps))
Cas = np.zeros((num_neurons, sim_steps))
Cas_reverse = np.zeros((num_neurons, sim_steps))
current_names = ['leak', 'Na', 'CaT', 'CaS', 'A', 'KCa', 'Kd', 'H']
currents = np.zeros((num_neurons, 8, sim_steps))

for s in range(sim_steps):
	
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

for n in range(num_neurons):
	plt.plot(Vs[n])
	plt.title('voltage')
	plt.show()
