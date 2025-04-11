import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

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

def exp_model(x, A, B, c):
    return A - B * np.exp(-c * x)

def fit_exponential(time_series):
    """
    Fits an exponential model y = A - B * exp(-c * x)
    to the given time series data.

    Parameters:
    - time_series: list or numpy array of y-values over time (x = 0, 1, 2, ...)

    Returns:
    - A, B, c: fitted parameters
    """
    y = np.array(time_series)
    x = np.arange(len(y))

    # Initial guesses: A ~ y[-1], B ~ y[0]-y[-1], c ~ 0.1
    initial_guess = [y[-1], y[0] - y[-1], 0.1]

    params, _ = curve_fit(exp_model, x, y, p0=initial_guess, maxfev=10000)

    A, B, c = params
    return A, B, c

num_neurons = 100
A = 6.28E-4 #area in cm^2?
C = 6.28E-1 #capacitence in nF, something tells me they picked this to mostly cancel with area
time_step = 0.05
sim_length = 3000
stabilize_time = 1000
sim_steps = int(sim_length/time_step)
stable_steps = int(stabilize_time/time_step)

areas = np.random.normal(A, A/5, num_neurons)
caps = np.random.normal(C, C/5, num_neurons)

#conductance in mS/cm^2 ?
mult = 1000
Na_g = 300*mult
CaT_g = 12.5*mult
CaS_g = 10*mult
A_g = 50*mult
KCa_g = 25*mult
Kd_g = 125*mult
H_g = 0.02*mult
Leak_g = 0.03*mult
cond_base = [Na_g, CaT_g, CaS_g, A_g, KCa_g, Kd_g, H_g, Leak_g]

NaReverse = 50
KCaReverse = -80
KdReverse = -80
AReverse = -80
HReverse = -20
LeakReverse = np.random.normal(-80, 10, num_neurons)

#volatile initialization
V_membrane = LeakReverse*np.ones(num_neurons)
Ca2 = 0.05*np.ones(num_neurons)
CaReverse = nernst(Ca2)*np.ones(num_neurons)
I_ext = 0.0*np.ones(num_neurons)

#I genuinely don't know if the time constants are meant to be constant, like yeah they should be but they are functions of V so like?
#I think they do change bc of top of page 3 but like 
#max conductance, reverse potential, p, m, h, m_infinity, h_infinity, tau_m, tau_h
all_params = np.zeros((num_neurons, 8, 9))
for n in range(num_neurons):
	test_random = np.random.rand(8)
	test_random[7] = (np.random.rand()*3 + 1)/4
	all_params[n, :, 0] = cond_base*test_random
	all_params[n, :, 1] = [NaReverse, CaReverse[n], CaReverse[n], AReverse, KCaReverse, KdReverse, HReverse, LeakReverse[n]] #hey if anything is wrong check here I'm highly skeptical
	all_params[n, :, 2] = [3, 3, 3, 3, 4, 4, 1, 1]
	#all_params[n] = mutate(all_params[n])
all_params = m_h_stuff(all_params, V_membrane, Ca2, init_m_h = True)

Vs = np.zeros((num_neurons, sim_steps))
Cas = np.zeros((num_neurons, sim_steps))
Cas_reverse = np.zeros((num_neurons, sim_steps))
current_names = ['Na', 'CaT', 'CaS', 'A', 'KCa', 'Kd', 'H', 'Leak']
currents = np.zeros((num_neurons, sim_steps))

for s in range(stable_steps):

	all_params = m_h_stuff(all_params, V_membrane, Ca2)
	
	I_ext = 0.0*np.ones(num_neurons)
	
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

'''
print('Cap', pearsonr(V_membrane, caps))
print('Area', pearsonr(V_membrane, areas))
for i in range(8):
	print(current_names[i], pearsonr(V_membrane, all_params[:, i, 0]))
print("Leak Reverse", pearsonr(LeakReverse, all_params[:, i, 0]))
'''

current_mult = 0.0
step_up = 6500
duty_cylce = 0.66

for s in range(sim_steps):
	
	all_params = m_h_stuff(all_params, V_membrane, Ca2)
	
		
	if (s%step_up == 0):
		current_mult += 0.03
	I_ext = current_mult*np.ones(num_neurons)*(s%step_up > step_up*(1-duty_cylce))
	
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
	currents[:, s] = I_ext

print('hi i tried')

for n in range(num_neurons):
	
	gaming = derived_ephys_props(Vs[n], time_step)

	fig, ax1 = plt.subplots(figsize=(10, 5))
	
	# Plot voltage
	ax1.set_xlabel('Time (ms)')
	ax1.set_ylabel('Voltage (mV)')
	ax1.plot(np.linspace(0, sim_length, sim_steps), Vs[n], label="voltage", color = [0, 0, 1])
	ax1.tick_params(axis='y')
	
	# Create second y-axis for current
	ax2 = ax1.twinx()
	ax2.set_ylabel('Current (nA)')
	ax2.plot(np.linspace(0, sim_length, sim_steps), currents[n], label="injected current", color = [1, 0, 0])
	ax2.tick_params(axis='y')
	
	fig.tight_layout()
	plt.title("Voltage and Current Over Time_"+str(n))
	plt.show()
	print(n, all_params[n, :, 0]/cond_base)



