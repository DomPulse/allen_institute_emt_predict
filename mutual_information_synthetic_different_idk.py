import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

def compute_mutual_information(data_path, input_cols, output_cols, save_path):
	"""
	Loads a CSV file, computes mutual information between every input-output pair,
	and saves results to a new CSV.

	Parameters:
	----------
	data_path : str
		Path to input CSV file.
	input_cols : list of str
		List of column names to use as input features.
	output_cols : list of str
		List of column names to use as output features.
	save_path : str
		Path to save results CSV.
	"""

	# --- Load data ---
	data = pd.read_csv(data_path)
	data = data[data['200_spike_count'] > 0]
	data.replace([np.inf, -np.inf], np.nan, inplace=True)
	data[current_named_ephys_feat] = data[current_named_ephys_feat].fillna(0)

	results = []

	# --- Compute mutual information for each input-output pair ---
	for out_col in output_cols:
		print(out_col)
		y = data[out_col].to_numpy(dtype=np.float64)

		for in_col in input_cols:
			x = data[in_col].to_numpy(dtype=np.float64).reshape(-1, 1)

			# Compute MI (mutual_info_regression expects 2D X)
			mi = mutual_info_regression(x, y, random_state=0)
			results.append({
				"input": in_col,
				"output": out_col,
				"mutual_information": mi[0]
			})

	# --- Save results ---
	results_df = pd.DataFrame(results)
	results_df.to_csv(save_path, index=False)
	print(f"Mutual information results saved to {save_path}")


if __name__ == "__main__":
	# --- Example usage ---
	data_path = r'F:\arbor_ubuntu\synth_data_one_morph_more_stims_v2.csv'

	cell_props = [
		#section_name
		'soma_Ra',
		'soma_g_pas',
		'soma_e_pas',
		'axon_g_pas',
		'axon_e_pas',
		'apic_g_pas',
		'apic_e_pas',
		'dend_g_pas',
		'dend_e_pas',
		'soma_cm',
		'axon_cm',
		'axon_Ra',
		'apic_cm',
		'apic_Ra',
		'dend_cm',
		'dend_Ra',
		'axon_gbar_NaV',
		'axon_gbar_K_T',
		'axon_gbar_Kd',
		'axon_gbar_Kv2like',
		'axon_gbar_Kv3_1',
		'axon_gbar_SK',
		'axon_gbar_Ca_HVA',
		'axon_gbar_Ca_LVA',
		'axon_gamma_CaDynamics',
		'axon_decay_CaDynamics',
		'soma_gbar_NaV',
		'soma_gbar_SK',
		'soma_gbar_Kv3_1',
		'soma_gbar_Ca_HVA',
		'soma_gbar_Ca_LVA',
		'soma_gamma_CaDynamics',
		'soma_decay_CaDynamics',
		'soma_gbar_Ih',
		'apic_gbar_NaV',
		'apic_gbar_Kv3_1',
		'apic_gbar_Im_v2',
		'apic_gbar_Ih',
		'dend_gbar_NaV',
		'dend_gbar_Kv3_1',
		'dend_gbar_Im_v2',
		'dend_gbar_Ih',
		'soma_ra'
		]	

	current_named_ephys_feat = []
	neg_currents_to_test = [-200, -100, -50]
	pos_currents_to_test = [50, 100, 150, 200]
	neg_ephys_properties = ['sag_ratio1', 'steady_state_voltage_stimend']
	pos_ephys_properties = ['AP_peak_upstroke', 'AP_peak_downstroke',
					 'voltage_base', 'AHP2_depth_from_peak',
					 'AP_amplitude_from_voltagebase', 'AP_width',
					 'AP_height', 'AP_amplitude', 'steady_state_voltage', 'time_to_first_spike', 'spike_count', 'time_to_last_spike']

	for current in neg_currents_to_test:
		for ephys_feat in neg_ephys_properties:
			name = f'{current}_{ephys_feat}'
			current_named_ephys_feat.append(name)

	for current in pos_currents_to_test:
		for ephys_feat in pos_ephys_properties:
			name = f'{current}_{ephys_feat}'
			current_named_ephys_feat.append(name)
			
	save_path = "mutual_information_results_synth_one_morph.csv"
	
	print(current_named_ephys_feat)
	
	compute_mutual_information(data_path, cell_props, current_named_ephys_feat, save_path)
