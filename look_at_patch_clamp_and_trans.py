import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import load_patch_clamp_data as lpcd
import classify_patch_clamp as cpc
import gc

def list_all_files(root_folder):
	session_search = []
	session_to_path = {}
	for dirpath, dirnames, filenames in os.walk(root_folder):
		for filename in filenames:
			post_ses = filename.split('ses-')[-1]
			#print(post_ses.split('_')[0])
			pre_other_stuff = post_ses.split('_')[0]
			try:
				session_search.append(int(pre_other_stuff))
				session_to_path[int(pre_other_stuff)] = dirpath
			except:
				pass
	return session_search, session_to_path

def main():
	meta_data_path = r'F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv'
	manifest_data_path = r'F:\Big_MET_data\2021-09-13_mouse_file_manifest.xlsx'
	counts_path = r'F:\Big_MET_data\trans_data_cpm\20200513_Mouse_PatchSeq_Release_cpm.v2.csv'
	ephys_path = r'F:\Big_MET_data\fine_and_dandi_ephys\000020'
	
	ephys_ses, ses_path_dict = list_all_files(ephys_path)
	
	meta_data_df = pd.read_csv(meta_data_path)
	#manifest_data_df = pd.read_excel(manifest_data_path)
	counts_df = pd.read_csv(counts_path)
	
	threshold = 0.95
	zero_counts = (counts_df != 0).sum(axis=1)
	max_zeros = threshold * counts_df.shape[1]
	filtered_df = counts_df[zero_counts >= max_zeros]
	
	col_names= list(filtered_df)
	gene_names = list(filtered_df[col_names[0]])
	
	in_ephys = meta_data_df.index[meta_data_df['ephys_session_id'].isin(ephys_ses)].tolist()
	in_trans = meta_data_df.index[meta_data_df['transcriptomics_sample_id'].isin(col_names)].tolist()
	right_row = meta_data_df.index[(meta_data_df['transcriptomics_sample_id'].isin(col_names)) & (meta_data_df['ephys_session_id'].isin(ephys_ses))].tolist()
	
	example_idx = np.random.randint(len(right_row))
	
	example_col = col_names[example_idx]
	folder_path = f'{ses_path_dict[ephys_ses[example_idx]]}\\'
	
	print(folder_path)
	
	files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
	all_times, all_currents, all_volts, orginal_lengths = lpcd.give_me_the_stuff(folder_path, files)
	
	input_data_dict = []
	output_data_dict = []
	in_col_names =  ['file_path', 'exp_index'] + ['current_type', 'second_time', 'last_time', 'last_time_end', 'current_second_edge', 'current_last_edge', 'integrated_current', 'min_current', 'max_current', 'num_edge', 'sim_time']+ gene_names
	out_col_names = ['file_path', 'exp_index'] + ['voltage_base', 'time_to_first_spike', 'time_to_last_spike',
					  'sag_amplitude', 'sag_ratio1', 'sag_time_constant',
					  'minimum_voltage', 'maximum_voltage', 'AP1_width', 'APlast_width', 
					  'deactivation_time_constant', 'activation_time_constant', 'inactivation_time_constant']
	for i in range(len(all_times)):
		dicts_key = f"{ses_path_dict[ephys_ses[example_idx]]}_{i}"
		stop_at = orginal_lengths[i]-1
		
		try:
			chunks, is_bad_data, diff_current, derived_current_params, derived_efeatures = cpc.i_like_em_chunky(stop_at, all_currents[i], all_times[i], all_volts[i])
			if not is_bad_data:
				#print(folder_path, i, derived_efeatures)
				output_data_dict.append([ses_path_dict[ephys_ses[example_idx]], i] + derived_efeatures)
				#print(filtered_df[example_col].to_list())
				input_data_dict.append([ses_path_dict[ephys_ses[example_idx]], i] + derived_current_params+filtered_df[example_col].to_list())
		except:
			pass
		
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
	
	plt.bar(gene_names, filtered_df[example_col])
	plt.show()
	
	pd.DataFrame(output_data_dict, columns = out_col_names).to_csv('out_test.csv')
	pd.DataFrame(input_data_dict, columns = in_col_names).to_csv('in_test.csv')

if __name__ == "__main__":
	main()
	gc.collect()
	