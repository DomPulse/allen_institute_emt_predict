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
			post_ses = filename.split("ses-")[-1]
			#print(post_ses.split("_")[0])
			pre_other_stuff = post_ses.split("_")[0]
			try:
				session_search.append(int(pre_other_stuff))
				session_to_path[int(pre_other_stuff)] = dirpath
			except:
				pass
	return session_search, session_to_path

def main():
	meta_data_path = r"F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv"
	manifest_data_path = r"F:\Big_MET_data\2021-09-13_mouse_file_manifest.xlsx"
	counts_path = r"F:\Big_MET_data\trans_data_cpm\20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
	ephys_path = r"F:\Big_MET_data\fine_and_dandi_ephys\000020"
	
	ephys_ses, ses_path_dict = list_all_files(ephys_path)
	#the below genes are on this site as being relevant to ion channels: https://www.guidetopharmacology.org/GRAC/IonChannelListForward?class=VGIC
	#'Kcnk17' and 'Scn2a' ommited as they are not found in the dataset
	gamer_approved_genes = ['Kcnma1', 
							'Kcnn1', 'Kcnn2', 'Kcnn3', 'Kcnn4',
							'Kcnt1', 'Kcnt2', 'Kcnu1', 
							'Catsper1', 'Catsper2', 'Catsper3', 'Catsper4',
							'Tpcn1', 'Tpcn2', 
							'Cnga1', 'Cnga2', 'Cnga3', 'Cnga4',
							'Cngb1', 'Cngb3', 
							'Hcn1', 'Hcn2', 'Hcn3', 'Hcn4',
							'Kcnj1', 'Kcnj2', 'Kcnj12', 'Kcnj4',
							'Kcnj14', 'Kcnj3', 'Kcnj6', 'Kcnj9',
							'Kcnj5', 'Kcnj10', 'Kcnj15', 'Kcnj16',
							'Kcnj8', 'Kcnj11', 'Kcnj13',
							'Ryr1', 'Ryr2', 'Ryr3',
							'Trpa1', 
							'Trpc1', 'Trpc2', 'Trpc3', 'Trpc4',
							'Trpc5', 'Trpc6', 'Trpc7', 
							'Trpm1', 'Trpm2', 'Trpm3', 'Trpm4',
							'Trpm5', 'Trpm6', 'Trpm7', 'Trpm8',
							'Mcoln1', 'Mcoln2', 'Mcoln3', 
							'Pkd2', 'Pkd2l1', 'Pkd2l2',
							'Trpv1', 'Trpv2', 'Trpv3',
							'Trpv4', 'Trpv5', 'Trpv6',
							'Kcnk1', 'Kcnk2', 'Kcnk3', 'Kcnk4',
							'Kcnk5', 'Kcnk6', 'Kcnk7',
							'Kcnk9', 'Kcnk10', 'Kcnk12',
							'Kcnk13', 'Kcnk15', 'Kcnk16', 'Kcnk18',
							'Cacna1s', 'Cacna1c', 'Cacna1d', 'Cacna1f', 'Cacna1a',
							'Cacna1b', 'Cacna1e', 'Cacna1g','Cacna1h', 'Cacna1i',
							'Kcna1', 'Kcna2', 'Kcna3', 'Kcna4',
							'Kcna5', 'Kcna6', 'Kcna7', 'Kcna10',
							'Kcnb1', 'Kcnb2', 
							'Kcnc1', 'Kcnc2', 'Kcnc3', 'Kcnc4',
							'Kcnd1', 'Kcnd2', 'Kcnd3',
							'Kcnf1',
							'Kcng1', 'Kcng2', 'Kcng3', 'Kcng4',
							'Kcnq1', 'Kcnq2', 'Kcnq3', 'Kcnq4', 'Kcnq5',
							'Kcnv1', 'Kcnv2',
							'Kcns1', 'Kcns2', 'Kcns3',
							'Kcnh1', 'Kcnh2', 'Kcnh3', 'Kcnh4',
							'Kcnh5', 'Kcnh6', 'Kcnh7', 'Kcnh8',
							'Hvcn1',
							'Scn1a', 'Scn3a', 'Scn4a',
							'Scn5a', 'Scn8a', 'Scn9a', 'Scn10a', 'Scn11a']
	
	meta_data_df = pd.read_csv(meta_data_path)
	counts_df = pd.read_csv(counts_path)
	counts_df_filtered = counts_df[counts_df["names"].isin(gamer_approved_genes)]
	col_names= list(counts_df_filtered) #these are like the names of the cells or the experiment or whatever

	missing_names = set(gamer_approved_genes) - set(counts_df_filtered['names'])
	print("Missing names:", missing_names)
	
	del counts_df
	gc.collect()
	
	right_row = meta_data_df.index[(meta_data_df['transcriptomics_sample_id'].isin(col_names)) & (meta_data_df['ephys_session_id'].isin(ephys_ses))].tolist()
	
	example_idx = np.random.randint(len(right_row))
	
	full_save_path = r'F:\Big_MET_data\single_pulse_selected_genes'

	in_col_names =  ['file_path', 'exp_index'] + ['second_time_start', 'second_time_end', 'current_second_edge', 'sim_length'] + gamer_approved_genes
	out_col_names = ['file_path', 'exp_index'] + ['voltage_base', 'time_to_first_spike', 'time_to_last_spike',
					  'sag_amplitude', 'sag_ratio1', 'sag_time_constant',
					  'minimum_voltage', 'maximum_voltage', 'spike_count']
	
	for example_idx in range(len(right_row)):
		
		example_col = col_names[example_idx+1] #80% sure this fixes a previous off by one error? like the first index is the names right
		folder_path = f'{ses_path_dict[ephys_ses[example_idx]]}\\'
			
		files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
		all_times, all_currents, all_volts, orginal_lengths = lpcd.give_me_the_stuff(folder_path, files)
		
		input_data_dict = []
		output_data_dict = []
		
		for i in range(len(all_times)):
			stop_at = orginal_lengths[i]-1
			
			try:
				chunks, is_bad_data, diff_current, derived_current_params, derived_efeatures = cpc.i_like_em_chunky(stop_at, all_currents[i], all_times[i], all_volts[i])
				if not is_bad_data:
					#print(folder_path, i, derived_efeatures)
					output_data_dict.append([ses_path_dict[ephys_ses[example_idx]], i] + derived_efeatures)
					#print(filtered_df[example_col].to_list())
					input_data_dict.append([ses_path_dict[ephys_ses[example_idx]], i] + derived_current_params+counts_df_filtered[example_col].to_list())
			except:
				pass
			
			'''
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

		
		plt.bar(gamer_approved_genes, counts_df_filtered[example_col]) #I ~think~ everything was offset by one because of the row name?
		plt.show()
		'''
		
		pd.DataFrame(output_data_dict, columns = out_col_names).to_csv(os.path.join(full_save_path, f'{example_idx}_out_test.csv'))
		pd.DataFrame(input_data_dict, columns = in_col_names).to_csv(os.path.join(full_save_path, f'{example_idx}_in_test.csv'))
		print('save success')
		del input_data_dict
		del output_data_dict
		del all_times, all_currents, all_volts, orginal_lengths
		gc.collect()
	

		
if __name__ == "__main__":
	main()
	gc.collect()
	