import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc

from neuron_morphology.swc_io import morphology_from_swc
from neuron_morphology.feature_extractor.data import Data
from neuron_morphology.features.branching.bifurcations import num_outer_bifurcations
from neuron_morphology.feature_extractor.utilities import unnest
from neuron_morphology.feature_extractor.feature_extractor import FeatureExtractor
from neuron_morphology.features.intrinsic import max_branch_order, num_tips, num_branches, num_nodes
from neuron_morphology.features.size import max_euclidean_distance, mean_diameter, total_length, total_surface_area, total_volume
from neuron_morphology.features.path import max_path_distance


def my_feat_pics(local_test_data):
	
	features = [
	    mean_diameter,
	    max_branch_order,
	    max_euclidean_distance,
		max_path_distance,
		num_outer_bifurcations,
		num_branches,
		num_nodes,
		num_tips,
		total_length,
		total_surface_area,
		total_volume
	]
	
	results = (
	    FeatureExtractor()
	    .register_features(features)
	    .extract(local_test_data)
	    .results
	)
	
	
	return unnest(results)

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
				session_to_path[int(pre_other_stuff)] = f'{dirpath}\\{filename}'
			except:
				pass
	return session_search, session_to_path

def main():
	
	save_path = r"F:\Big_MET_data"
	meta_data_path = r"F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv"
	counts_path = r"F:\Big_MET_data\trans_data_cpm\20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
	morph_path = "F:\Big_MET_data\single_pulses_only"
	ephys_path = r"F:\Big_MET_data\fine_and_dandi_ephys\000020"
	ephys_ses, ses_path_dict = list_all_files(ephys_path)
	meta_data_df = pd.read_csv(meta_data_path)
	counts_df = pd.read_csv(counts_path)
	
	morph_features = [
	    'mean_diameter',
	    'max_branch_order',
	    'max_euclidean_distance',
		'max_path_distance',
		'num_outer_bifurcations',
		'num_branches',
		'num_nodes',
		'num_tips',
		'total_length',
		'total_surface_area',
		'total_volume']

	col_names = list(counts_df) #these are like the names of the cells or the experiment or whatever
	right_row = meta_data_df.index[(meta_data_df['transcriptomics_sample_id'].isin(col_names)) & (meta_data_df['ephys_session_id'].isin(ephys_ses))].tolist()
	gene_names = list(counts_df[col_names[0]])
		
	col_names_for_my_csv = ['ephys_path', 'cell_id', 'correct_eses_id', 'transcriptomics_sample_id'] + morph_features + gene_names
	batch_size = 100
	num_batches = len(right_row)//batch_size
	
	combo_data_dict = []
	for example_idx in range(15):
		transcriptomics_sample_id = col_names[example_idx+1]
		cell_id = meta_data_df['cell_specimen_id'].loc[example_idx]
		correct_eses_id = meta_data_df['ephys_session_id'].loc[example_idx]
		ephys_path = ses_path_dict[correct_eses_id]
		swc_path = f"F:\Big_MET_data\just_morpho\{cell_id}_raw.swc"
		try:
			test_data = Data(morphology_from_swc(swc_path))
			gaimen = my_feat_pics(test_data)
			my_morpho_data = []
			for key, val in gaimen.items():
				my_morpho_data.append(val)
			#print(cell_id, correct_eses_id, ephys_path, transcriptomics_sample_id)
			my_meta_data = [ephys_path, cell_id, correct_eses_id, transcriptomics_sample_id]
			combo_data_dict.append(my_meta_data + my_morpho_data + counts_df[transcriptomics_sample_id].to_list())
		except:
			print('no morphology to parse')
			
	
	pd.DataFrame(combo_data_dict, columns = col_names_for_my_csv).to_csv(os.path.join(save_path, f'combo_morph_trans.csv'))
	del combo_data_dict
	gc.collect()
		
if __name__ == "__main__":
	main()
	