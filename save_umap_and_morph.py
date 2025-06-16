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
	umap_and_meta_path = r"F:\Big_MET_data\umap_pos.csv"
	umap_and_meta_data = pd.read_csv(umap_and_meta_path)
	
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
		
	col_names_for_my_csv = list(umap_and_meta_data.columns) + morph_features
	
	combo_data_dict = []
	for example_idx in range(len(umap_and_meta_data.iloc[:, 0])):
	#for example_idx in range(5):
		cell_id = umap_and_meta_data['cell_specimen_id'].loc[example_idx]
		swc_path = f"F:\Big_MET_data\just_morpho\{cell_id}_raw.swc"
		try:
			test_data = Data(morphology_from_swc(swc_path))
			gaimen = my_feat_pics(test_data)
			my_morpho_data = []
			for key, val in gaimen.items():
				my_morpho_data.append(val)
			#print(cell_id, correct_eses_id, ephys_path, transcriptomics_sample_id)
			my_meta_data = list(umap_and_meta_data.iloc[example_idx, :])
			combo_data_dict.append(my_meta_data + my_morpho_data)
		except:
			print('no morphology to parse')
			
	print(example_idx)
	pd.DataFrame(combo_data_dict, columns = col_names_for_my_csv).to_csv(os.path.join(save_path, 'umap_and_morph.csv'))
	del combo_data_dict
	gc.collect()
		
if __name__ == "__main__":
	main()
	