import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as col_map
import pandas as pd
import umap
from sklearn.cluster import KMeans
from scipy import stats
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from matplotlib.lines import Line2D

meta_data_path = r"F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv"
counts_path = r"F:\Big_MET_data\trans_data_cpm\20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
raw_ephys_path = r"F:\Big_MET_data\fine_and_dandi_ephys\000020"
derived_ephys_path = 'F:\Big_MET_data\ephys_copy_allen'
allen_institure_direct_ephys_path = 'F:\Big_MET_data\efeat_exp_umap_embedding.csv'
#^ taken from https://github.com/AllenInstitute/All-active-Manuscript/blob/master/assets/aggregated_data/efeat_exp_umap_embedding.csv

neg_current = -0.03
pos_current = 0.12

def get_n_colors(n):

	colors = col_map.get_cmap('hsv', n+1)
	return np.asarray([colors(i)[:3] for i in range(1, n+1)])  # Remove alpha

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

ephys_ses, ses_path_dict = list_all_files(raw_ephys_path)
metadata = pd.read_csv(meta_data_path)
allen_data = pd.read_csv(allen_institure_direct_ephys_path)

simple_cell_types = ['Vip', 'Sst', 'Pvalb', 'Pyr'] #consider adding Lamp5
cell_types_color_dict = {'Vip': 'r', 'Sst': 'g', 'Pvalb': 'b', 'Pyr': 'm', 'Other': 'k'}
metadata['simple_cell_type'] = 'Other'
metadata['my_color'] = 'k'
ephys_properties = ['AHP_depth', 'AP_amplitude_from_voltagebase', 'AP_width', 'mean_frequency', 'steady_state_voltage', 'voltage_base', 'time_to_first_spike']

#chat wrote this :/
legend_elements = [
	Line2D([0], [0], marker='o', color='w', label=ct,
		   markerfacecolor=color, markersize=8)
	for ct, color in cell_types_color_dict.items()
]

for ephys_prop in ephys_properties:
	metadata[ephys_prop] = 0

cells_with_ephys_idxs = []

for data_idx in range(len(metadata['cell_specimen_id'].to_list())):
#for data_idx in range(50):

	try:

		cell_id = int(metadata['cell_specimen_id'].loc[data_idx])
		transcriptomics_sample_id = metadata['transcriptomics_sample_id'].loc[data_idx]
		this_cell_ephys = pd.read_csv(f'{derived_ephys_path}\{cell_id}.csv')
		
		has_pos_curr = np.isclose(this_cell_ephys['current_second_edge'], pos_current, atol = 0.0001)
		has_neg_curr = np.isclose(this_cell_ephys['current_second_edge'], neg_current, atol = 0.0001)
		pos_index = np.where(has_pos_curr)[0]
		neg_index = np.where(has_neg_curr)[0]
		
		ap_width = 0
		for ephys_prop in ephys_properties:
			
			prop_value = np.mean(this_cell_ephys[ephys_prop].loc[pos_index].to_numpy())
			
			if ephys_prop == 'AP_width':
				ap_width = prop_value
			if ephys_prop == 'time_to_first_spike':
				#corrects for when the current stimulus starts
				prop_value -= 0*np.mean(this_cell_ephys['current_second_edge'].loc[pos_index].to_numpy())
				
			metadata[ephys_prop].loc[data_idx] = prop_value
		
		for test_cell_type in simple_cell_types:
			if test_cell_type in metadata['T-type Label'].loc[data_idx]:
				metadata['simple_cell_type'].loc[data_idx] = test_cell_type
				metadata['my_color'].loc[data_idx] = cell_types_color_dict[test_cell_type]
		
		if np.sum(pos_index) > 0 and ap_width > 0:
			cells_with_ephys_idxs.append(data_idx)

	except:
		pass
		#print(f'{cell_id} data not found!')
		
cells_with_ephys_idxs = np.asarray(cells_with_ephys_idxs)
transcriptomics_sample_ids = metadata['transcriptomics_sample_id'].loc[cells_with_ephys_idxs].to_numpy()
transcriptomics_sample_ids = transcriptomics_sample_ids.astype(str)

embedding_dim = 2
embedding = umap.UMAP(n_components=embedding_dim, n_neighbors=10, random_state=2).fit_transform(
	metadata.loc[cells_with_ephys_idxs, ephys_properties].to_numpy(dtype=np.float64))

plt.title('umap positions of ephys')
plt.scatter(embedding[:, 0], embedding[:, 1], color = metadata['my_color'].loc[cells_with_ephys_idxs].to_list(), marker='o', alpha = 0.4)
plt.legend(handles=legend_elements)
plt.show()

embedding = umap.UMAP(n_components=embedding_dim, n_neighbors=10, random_state=2).fit_transform(
	allen_data[ephys_properties].to_numpy(dtype=np.float64))

plt.title('umap positions of ephys copied from Allen github')
plt.scatter(embedding[:, 0], embedding[:, 1], color = [cell_types_color_dict[item] for item in allen_data['ttype'].to_list()], marker='o', alpha = 0.4)
plt.legend(handles=legend_elements)
plt.show()

#plots histograms of each cell type's gene expression
'''
for gene_of_interest in ion_channel_genes + high_variance_genes:
	
	plt.title(gene_of_interest)
	ion_channel_gene_idx = gene_names.index(gene_of_interest)
	all_dists = []
	for cell_type in range(num_clusters):
		this_cell_type_idxs = np.where(labels == cell_type)[0]
		gene_expression_hist = np.log2(gene_data.iloc[ion_channel_gene_idx, this_cell_type_idxs].values + 1)
		plt.hist(gene_expression_hist, color = color_options[cell_type], alpha = 0.6)
		all_dists.append(list(gene_expression_hist))
	plt.show()
	try:
		result = stats.kruskal(*all_dists)
		print(f'{gene_of_interest}: {result.pvalue}')
	except:
		print(f'{gene_of_interest} is harshing my mellow')
'''


