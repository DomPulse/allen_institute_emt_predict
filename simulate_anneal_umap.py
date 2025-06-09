import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import umap
import seaborn as sns
from sklearn.cluster import KMeans
import os
import random
from sklearn.neighbors import NearestNeighbors

#many would say chat gpt wrote this function
def get_nearest_neighbor_values(array_of_points, query_points, l, values_array):
	"""
	For each query point, find the l nearest neighbors in `array_of_points`,
	and return the corresponding values from `values_array`.
	
	Assumes `values_array` is the same length as `array_of_points`.
	"""
	nbrs = NearestNeighbors(n_neighbors=l+1, algorithm='auto', n_jobs=10)
	nbrs.fit(array_of_points)

	distances, indices = nbrs.kneighbors(query_points)

	# Remove self-match (first column)
	trimmed_indices = indices[:, 1:]
	
	# Lookup corresponding values
	neighbor_values = np.take(values_array, trimmed_indices)

	return neighbor_values

#many would say chat gpt wrote this function
def replace_k_elements(daughter_list, mother_list, k):
	current_set = set(daughter_list)
	available_pool = list(set(mother_list) - current_set)

	# Sanity check
	if k > len(available_pool):
		raise ValueError("Not enough unique elements in mother_list to replace.")

	# Choose k elements to remove from daughter list
	to_remove = random.sample(daughter_list, k)

	# Choose k new elements from the available pool
	to_add = random.sample(available_pool, k)

	# Replace
	updated_daughter = daughter_list.copy()
	for r, a in zip(to_remove, to_add):
		idx = updated_daughter.index(r)
		updated_daughter[idx] = a

	return updated_daughter

def make_age_numerical(given_age):
	age = int(given_age.split('P')[1])
	return age

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

def get_n_colors(n):

	colors = cm.get_cmap('hsv', n)
	return [colors(i)[:3] for i in range(n)]  # Remove alpha

def norm_col(array):
	min_by_col = np.min(array, axis = 0) 
	array = np.subtract(array, min_by_col)
	max_by_col = np.max(array, axis = 0)
	return np.divide(array, max_by_col + 1E-8)

save_path = r"F:\Big_MET_data"
meta_data_path = r"F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv"
counts_path = r"F:\Big_MET_data\trans_data_cpm\20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
morph_path = "F:\Big_MET_data\single_pulses_only"
ephys_path = r"F:\Big_MET_data\fine_and_dandi_ephys\000020"
output_data_dir = 'F:\Big_MET_data\derived_ephys'
ephys_ses, ses_path_dict = list_all_files(ephys_path)
meta_data_df = pd.read_csv(meta_data_path)
counts_df = pd.read_csv(counts_path)

col_names = list(counts_df) #these are like the names of the cells or the experiment or whatever
valid_ephys_row = meta_data_df.index[(meta_data_df['transcriptomics_sample_id'].isin(col_names)) & (meta_data_df['ephys_session_id'].isin(ephys_ses))].tolist()
meta_data_df = meta_data_df.iloc[valid_ephys_row]
gene_names = list(counts_df[col_names[0]])

interesting_meta = ['transcriptomics_sample_id', 'cell_specimen_id', 'ephys_session_id', 'hemisphere', 'structure', 'biological_sex', 'age']
filt_meta_data_df = meta_data_df[interesting_meta]

#filter out some of the ages that are in a different format
valid_age_rows = filt_meta_data_df.index[(filt_meta_data_df['age'].str.contains('P'))].tolist()
filt_meta_data_df = filt_meta_data_df.iloc[valid_age_rows]

#convert all those ages to being numbers
ages = filt_meta_data_df['age'].tolist()
for index, age in enumerate(ages):
	ages[index] = make_age_numerical(age)
ages = np.asarray(ages)
filt_meta_data_df['age'] = ages

#add in the full ephys path
correct_eses_paths = filt_meta_data_df['ephys_session_id'].tolist()
for index, eses_id in enumerate(correct_eses_paths):
	correct_eses_paths[index] = ses_path_dict[eses_id]
filt_meta_data_df['ephys_path'] = correct_eses_paths

#sex and hemisphere are binary and thus don't need two columns
bin_cat_cols = ['hemisphere', 'biological_sex']
decode_meta_data_df = pd.get_dummies(filt_meta_data_df, columns = bin_cat_cols, drop_first=True)

num_clusters = 10
embed_dim = 2

gene_data = pd.read_csv(counts_path, index_col=0)
gene_data_numpy = gene_data.to_numpy()
gene_std = np.std(gene_data_numpy, axis = 1)
thresh = np.sort(gene_std)[-500] #takes the 500 highest std genes?
genes_over_thresh_bin = gene_std > thresh
genes_over_thresh = np.where(genes_over_thresh_bin)[0]
metadata = pd.read_csv(meta_data_path)
gene_names = list(gene_data.index)
random_gene_selection = random.sample(gene_names, 500)

embedding = umap.UMAP(n_components=embed_dim, n_neighbors=25).fit_transform(
    np.log2(gene_data[decode_meta_data_df['transcriptomics_sample_id']].loc[random_gene_selection].values.T + 1)
)
embedding = np.asarray(embedding)

kmeans = KMeans(n_clusters = num_clusters)
labels = kmeans.fit_predict(embedding)

decode_meta_data_df['just_my_type'] = labels

centers = np.zeros((num_clusters, embed_dim))
pos_relative = np.zeros((embedding.shape[0], embed_dim))
for j in range(num_clusters):
	this_clust_idxs = np.where(labels == j)[0]
	centers[j] = np.mean(embedding[this_clust_idxs], axis = 0)

for idx, i in enumerate(labels):
	pos_relative[idx] = embedding[idx] - centers[i]

for k in range(embed_dim):
	decode_meta_data_df[f'{k}_dim_pos'] = embedding[:, k]
	#decode_meta_data_df[f'{k}_dim_pos'] = norm_col(pos_relative[:, k])
	
class_cal_cols = ['structure', 'just_my_type']
decode_meta_data_df = pd.get_dummies(decode_meta_data_df, columns = class_cal_cols)
decode_meta_data_df.index = range(decode_meta_data_df.shape[0]) #renames the rows just numerically

#we don't like true and false, we want binary
to_be_int = ['hemisphere_right', 'biological_sex_M', 'structure_VISa5', 'structure_VISa6a', 'structure_VISal1', 'structure_VISal2/3', 'structure_VISal4', 'structure_VISal5', 'structure_VISal6a', 'structure_VISam2/3', 'structure_VISam4', 'structure_VISam5', 'structure_VISam6a', 'structure_VISl1', 'structure_VISl2/3', 'structure_VISl4', 'structure_VISl5', 'structure_VISl6a', 'structure_VISl6b', 'structure_VISli1', 'structure_VISli2/3', 'structure_VISli4', 'structure_VISli5', 'structure_VISli6a', 'structure_VISli6b', 'structure_VISp', 'structure_VISp1', 'structure_VISp2/3', 'structure_VISp4', 'structure_VISp5', 'structure_VISp6a', 'structure_VISp6b', 'structure_VISpl2/3', 'structure_VISpl4', 'structure_VISpl5', 'structure_VISpl6a', 'structure_VISpm1', 'structure_VISpm2/3', 'structure_VISpm4', 'structure_VISpm5', 'structure_VISpm6a', 'structure_VISpor1', 'structure_VISpor2/3', 'structure_VISpor4', 'structure_VISpor5', 'structure_VISpor6a', 'structure_VISpor6b', 'structure_VISrl2/3', 'structure_VISrl4', 'structure_VISrl5', 'structure_VISrl6a', 'just_my_type_0', 'just_my_type_1', 'just_my_type_2', 'just_my_type_3', 'just_my_type_4', 'just_my_type_5', 'just_my_type_6', 'just_my_type_7', 'just_my_type_8', 'just_my_type_9']
decode_meta_data_df[to_be_int] = decode_meta_data_df[to_be_int].astype(int)


print(list(decode_meta_data_df.columns))
decode_meta_data_df.to_csv(os.path.join(save_path, 'annealed_umap_pos.csv'))

approved_input_features = ['0_dim_pos', '1_dim_pos']
input_current = 0.15 #we only want to look at one current input level so the model doesn't have to learn dependence on that in addition to the morphology and trans stuff
output_feature = 'steady_state_voltage_stimend' #only predicting one output feature at a time, some things like time to first spike, are not always going to have a meaningful value for every experiment so it's just easiest to do sperate models for each feature
mean_output_feature = []
valid_inputs = []
cells_with_this_efeature = []
for data_idx in range(len(decode_meta_data_df['cell_specimen_id'].to_list())):
#for data_idx in range(50):
	#print(just_trans_data['ephys_path'].loc[data_idx], just_trans_data['cell_id'].loc[data_idx]) 
	
	try:
		cell_id = int(decode_meta_data_df['cell_specimen_id'].loc[data_idx])
		
		this_cell_ephys = pd.read_csv(f'{output_data_dir}\{cell_id}.csv')
	

		this_current_ephys = this_cell_ephys[np.isclose(this_cell_ephys['current_second_edge'], input_current, atol = 0.0001)]
		
		#we want to ignore things with value 0 for certain features
		#like we don't want to inclde time to first spike when there were no spikes, ya dig?
		output_features = this_current_ephys[this_current_ephys[output_feature] != 0][output_feature].to_numpy()
		
		if len(output_features) > 0:
			mean_output_feature.append(np.mean(output_features))
			this_input_data = decode_meta_data_df.loc[data_idx, approved_input_features].to_numpy().astype(np.float64)
			valid_inputs.append(list(this_input_data))
			cells_with_this_efeature.append(data_idx)
	except:
		pass
		#print(f'{cell_id} data not found!')
	
mean_output_feature = np.asarray(mean_output_feature)		
valid_inputs = np.asarray(valid_inputs)
mean_output_feature = norm_col(mean_output_feature)
bottom_percentile = np.percentile(mean_output_feature, 10)
top_percentile = np.percentile(mean_output_feature, 90)

current_loss = 1
temp = 0.014
cooling = 0.99999
for i in range(100000):
	cand_random_gene_selection = replace_k_elements(random_gene_selection, gene_names, 50)
	embedding = umap.UMAP(n_components=embed_dim, n_neighbors=25).fit_transform(
	    np.log2(gene_data[decode_meta_data_df['transcriptomics_sample_id']].loc[cand_random_gene_selection].values.T + 1)
	)
	embedding = np.asarray(embedding)[cells_with_this_efeature]
	embed_dist_mse = np.mean(np.square(mean_output_feature - np.mean(get_nearest_neighbor_values(embedding, embedding, 5, mean_output_feature), axis = 1)))
	
	if embed_dist_mse < current_loss:
		accept = True
	else:
		delta = embed_dist_mse - current_loss
		accept_prob = np.exp(-1*delta/temp)
		accept = np.random.rand() < accept_prob
	
	if accept:
		random_gene_selection = cand_random_gene_selection
		current_loss = embed_dist_mse
	
	temp *= cooling	
	
	plt.scatter(
	    embedding[:,0],
		embedding[:,1],
	    s=10,
	    c=mean_output_feature,
		vmin = bottom_percentile,
		vmax = top_percentile,
	    cmap="viridis",
	    edgecolor="none"
	)
	plt.show()
	print(f'iteration: {i}, error: {embed_dist_mse:.5f}, temp: {temp:.5f}')
	if i%100 == 0:
		np.save('annealed_genes', np.asarray(random_gene_selection))

np.save('annealed_genes', np.asarray(random_gene_selection))