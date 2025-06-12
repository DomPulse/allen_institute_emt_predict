import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import umap
import seaborn as sns
from sklearn.cluster import KMeans
import os
from mpl_toolkits.mplot3d import Axes3D

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

ion_channel_genes = ['Kcnma1', 
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

num_clusters = 11
embed_dim = 3

metadata = pd.read_csv(meta_data_path)
gene_data = pd.read_csv(counts_path, index_col=0)
gene_names = list(gene_data.index)
gene_data_numpy = gene_data.to_numpy()
gene_std = np.std(gene_data_numpy, axis = 1)
thresh = np.sort(gene_std)[-500] #takes the 500 highest std genes?
genes_over_thresh_bin = gene_std > thresh
genes_over_thresh = np.where(genes_over_thresh_bin)[0]
metadata = pd.read_csv(meta_data_path)
gene_names = list(gene_data.index)
mask = np.isin(gene_names, ion_channel_genes)
high_variance_genes = list(np.asarray(gene_names)[genes_over_thresh])
indexes = np.where(mask)[0]
print(indexes)

marker_genes_for_umap = gene_names#pd.read_csv("select_markers.csv", index_col=0)

embedding = umap.UMAP(n_components=embed_dim, n_neighbors=25).fit_transform(
    np.log2(gene_data[decode_meta_data_df['transcriptomics_sample_id']].loc[high_variance_genes].values.T + 1)
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

color_options = get_n_colors(num_clusters)
decode_meta_data_df['r'] = ''
decode_meta_data_df['g'] = ''
decode_meta_data_df['b'] = ''
for idx, i in enumerate(labels):
	pos_relative[idx] = embedding[idx] - centers[i]

	decode_meta_data_df['r'].iloc[idx] = list(color_options[i])[0]
	decode_meta_data_df['g'].iloc[idx] = list(color_options[i])[1]
	decode_meta_data_df['b'].iloc[idx] = list(color_options[i])[2]

for k in range(embed_dim):
	decode_meta_data_df[f'{k}_dim_pos'] = norm_col(pos_relative[:, k])
	decode_meta_data_df[f'{k}_embed_pos'] = embedding[:, k]
	
class_cal_cols = ['structure', 'just_my_type']
decode_meta_data_df = pd.get_dummies(decode_meta_data_df, columns = class_cal_cols)
decode_meta_data_df.index = range(decode_meta_data_df.shape[0]) #renames the rows just numerically

#we don't like true and false, we want binary
to_be_int = ['hemisphere_right', 'biological_sex_M', 'structure_VISa5', 'structure_VISa6a', 'structure_VISal1', 'structure_VISal2/3', 'structure_VISal4', 'structure_VISal5', 'structure_VISal6a', 'structure_VISam2/3', 'structure_VISam4', 'structure_VISam5', 'structure_VISam6a', 'structure_VISl1', 'structure_VISl2/3', 'structure_VISl4', 'structure_VISl5', 'structure_VISl6a', 'structure_VISl6b', 'structure_VISli1', 'structure_VISli2/3', 'structure_VISli4', 'structure_VISli5', 'structure_VISli6a', 'structure_VISli6b', 'structure_VISp', 'structure_VISp1', 'structure_VISp2/3', 'structure_VISp4', 'structure_VISp5', 'structure_VISp6a', 'structure_VISp6b', 'structure_VISpl2/3', 'structure_VISpl4', 'structure_VISpl5', 'structure_VISpl6a', 'structure_VISpm1', 'structure_VISpm2/3', 'structure_VISpm4', 'structure_VISpm5', 'structure_VISpm6a', 'structure_VISpor1', 'structure_VISpor2/3', 'structure_VISpor4', 'structure_VISpor5', 'structure_VISpor6a', 'structure_VISpor6b', 'structure_VISrl2/3', 'structure_VISrl4', 'structure_VISrl5', 'structure_VISrl6a', 'just_my_type_0', 'just_my_type_1', 'just_my_type_2', 'just_my_type_3', 'just_my_type_4', 'just_my_type_5', 'just_my_type_6', 'just_my_type_7', 'just_my_type_8', 'just_my_type_9', 'just_my_type_10']
decode_meta_data_df[to_be_int] = decode_meta_data_df[to_be_int].astype(int)

#gives a list of columns where there just aren't enough samples imo
col_sums = decode_meta_data_df.sum()
cols_to_delete = []
for col in to_be_int:
	if col_sums[col] < 40:
		#print(col, col_sums[col])
		cols_to_delete.append(col)
print(cols_to_delete)
print(decode_meta_data_df)

#actually deletes given columns and rows
for col in cols_to_delete:
	decode_meta_data_df = decode_meta_data_df[decode_meta_data_df[col] != 1]
	decode_meta_data_df = decode_meta_data_df.drop(col, axis=1)

print(list(decode_meta_data_df.columns))

r = decode_meta_data_df['r'].to_numpy()
g = decode_meta_data_df['g'].to_numpy()
b = decode_meta_data_df['b'].to_numpy()
colors = np.stack((r, g, b), axis=1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(decode_meta_data_df['0_embed_pos'], decode_meta_data_df['1_embed_pos'], decode_meta_data_df['2_embed_pos'], color = colors, marker='o', alpha = 0.1)
plt.show()

input_current = 0.15 #we only want to look at one current input level so the model doesn't have to learn dependence on that in addition to the morphology and trans stuff
output_feature = 'steady_state_voltage' #only predicting one output feature at a time, some things like time to first spike, are not always going to have a meaningful value for every experiment so it's just easiest to do sperate models for each feature
decode_meta_data_df['mean_resting_volt'] = 0.0
for data_idx in range(len(decode_meta_data_df['cell_specimen_id'].to_list())):
#for data_idx in range(50):
	#print(just_trans_data['ephys_path'].loc[data_idx], just_trans_data['cell_id'].loc[data_idx]) 
	
	try:
		cell_id = int(decode_meta_data_df['cell_specimen_id'].loc[data_idx])
		
		this_cell_ephys = pd.read_csv(f'{output_data_dir}\{cell_id}.csv')
		decode_meta_data_df['mean_resting_volt'].loc[data_idx] = np.mean(this_cell_ephys['steady_state_voltage'].to_numpy())
	
	except:
		pass
		#print(f'{cell_id} data not found!')
		
decode_meta_data_df = decode_meta_data_df[decode_meta_data_df['mean_resting_volt'] != 0.0] #deletes rows with no voltage data
		
mean_output_feature = decode_meta_data_df['mean_resting_volt'].to_numpy()
bottom_percentile = np.percentile(mean_output_feature, 10)
top_percentile = np.percentile(mean_output_feature, 90)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(decode_meta_data_df['0_embed_pos'], decode_meta_data_df['1_embed_pos'], decode_meta_data_df['2_embed_pos'],
			 c=mean_output_feature, vmin = bottom_percentile,
			vmax = top_percentile,
		    cmap="viridis",
		    edgecolor="none",
			marker='o', alpha = 0.1)
plt.show()

for i in range(num_clusters):
	cluster = f'just_my_type_{i}'
	subset_df = decode_meta_data_df[decode_meta_data_df[cluster] == 1]
	mean_output_feature = subset_df['mean_resting_volt'].to_numpy()
	bottom_percentile = np.percentile(mean_output_feature, 10)
	top_percentile = np.percentile(mean_output_feature, 90)
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter3D(subset_df['0_embed_pos'], subset_df['1_embed_pos'], subset_df['2_embed_pos'],
				 c=mean_output_feature, vmin = bottom_percentile,
				vmax = top_percentile,
			    cmap="viridis",
			    edgecolor="none",
				marker='o', alpha = 0.7)
	plt.show()
	
decode_meta_data_df.to_csv(os.path.join(save_path, 'umap_pos.csv'))