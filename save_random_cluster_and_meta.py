import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc
from sklearn.cluster import KMeans

def make_random_clusters(gene_expression_mat, num_clusters = 5, num_genes_in_subset = 300):
	kmeans = KMeans(n_clusters = num_clusters)
	gene_subset_list = np.random.randint(0, len(gene_expression_mat[0]), num_genes_in_subset)
	double_subset = gene_expression_mat[:, gene_subset_list]
	labels = kmeans.fit_predict(double_subset)
	return labels

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

#thanks chat <3
prefixes = ['Vip', 'Sst', 'Pvalb', 'Sncg', 'Serpinf1', 'Lamp5']
def map_to_prefix(cell):
	for prefix in prefixes:
		if str(cell).startswith(prefix):
			return prefix
	return 'Other'  # or np.nan if you prefer to ignore unknowns

def make_age_numerical(given_age):
	age = int(given_age.split('P')[1])
	return age

save_path = r"F:\Big_MET_data"
meta_data_path = r"F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv"
counts_path = r"F:\Big_MET_data\trans_data_cpm\20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
morph_path = "F:\Big_MET_data\single_pulses_only"
ephys_path = r"F:\Big_MET_data\fine_and_dandi_ephys\000020"
ephys_ses, ses_path_dict = list_all_files(ephys_path)
meta_data_df = pd.read_csv(meta_data_path)
counts_df = pd.read_csv(counts_path)

col_names = list(counts_df) #these are like the names of the cells or the experiment or whatever
valid_ephys_row = meta_data_df.index[(meta_data_df['transcriptomics_sample_id'].isin(col_names)) & (meta_data_df['ephys_session_id'].isin(ephys_ses))].tolist()
meta_data_df = meta_data_df.iloc[valid_ephys_row]
gene_names = list(counts_df[col_names[0]])
	
#just get some specific labels that will hopefully be predictive
interesting_meta = ['transcriptomics_sample_id', 'cell_specimen_id', 'ephys_session_id', 'hemisphere', 'structure', 'biological_sex', 'age', 'T-type Label']
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

#but the others do
#but we want to optionally filter the T-type first
decode_meta_data_df['T-type Label'] = decode_meta_data_df['T-type Label'].apply(map_to_prefix)

class_cal_cols = ['structure', 'T-type Label']
decode_meta_data_df = pd.get_dummies(decode_meta_data_df, columns = class_cal_cols)
decode_meta_data_df = decode_meta_data_df.drop('T-type Label_Other', axis=1) #deletes the other column
decode_meta_data_df.index = range(decode_meta_data_df.shape[0]) #renames the rows just numerically

#we don't like true and false, we want binary
to_be_int = ['hemisphere_right', 'biological_sex_M', 'structure_VISa5', 'structure_VISa6a', 'structure_VISal1', 'structure_VISal2/3', 'structure_VISal4', 'structure_VISal5', 'structure_VISal6a', 'structure_VISam2/3', 'structure_VISam4', 'structure_VISam5', 'structure_VISam6a', 'structure_VISl1', 'structure_VISl2/3', 'structure_VISl4', 'structure_VISl5', 'structure_VISl6a', 'structure_VISl6b', 'structure_VISli1', 'structure_VISli2/3', 'structure_VISli4', 'structure_VISli5', 'structure_VISli6a', 'structure_VISli6b', 'structure_VISp', 'structure_VISp1', 'structure_VISp2/3', 'structure_VISp4', 'structure_VISp5', 'structure_VISp6a', 'structure_VISp6b', 'structure_VISpl2/3', 'structure_VISpl4', 'structure_VISpl5', 'structure_VISpl6a', 'structure_VISpm1', 'structure_VISpm2/3', 'structure_VISpm4', 'structure_VISpm5', 'structure_VISpm6a', 'structure_VISpor1', 'structure_VISpor2/3', 'structure_VISpor4', 'structure_VISpor5', 'structure_VISpor6a', 'structure_VISpor6b', 'structure_VISrl2/3', 'structure_VISrl4', 'structure_VISrl5', 'structure_VISrl6a', 'T-type Label_Lamp5', 'T-type Label_Pvalb', 'T-type Label_Serpinf1', 'T-type Label_Sncg', 'T-type Label_Sst', 'T-type Label_Vip']
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

transcriptomics_sample_ids = decode_meta_data_df['transcriptomics_sample_id'].tolist()
#print(transcriptomics_sample_ids)
counts_for_our_cells = counts_df[transcriptomics_sample_ids].T.to_numpy() > 0
glob_num_clusters = 3
for i in range(15):
	col_name = f'cluster_{i}'
	decode_meta_data_df[col_name] = make_random_clusters(counts_for_our_cells, glob_num_clusters)
	decode_meta_data_df = pd.get_dummies(decode_meta_data_df, columns = [col_name])
	for j in range(glob_num_clusters):
		new_col_name = f'{col_name}_{j}'
		decode_meta_data_df[new_col_name] = decode_meta_data_df[new_col_name].astype(int)

#actually deletes given columns and rows
for col in cols_to_delete:
	decode_meta_data_df = decode_meta_data_df[decode_meta_data_df[col] != 1]
	decode_meta_data_df = decode_meta_data_df.drop(col, axis=1)

print(decode_meta_data_df.columns)
decode_meta_data_df.to_csv(os.path.join(save_path, 'double_meta_plus_clusters.csv'))
