import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc
from sklearn.decomposition import PCA

counts_path = r"F:\Big_MET_data\trans_data_cpm\20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
just_trans_path = 'F:\Big_MET_data\metadata_pluse_trans.csv'
mopho_trans_path = 'F:\Big_MET_data\combo_morph_trans.csv'
ephys_path = r'F:\Big_MET_data\derived_ephys'
full_save_path = r'F:\Big_MET_data\feature_extraction'
morpho_trans_data = pd.read_csv(mopho_trans_path)
currents_to_check = [-0.11, -0.09, -0.07, -0.05, -0.03, 0.05, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25]

def norm_col(array):
	min_by_col = np.min(array, axis = 0) 
	array = np.subtract(array, min_by_col)
	max_by_col = np.max(array, axis = 0)
	return np.divide(array, max_by_col)

def norm_row(array):
	min_by_row = np.min(array, axis = 1) 
	min_by_row = min_by_row.reshape(-1,1)
	array = np.subtract(array, min_by_row)
	max_by_row = np.max(array, axis = 1)
	max_by_row = max_by_row.reshape(-1,1)
	return np.divide(array, max_by_row)

counts_df = pd.read_csv(counts_path)
col_names = list(counts_df)
gene_names = list(counts_df[col_names[0]]) #needed for labeling input features :)
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

del col_names, counts_df
gc.collect()

just_morph_trans = morpho_trans_data.iloc[:, 5:].to_numpy()
morph_end = len(morph_features)
just_morph_trans[:, :morph_end] = norm_col(just_morph_trans[:, :morph_end])
just_morph_trans[:, morph_end:] = norm_row(just_morph_trans[:, morph_end:])
morpho_trans_as_df = pd.DataFrame(just_morph_trans, columns = morph_features + gene_names)

num_components = 100
pca = PCA(n_components = num_components)

#fit PCA model to data
pca_fit = pca.fit(morpho_trans_as_df)

PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

print(pca.explained_variance_ratio_)

cumulative_explained = np.zeros(num_components)
for i in range(num_components):
	cumulative_explained[i] = np.sum(pca.explained_variance_ratio_[:i])
	print(i, cumulative_explained[i])

plt.plot(PC_values, cumulative_explained, 'o-', linewidth=2, color='blue')
plt.title('Cumulative Explanation')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.ylim(0, 1)
plt.show()

# Number of PCA components to retain
n = 50  # change as needed

# Transform the data using the fitted PCA model and keep first n components
projected_data = pca.transform(morpho_trans_as_df)[:, :n]

# Create a DataFrame with PCA components
pca_columns = [f'PC{i+1}' for i in range(n)]
pca_df = pd.DataFrame(projected_data, columns=pca_columns)

# Extract the first 5 metadata columns from the original morpho_trans_data
meta_columns = morpho_trans_data.iloc[:, 1:5]

# Concatenate metadata with PCA projection
final_df = pd.concat([meta_columns.reset_index(drop=True), pca_df], axis=1)

# Optional: Save or inspect
print(final_df.head())
final_df.to_csv('F:\Big_MET_data\PCAd_morph_trans.csv')

