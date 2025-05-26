import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from sklearn.cluster import KMeans

def norm_row(array):
	min_by_row = np.min(array, axis = 1) 
	min_by_row = min_by_row.reshape(-1,1)
	array = np.subtract(array, min_by_row)
	max_by_row = np.max(array, axis = 1)
	max_by_row = max_by_row.reshape(-1,1)
	return np.divide(array, max_by_row)

save_path = r"F:\Big_MET_data"
meta_data_path = r"F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv"
counts_path = r"F:\Big_MET_data\trans_data_cpm\20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
morph_path = "F:\Big_MET_data\single_pulses_only"
ephys_path = r"F:\Big_MET_data\fine_and_dandi_ephys\000020"
input_data_path = 'F:\Big_MET_data\double_metadata_plus_trans.csv'

counts_df = pd.read_csv(counts_path)
col_names = list(counts_df) #these are like the names of the cells or the experiment or whatever
gene_names = list(counts_df[col_names[0]])
epic_gamer_data = pd.read_csv(input_data_path)

num_samples = epic_gamer_data.shape[0]
sum_binary_genes = np.sum(epic_gamer_data[gene_names].to_numpy() > 0, axis = 0)/num_samples
plt.hist(sum_binary_genes)
plt.ylabel('counts')
plt.xlabel('fraction of cells gene is expressed')
plt.show()
things_we_keep = (sum_binary_genes > 0.3)*(sum_binary_genes < 0.7)
approved_genes_idxs = np.where(things_we_keep == 1)[0]
print(len(approved_genes_idxs))

gene_names = np.asarray(gene_names) #??? ok so i don't have to do anything because this works but i found out you can add todo list items (kind of) with these 3 question marks!
gene_names = gene_names[approved_genes_idxs]

row_normed_gene_data = norm_row(epic_gamer_data[gene_names].to_numpy())
row_normed_gene_data = row_normed_gene_data > 0

num_cells_in_subset = 1500
num_genes_in_subset = 300
cells_to_check = np.random.randint(0, num_samples, num_cells_in_subset)
k = 5  # number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)

gene_subset_list = np.random.randint(0, len(approved_genes_idxs), num_genes_in_subset)
double_subset = row_normed_gene_data[np.ix_(cells_to_check, gene_subset_list)]
plt.imshow(double_subset, aspect = 'auto')
plt.show()

first_labels = kmeans.fit_predict(double_subset)
sorted_indices = np.argsort(first_labels)

plt.imshow(double_subset[sorted_indices], aspect = 'auto')
plt.xlabel('genes')
plt.ylabel('subset of cells')
plt.show()

gene_subset_list = np.random.randint(0, len(approved_genes_idxs), num_genes_in_subset)
double_subset = row_normed_gene_data[np.ix_(cells_to_check, gene_subset_list)]
plt.imshow(double_subset, aspect = 'auto')
plt.show()

second_labels = kmeans.fit_predict(double_subset)
sorted_indices = np.argsort(second_labels)

plt.imshow(double_subset[sorted_indices], aspect = 'auto')
plt.xlabel('genes')
plt.ylabel('subset of cells')
plt.show()

overlap_img = np.zeros((k, k))
for i in range(num_cells_in_subset):
	overlap_img[first_labels[i], second_labels[i]] += 1

plt.imshow(overlap_img)
plt.ylabel('first set of genes cluster index')
plt.xlabel('seecond set of genes cluster index')
plt.show()