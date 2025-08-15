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
from sklearn.model_selection import train_test_split

meta_data_path = r'F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv'
counts_path = r'F:\Big_MET_data\trans_data_cpm\20200513_Mouse_PatchSeq_Release_cpm.v2.csv'
raw_ephys_path = r'F:\Big_MET_data\fine_and_dandi_ephys\000020'
derived_ephys_path = 'F:\Big_MET_data\derived_ephys'


def norm_col(array):
	min_by_col = np.min(array, axis = 0) 
	array = np.subtract(array, min_by_col)
	max_by_col = np.max(array, axis = 0)
	return np.divide(array, max_by_col + 1E-8)

def get_n_colors(n):

	colors = col_map.get_cmap('hsv', n+1)
	return np.asarray([colors(i)[:3] for i in range(1, n+1)])  # Remove alpha

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
				session_to_path[int(pre_other_stuff)] = f'{dirpath}\\{filename}'
			except:
				pass
	return session_search, session_to_path

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

ephys_ses, ses_path_dict = list_all_files(raw_ephys_path)
metadata = pd.read_csv(meta_data_path)
gene_data = pd.read_csv(counts_path, index_col=0)
gene_names = list(gene_data.index)
gene_data_numpy = gene_data.to_numpy()
gene_std = np.std(gene_data_numpy, axis = 1)#, where = gene_data_numpy > 0)
num_genes_evaled = 1000
thresh = np.sort(gene_std)[-num_genes_evaled]
genes_over_thresh_bin = gene_std > thresh
genes_over_thresh = np.where(genes_over_thresh_bin)[0]
gene_names = list(gene_data.index)
high_variance_genes = list(np.asarray(gene_names)[genes_over_thresh])

#adds the resting potential of each neuron to the gene expression data frame
gene_data.loc['resting_potential'] = np.zeros(len(gene_data.columns)) #want a row in gene data that is now the resiting potental, I know its odd but hey
cells_with_ephys_idxs = []
for data_idx in range(len(metadata['cell_specimen_id'].to_list())):
#for data_idx in range(50):
		
	try:
		cell_id = int(metadata['cell_specimen_id'].loc[data_idx])
		transcriptomics_sample_id = metadata['transcriptomics_sample_id'].loc[data_idx]
		
		this_cell_ephys = pd.read_csv(f'{derived_ephys_path}\{cell_id}.csv')
		gene_data.loc['resting_potential', transcriptomics_sample_id] = np.mean(this_cell_ephys['steady_state_voltage'].to_numpy())
		cells_with_ephys_idxs.append(data_idx)
		
	except:
		pass
		#print(f'{cell_id} data not found!')
cells_with_ephys_idxs = np.asarray(cells_with_ephys_idxs)

#embeds and clusters neurons
num_clusters = 60
embedding_dim = 10
embedding = umap.UMAP(n_components=embedding_dim, n_neighbors=25, random_state=42).fit_transform(
    np.log2(gene_data.loc[high_variance_genes].values.T + 1)
)

kmeans = KMeans(n_clusters = num_clusters)
labels = kmeans.fit_predict(embedding)
color_options = get_n_colors(num_clusters)
colors = color_options[labels]
plt.scatter(embedding[:, 0], embedding[:, 1], color = colors, marker='o', alpha = 0.2)
plt.show()

#plots histograms of each cell type's gene expression

median_gene_express_cluster = []
median_ephys_feature = []
ephys_feature_idx = list(gene_data.index).index('resting_potential')

for cell_type in range(num_clusters):
	
	this_cell_type_idxs = np.where(labels == cell_type)[0]
	
	all_gene_vector = []
	
	for gene_of_interest in high_variance_genes:

		ion_channel_gene_idx = gene_names.index(gene_of_interest)
		gene_expression_of_cluster = np.median(np.log2(gene_data.iloc[ion_channel_gene_idx, this_cell_type_idxs].values + 1))
		all_gene_vector.append(gene_expression_of_cluster)
	
	median_gene_express_cluster.append(all_gene_vector)
	median_ephys_feature.append(np.median(gene_data.iloc[ephys_feature_idx, this_cell_type_idxs].values))

median_ephys_feature = np.asarray(median_ephys_feature)
median_gene_express_cluster = np.asarray(median_gene_express_cluster)
accepted_data_points = np.where(median_ephys_feature < np.percentile(median_ephys_feature, 90))[0]
median_ephys_feature = median_ephys_feature[accepted_data_points]
median_gene_express_cluster = median_gene_express_cluster[accepted_data_points]
median_ephys_feature = norm_col(median_ephys_feature)

#train test split in a weird way

num_repeats = 100
mean_r2 = 0
for i in range(num_repeats):
	X_train, X_test, y_train, y_test = train_test_split(median_gene_express_cluster, median_ephys_feature, test_size=0.5, random_state=i)
	
	clf = Ridge(alpha=1.0)
	neigh = KNeighborsRegressor(n_neighbors=20, weights = 'distance')
	clf.fit(X_train, y_train)
	
	predicted_AP_width = clf.predict((X_test))
	plt.scatter(y_test, predicted_AP_width)
	test_r2 = r2_score(y_test, predicted_AP_width)
	mean_r2 += test_r2/num_repeats
	plt.xlabel('true resting potential')
	plt.ylabel('predicted resting potential')
	plt.ylim(0, 1)
	plt.xlim(0, 1)
	plt.show()
	
print(f'r2: {mean_r2}')



'''
for gene_of_interest in ion_channel_genes + high_variance_genes:
	
	gene_of_interest_expression = np.log2(gene_data.loc[gene_of_interest, train_cell_specimen_ids].to_numpy() + 1)
	
	plt.title(gene_of_interest)
	plt.scatter(gene_of_interest_expression, resting_potential)
	plt.show()
'''
