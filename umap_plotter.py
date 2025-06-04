import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import umap
import seaborn as sns
from sklearn.cluster import KMeans

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

counts_path = r"F:\Big_MET_data\trans_data_cpm\20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
meta_data_path = r"F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv"

num_clusters = 10
embed_dim = 3

gene_data = pd.read_csv(counts_path, index_col=0)
metadata = pd.read_csv(meta_data_path)
gene_names = list(gene_data.index)
mask = np.isin(gene_names, ion_channel_genes)
indexes = np.where(mask)[0]
print(indexes)

marker_genes_for_umap = gene_names#pd.read_csv("select_markers.csv", index_col=0)

embedding = umap.UMAP(n_components=embed_dim, n_neighbors=25).fit_transform(
    np.log2(gene_data.loc[ion_channel_genes].values.T + 1)
)

embedding = np.asarray(embedding)

kmeans = KMeans(n_clusters = num_clusters)
labels = kmeans.fit_predict(embedding)
color_options = get_n_colors(num_clusters)

colors = []
centers = np.zeros((num_clusters, embed_dim))
pos_relative = np.zeros((embedding.shape[0], embed_dim))
for j in range(num_clusters):
	this_clust_idxs = np.where(labels == j)[0]
	centers[j] = np.mean(embedding[this_clust_idxs], axis = 0)

for idx, i in enumerate(labels):
	pos_relative[idx] = embedding[idx] - centers[i]
	colors.append(list(color_options[i]))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')
ax.scatter3D(*embedding.T, s=1, edgecolor="none", color = colors, alpha = 0.5)
ax.scatter3D(*centers.T, s=10, edgecolor="none", color = 'black')
sns.despine()
plt.show()

