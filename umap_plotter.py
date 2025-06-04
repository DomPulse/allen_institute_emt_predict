import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import umap
import seaborn as sns

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

gene_data = pd.read_csv(counts_path, index_col=0)
metadata = pd.read_csv(meta_data_path)
gene_names = list(gene_data.index)
mask = np.isin(gene_names, ion_channel_genes)
indexes = np.where(mask)[0]
print(indexes)

marker_genes_for_umap = gene_names#pd.read_csv("select_markers.csv", index_col=0)

embedding = umap.UMAP(n_components=3, n_neighbors=25).fit_transform(
    np.log2(gene_data.loc[ion_channel_genes].values.T + 1)
)
my_ttype_metadata = metadata.loc[metadata["T-type Label"] == "Lamp5 Plch2 Dock5", :]
my_ttype_mask = gene_data.columns.isin(my_ttype_metadata["transcriptomics_sample_id"].tolist())

'''
plt.figure(figsize=(8, 8))
plt.scatter(*embedding.T, s=1, edgecolor="none")
plt.scatter(*embedding[my_ttype_mask, :].T, s=2, edgecolor="none")
sns.despine()
'''

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')
ax.scatter3D(*embedding.T, s=1, edgecolor="none")
ax.scatter3D(*embedding[my_ttype_mask, :].T, s=2, edgecolor="none")
sns.despine()
plt.show()


input_current = 0.15 #we only want to look at one current input level so the model doesn't have to learn dependence on that in addition to the morphology and trans stuff
output_feature = 'steady_state_voltage_stimend' #only predicting one output feature at a time, some things like time to first spike, are not always going to have a meaningful value for every experiment so it's just easiest to do sperate models for each feature
approved_input_features = ['hemisphere_right', 'biological_sex_M', 'structure_VISl2/3', 'structure_VISl5', 'structure_VISl6a', 'structure_VISp1', 'structure_VISp2/3', 'structure_VISp4', 'structure_VISp5', 'structure_VISp6a', 'structure_VISpm2/3', 'structure_VISpm5', 'structure_VISpm6a', 'T-type Label_Lamp5', 'T-type Label_Pvalb', 'T-type Label_Serpinf1', 'T-type Label_Sncg', 'T-type Label_Sst', 'T-type Label_Vip', 'cluster_0_0', 'cluster_0_1', 'cluster_0_2', 'cluster_1_0', 'cluster_1_1', 'cluster_1_2', 'cluster_2_0', 'cluster_2_1', 'cluster_2_2', 'cluster_3_0', 'cluster_3_1', 'cluster_3_2', 'cluster_4_0', 'cluster_4_1', 'cluster_4_2', 'cluster_5_0', 'cluster_5_1', 'cluster_5_2', 'cluster_6_0', 'cluster_6_1', 'cluster_6_2', 'cluster_7_0', 'cluster_7_1', 'cluster_7_2', 'cluster_8_0', 'cluster_8_1', 'cluster_8_2', 'cluster_9_0', 'cluster_9_1', 'cluster_9_2', 'cluster_10_0', 'cluster_10_1', 'cluster_10_2', 'cluster_11_0', 'cluster_11_1', 'cluster_11_2', 'cluster_12_0', 'cluster_12_1', 'cluster_12_2', 'cluster_13_0', 'cluster_13_1', 'cluster_13_2', 'cluster_14_0', 'cluster_14_1', 'cluster_14_2', 'cluster_15_0', 'cluster_15_1', 'cluster_15_2', 'cluster_16_0', 'cluster_16_1', 'cluster_16_2', 'cluster_17_0', 'cluster_17_1', 'cluster_17_2', 'cluster_18_0', 'cluster_18_1', 'cluster_18_2', 'cluster_19_0', 'cluster_19_1', 'cluster_19_2', 'cluster_20_0', 'cluster_20_1', 'cluster_20_2', 'cluster_21_0', 'cluster_21_1', 'cluster_21_2', 'cluster_22_0', 'cluster_22_1', 'cluster_22_2', 'cluster_23_0', 'cluster_23_1', 'cluster_23_2', 'cluster_24_0', 'cluster_24_1', 'cluster_24_2']

print(len(approved_input_features))
output_features_to_ignore_zeros = [
	'time_to_first_spike', 'time_to_last_spike',
	'AP_height', 'AP_width',
	'sag_amplitude', 'sag_ratio1', 'sag_time_constant']
metadata_path = 'F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv'
input_data_path = 'F:\Big_MET_data\\double_meta_plus_clusters.csv'
output_data_dir = 'F:\Big_MET_data\derived_ephys'
mean_output_feature = []
valid_inputs = []
just_trans_data = pd.read_csv(input_data_path, skip_blank_lines=True)
for data_idx in range(len(just_trans_data['cell_specimen_id'].to_list())):
#for data_idx in range(50):
	#print(just_trans_data['ephys_path'].loc[data_idx], just_trans_data['cell_id'].loc[data_idx]) 
	
	try:
		cell_id = int(just_trans_data['cell_specimen_id'].loc[data_idx])
		
		this_cell_ephys = pd.read_csv(f'{output_data_dir}\{cell_id}.csv')
	

		this_current_ephys = this_cell_ephys[np.isclose(this_cell_ephys['current_second_edge'], input_current, atol = 0.0001)]
		
		#we want to ignore things with value 0 for certain features
		#like we don't want to inclde time to first spike when there were no spikes, ya dig?
		output_features = this_current_ephys[this_current_ephys[output_feature] != 0].to_numpy()
		
		if len(output_features) > 0:
			mean_output_feature.append(output_features[0])
			this_input_data = just_trans_data.loc[data_idx, approved_input_features].to_numpy().astype(np.float64)
			valid_inputs.append(list(this_input_data))
	except:
		pass
		#print(f'{cell_id} data not found!')
	
mean_output_feature = np.asarray(mean_output_feature)		
valid_inputs = np.asarray(valid_inputs)

mean_output_feature = norm_col(mean_output_feature)

embedding = umap.UMAP(n_components=3, n_neighbors=25).fit_transform(mean_output_feature)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')
ax.scatter3D(*embedding.T, s=1, edgecolor="none")
sns.despine()
plt.show()