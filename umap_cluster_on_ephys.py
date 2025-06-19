import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as col_map
import pandas as pd
import umap
import seaborn as sns
from sklearn.cluster import KMeans
import os
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from sklearn.metrics import confusion_matrix

def get_n_colors(n):

	colors = col_map.get_cmap('hsv', n+1)
	return [colors(i)[:3] for i in range(1, n+1)]  # Remove alpha

neg_current = -0.03
neg_current_feat = ['steady_state_voltage', 'steady_state_voltage_stimend', 'sag_amplitude', 'sag_ratio1', 'sag_time_constant']
pos_current = 0.12
pos_current_feat = ['steady_state_voltage', 'steady_state_voltage_stimend', 'time_to_first_spike', 'time_to_last_spike', 'spike_count', 'AP_height', 'AP_width']
col_from_meta = ['transcriptomics_sample_id','cell_specimen_id', 'ephys_session_id', 'ephys_path', 'just_my_type_original']
cols_for_embed = []
for col in neg_current_feat:
	neg_col_name = f'{neg_current}_{col}'
	cols_for_embed.append(neg_col_name)

for col in pos_current_feat:
	pos_col_name = f'{pos_current}_{col}'
	cols_for_embed.append(pos_col_name)

metadata_path = r'F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv'
input_data_path = r'F:\Big_MET_data\umap_pos.csv'
output_data_dir = r'F:\Big_MET_data\derived_ephys'

wombo_combo = defaultdict(list)

input_meta = pd.read_csv(input_data_path, skip_blank_lines=True)
for data_idx in range(len(input_meta['cell_specimen_id'].to_list())):
#for data_idx in range(50):
	#print(input_meta['ephys_path'].loc[data_idx], just_trans_data['cell_id'].loc[data_idx]) 
	
	try:
		cell_id = int(input_meta['cell_specimen_id'].loc[data_idx])
	
		this_cell_ephys = pd.read_csv(f'{output_data_dir}\{cell_id}.csv')
		
		has_pos_curr = np.isclose(this_cell_ephys['current_second_edge'], pos_current, atol = 0.0001)
		has_neg_curr = np.isclose(this_cell_ephys['current_second_edge'], neg_current, atol = 0.0001)
		right_duration = np.isclose(this_cell_ephys['end_time'], 2759, atol = 30) 
		#we want to make sure that the guys we're using were all stimmed about the same time so that we can meaningfully use the #of spikes and what not
		#that said, ignoring this does not really change the conclusion that this sucks
		pos_index = np.where(has_pos_curr)[0]
		neg_index = np.where(has_neg_curr)[0]
		right_duration_idx = np.where(right_duration)[0]
		
		#yes this is stupid, I don't care
		pos_idx_we_take = 0
		neg_idx_we_take = 0
		found_pos = False
		found_neg = False
		for i in range(100):
			if i in pos_index and i in right_duration_idx:
				pos_idx_we_take = i
				found_pos = True
			if i in neg_index and i in right_duration_idx:
				neg_idx_we_take = i
				found_neg = True
		
		if found_neg and found_pos:
		
			for col in col_from_meta:
				wombo_combo[col].append(input_meta[col].loc[data_idx])
				
			for col in neg_current_feat:
				neg_col_name = f'{neg_current}_{col}'
				wombo_combo[neg_col_name].append(this_cell_ephys[col].loc[neg_idx_we_take])
			
			for col in pos_current_feat:
				pos_col_name = f'{pos_current}_{col}'
				wombo_combo[pos_col_name].append(this_cell_ephys[col].loc[pos_idx_we_take])

	except:
		pass

wombinus_combinus = pd.DataFrame.from_dict(wombo_combo)

num_clusters = 3
embed_dim = 3
ephys_umap_pos = []
for i in range(embed_dim):
	ephys_umap_pos.append(f'ephys_umap_pos_{i}')

wombinus_combinus[ephys_umap_pos + ['just_my_type_ephys']] = 0
	
embedding = umap.UMAP(n_components=embed_dim, n_neighbors=25).fit_transform(wombinus_combinus[cols_for_embed].to_numpy())

embedding = np.asarray(embedding)

for i in range(embed_dim):
	wombinus_combinus[f'ephys_umap_pos_{i}'] = embedding[:, i]

kmeans = KMeans(n_clusters = num_clusters)
labels = kmeans.fit_predict(embedding)

wombinus_combinus['just_my_type_ephys'] = labels
wombinus_combinus['r'] = 0
wombinus_combinus['g'] = 0
wombinus_combinus['b'] = 0
color_options = get_n_colors(num_clusters)

for idx, i in enumerate(labels):
	wombinus_combinus['r'].iloc[idx] = list(color_options[i])[0]
	wombinus_combinus['g'].iloc[idx] = list(color_options[i])[1]
	wombinus_combinus['b'].iloc[idx] = list(color_options[i])[2]

r = wombinus_combinus['r'].to_numpy()
g = wombinus_combinus['g'].to_numpy()
b = wombinus_combinus['b'].to_numpy()
colors = np.stack((r, g, b), axis=1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(embedding[:, 0], embedding[:, 1], embedding[:, 2], color = colors, marker='o', alpha = 0.1)
plt.show()

for idx, i in enumerate(list(wombinus_combinus['just_my_type_original'])):
	wombinus_combinus['r'].iloc[idx] = list(color_options[i])[0]
	wombinus_combinus['g'].iloc[idx] = list(color_options[i])[1]
	wombinus_combinus['b'].iloc[idx] = list(color_options[i])[2]

r = wombinus_combinus['r'].to_numpy()
g = wombinus_combinus['g'].to_numpy()
b = wombinus_combinus['b'].to_numpy()
colors = np.stack((r, g, b), axis=1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(embedding[:, 0], embedding[:, 1], embedding[:, 2], color = colors, marker='o', alpha = 0.1)
plt.show()

gene_type = wombinus_combinus['just_my_type_original'].to_numpy()
ephys_type = wombinus_combinus['just_my_type_ephys'].to_numpy()

cm = confusion_matrix(ephys_type, gene_type, normalize='pred')

plt.imshow(cm)
plt.xlabel('ephys type')
plt.ylabel('gene type')
plt.show()