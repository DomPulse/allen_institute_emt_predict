import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def norm_col(array):
	min_by_col = np.min(array, axis = 0) 
	array = np.subtract(array, min_by_col)
	max_by_col = np.max(array, axis = 0)
	return np.divide(array, max_by_col + 1E-8)

efeature_data_path = r'F:\Big_MET_data\conductance_fit_data\efeat_model_umap_embedding.csv'
cell_prop_data_path = r'F:\Big_MET_data\conductance_fit_data\allactive_params.csv'
morph_data_path = r'F:\Big_MET_data\conductance_fit_data\morph_data.csv'

efeature_data = pd.read_csv(efeature_data_path)
cell_prop_data = pd.read_csv(cell_prop_data_path)
morph_data = pd.read_csv(morph_data_path)

efeatures_of_interest = ['AHP_depth', 'AP_amplitude_from_voltagebase', 
						 'AP_width', 'mean_frequency', 
						 'time_to_first_spike', 'voltage_base']

cell_props_to_predict = ['Ra.all', 'cm.axonal', 'cm.basal', 
						 'cm.somatic', 'decay_CaDynamics.axonal', 'decay_CaDynamics.somatic',
						 'e_pas.all', 'g_pas.all', 'gamma_CaDynamics.axonal', 
						 'gamma_CaDynamics.somatic', 'gbar_Ca_HVA.axonal', 'gbar_Ca_HVA.somatic',
						 'gbar_Ca_LVA.axonal', 'gbar_Ca_LVA.somatic', 'gbar_Ih.apical',
						 'gbar_Ih.basal', 'gbar_Ih.somatic', 'gbar_Im.apical',
						 'gbar_Im.basal', 'gbar_Im_v2.basal', 'gbar_K_Pst.axonal',
						 'gbar_K_Pst.somatic', 'gbar_K_Tst.axonal', 'gbar_K_Tst.somatic',
						 'gbar_Kd.axonal', 'gbar_Kv2like.axonal', 'gbar_Kv3_1.apical', 
						 'gbar_Kv3_1.axonal', 'gbar_Kv3_1.basal', 'gbar_Kv3_1.somatic',
						 'gbar_NaTa_t.axonal', 'gbar_NaTs2_t.apical', 'gbar_NaTs2_t.basal',
						 'gbar_NaTs2_t.somatic', 'gbar_NaV.axonal', 'gbar_NaV.basal',
						 'gbar_NaV.somatic', 'gbar_Nap_Et2.axonal', 'gbar_Nap_Et2.somatic',
						 'gbar_SK.axonal', 'gbar_SK.somatic', 'cm.apical',]

morph_features = ['area.all', 'area.apical_dendrite', 'area.axon',
				  'area.basal_dendrite', 'length.all', 'length.apical_dendrite',
				  'length.axon', 'length.basal_dendrite', 'soma_radius',
				  'soma_suface', 'taper_rate.all', 'taper_rate.apical_dendrite',
				  'taper_rate.axon', 'taper_rate.basal_dendrite', 'volume.all',
				  'volume.apical_dendrite', 'volume.axon', 'volume.basal_dendrite']

for cell_prop in cell_props_to_predict:
	subset_cell_prop_df = cell_prop_data[(~cell_prop_data[cell_prop].isna())*(cell_prop_data['hof_index'] == 0)][[cell_prop] + ['Cell_id']]
	subset_efeature_data_df = efeature_data[efeature_data['hof_index'] == 0][efeatures_of_interest + ['Cell_id']]
	aligned_data_df = pd.merge(subset_efeature_data_df, subset_cell_prop_df, on='Cell_id')
	aligned_data_df = pd.merge(morph_data, aligned_data_df)
	
	aligned_efeat_and_morph_data = (aligned_data_df[efeatures_of_interest + morph_features]
	.dropna(axis=1, how="any")   # drop columns where any values are NaN
	.to_numpy()
	)

	aligned_cell_prop_data = aligned_data_df[cell_prop].to_numpy()
	
	accepted_data_points = np.where(aligned_cell_prop_data < np.percentile(aligned_cell_prop_data, 90))[0]
	
	aligned_efeat_and_morph_data = norm_col(aligned_efeat_and_morph_data[accepted_data_points])
	aligned_cell_prop_data = norm_col(aligned_cell_prop_data[accepted_data_points])
		
	X_train, X_test, y_train, y_test = train_test_split(aligned_efeat_and_morph_data, aligned_cell_prop_data, test_size=0.2, random_state=1)
	neigh = KNeighborsRegressor(n_neighbors=20, weights = 'distance')
	clf = Ridge(alpha=1.0)
	clf.fit(X_train, y_train)
	
	predicted_cell_prop = clf.predict((X_test))
	plt.scatter(y_test, predicted_cell_prop)
	test_r2 = r2_score(y_test, predicted_cell_prop)
	plt.xlabel('true value')
	plt.ylabel('predicted value')
	plt.title(f'{cell_prop} r squared: {test_r2:.2f}')
	plt.ylim(0, 1)
	plt.xlim(0, 1)
	plt.show()
	