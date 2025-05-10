import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import __version__ as sklearn_version
from packaging import version
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import joblib
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import RidgeCV, LassoCV

test_size = 0.2
batch_size = 64

input_current = 0.15 #we only want to look at one current input level so the model doesn't have to learn dependence on that in addition to the morphology and trans stuff
output_feature = 'steady_state_voltage_stimend' #only predicting one output feature at a time, some things like time to first spike, are not always going to have a meaningful value for every experiment so it's just easiest to do sperate models for each feature
approved_input_features = ['num_branches', 'num_nodes', 'num_tips', 'total_length', 'total_surface_area', 'total_volume', 'A230073K19Rik', 'Acsl6', 'Adipor1', 'Agpat4', 'BC052040', 'Baz2a', 'Brinp1', 'Cask', 'Ccnl1', 'Celf4', 'Celf6', 'Cib2', 'Cnot10', 'Cnot6', 'Copg1', 'Cwc27', 'Erbb2ip', 'Erc1', 'Fam13b', 'Fam81a', 'Gatsl2', 'Gigyf1', 'Gpr89', 'Hypk', 'Ip6k2', 'Kcna3', 'Kctd1', 'Kmt2a', 'Kmt2e', 'Ldlrad4', 'Map4k4', 'Mif', 'Mpc1', 'Mtdh', 'Ndufaf4', 'Nkain3', 'Nup93', 'Pfkl', 'Pmvk', 'Ppp1r12c', 'Ppp2r4', 'Ppp4r4', 'Psma1', 'Psmc5', 'Rpl17', 'Runx1t1', 'Rusc1', 'Safb2', 'Scamp5', 'Sdccag3', 'Sirt5', 'Slain1', 'Slc22a23', 'Slc38a6', 'St3gal5', 'Tbc1d10b', 'Tsn', 'Ube2g1', 'Ube2i', 'Uhrf1bp1l', 'Vdac3', 'Xrn1', 'Ywhae', 'Zc3h12b', 'Zfp281'] #determined to have relatively high importance via mutual information 
output_features_to_ignore_zeros = [
	'time_to_first_spike', 'time_to_last_spike',
	'AP_height', 'AP_width',
	'sag_amplitude', 'sag_ratio1', 'sag_time_constant']

metadata_path = 'F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv'
input_data_path = 'F:\Big_MET_data\combo_morph_trans.csv'
output_data_dir = 'F:\Big_MET_data\derived_ephys'

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

def create_dataloaders(X, y, test_size, batch_size, seed=42):
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=seed
	)

	train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
								  torch.tensor(y_train, dtype=torch.float32))
	test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
								 torch.tensor(y_test, dtype=torch.float32))

	g = torch.Generator()
	g.manual_seed(seed)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, test_loader


mean_output_feature = []
valid_inputs = []
morpho_trans_data = pd.read_csv(input_data_path)
for data_idx in range(len(morpho_trans_data['cell_id'].to_list())):
#for data_idx in range(20):
	#print(morpho_trans_data['ephys_path'].loc[data_idx], morpho_trans_data['cell_id'].loc[data_idx]) 
	
	cell_id = morpho_trans_data['cell_id'].loc[data_idx]
	
	try:
		this_cell_ephys = pd.read_csv(f'{output_data_dir}\{cell_id}.csv')
	except:
		pass
		#print(f'{cell_id} data not found!')

	this_current_ephys = this_cell_ephys[np.isclose(this_cell_ephys['current_second_edge'], input_current, atol = 0.0001)]
	
	#we want to ignore things with value 0 for certain features
	#like we don't want to inclde time to first spike when there were no spikes, ya dig?
	if output_feature in output_features_to_ignore_zeros:
		output_features = this_current_ephys[this_current_ephys[output_feature] != 0][output_feature].to_numpy()
	else:
		output_features = this_current_ephys[output_feature].to_numpy()
		
	if len(output_features) > 0:
		mean_output_feature.append(np.mean(output_features))
		this_input_data = morpho_trans_data.loc[data_idx, approved_input_features].to_numpy().astype(np.float64)
		valid_inputs.append(list(this_input_data))
	
mean_output_feature = np.asarray(mean_output_feature)		
valid_inputs = np.asarray(valid_inputs)

#we normalize morphology based on all cells, gene expression within a cell
morph_end = 6 
valid_inputs[:, :morph_end] = norm_col(valid_inputs[:, :morph_end])
valid_inputs[:, morph_end:] = norm_row(valid_inputs[:, morph_end:])
mean_output_feature = norm_col(mean_output_feature)

X_train, X_test, y_train, y_test = train_test_split(valid_inputs, mean_output_feature, test_size=0.2, random_state=42)

ridge = Ridge(alpha=1.0)  # alpha is the regularization strength
ridge.fit(X_train, y_train)

ridge_preds = ridge.predict(X_test)

print("Ridge R²:", r2_score(y_test, ridge_preds))
print("Ridge MSE:", mean_squared_error(y_test, ridge_preds))

lasso = Lasso(alpha=0.01)  # alpha can be tuned; too high may zero-out too many weights
lasso.fit(X_train, y_train)

lasso_preds = lasso.predict(X_test)

print("Lasso R²:", r2_score(y_test, lasso_preds))
print("Lasso MSE:", mean_squared_error(y_test, lasso_preds))

alphas = np.logspace(-4, 2, 100)

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)
print("Best Ridge alpha:", ridge_cv.alpha_)

lasso_cv = LassoCV(alphas=alphas, cv=5)
lasso_cv.fit(X_train, y_train)
print("Best Lasso alpha:", lasso_cv.alpha_)


plt.scatter(ridge_preds, y_test, label='Ridge', alpha=0.5)
plt.scatter(lasso_preds, y_test, label='Lasso', alpha=0.5)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Ridge and Lasso Predictions')
plt.legend()
plt.show()
