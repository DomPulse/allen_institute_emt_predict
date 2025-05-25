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
from sklearn.metrics import r2_score, mean_squared_error

test_size = 0.2
batch_size = 32

input_current = 0.15 #we only want to look at one current input level so the model doesn't have to learn dependence on that in addition to the morphology and trans stuff
output_feature = 'steady_state_voltage_stimend' #only predicting one output feature at a time, some things like time to first spike, are not always going to have a meaningful value for every experiment so it's just easiest to do sperate models for each feature
#output_feature = 'sag_time_constant'
#approved_input_features = ['num_branches', 'num_nodes', 'num_tips', 'total_length', 'total_surface_area', 'total_volume', 'A230073K19Rik', 'Acsl6', 'Adipor1', 'Agpat4', 'BC052040', 'Baz2a', 'Brinp1', 'Cask', 'Ccnl1', 'Celf4', 'Celf6', 'Cib2', 'Cnot10', 'Cnot6', 'Copg1', 'Cwc27', 'Erbb2ip', 'Erc1', 'Fam13b', 'Fam81a', 'Gatsl2', 'Gigyf1', 'Gpr89', 'Hypk', 'Ip6k2', 'Kcna3', 'Kctd1', 'Kmt2a', 'Kmt2e', 'Ldlrad4', 'Map4k4', 'Mif', 'Mpc1', 'Mtdh', 'Ndufaf4', 'Nkain3', 'Nup93', 'Pfkl', 'Pmvk', 'Ppp1r12c', 'Ppp2r4', 'Ppp4r4', 'Psma1', 'Psmc5', 'Rpl17', 'Runx1t1', 'Rusc1', 'Safb2', 'Scamp5', 'Sdccag3', 'Sirt5', 'Slain1', 'Slc22a23', 'Slc38a6', 'St3gal5', 'Tbc1d10b', 'Tsn', 'Ube2g1', 'Ube2i', 'Uhrf1bp1l', 'Vdac3', 'Xrn1', 'Ywhae', 'Zc3h12b', 'Zfp281'] #determined to have relatively high importance via mutual information 
always_given_features = ['age', 'hemisphere_right', 'biological_sex_M', 'structure_VISa5', 'structure_VISa6a', 'structure_VISal1', 'structure_VISal2/3', 'structure_VISal4', 'structure_VISal5', 'structure_VISal6a', 'structure_VISam2/3', 'structure_VISam4', 'structure_VISam5', 'structure_VISam6a', 'structure_VISl1', 'structure_VISl2/3', 'structure_VISl4', 'structure_VISl5', 'structure_VISl6a', 'structure_VISl6b', 'structure_VISli1', 'structure_VISli2/3', 'structure_VISli4', 'structure_VISli5', 'structure_VISli6a', 'structure_VISli6b', 'structure_VISp', 'structure_VISp1', 'structure_VISp2/3', 'structure_VISp4', 'structure_VISp5', 'structure_VISp6a', 'structure_VISp6b', 'structure_VISpl2/3', 'structure_VISpl4', 'structure_VISpl5', 'structure_VISpl6a', 'structure_VISpm1', 'structure_VISpm2/3', 'structure_VISpm4', 'structure_VISpm5', 'structure_VISpm6a', 'structure_VISpor1', 'structure_VISpor2/3', 'structure_VISpor4', 'structure_VISpor5', 'structure_VISpor6a', 'structure_VISpor6b', 'structure_VISrl2/3', 'structure_VISrl4', 'structure_VISrl5', 'structure_VISrl6a', 'T-type Label_Lamp5', 'T-type Label_Pvalb', 'T-type Label_Serpinf1', 'T-type Label_Sncg', 'T-type Label_Sst', 'T-type Label_Vip']
approved_input_features = always_given_features #+ ['1700086L19Rik', '9530059O14Rik', '9630002D21Rik', 'Adarb2', 'Adra1b', 'Asic4', 'B3gat1', 'Cacna2d2', 'Caln1', 'Cbln2', 'Cdh9', 'Cntnap3', 'Col11a1', 'Col6a6', 'Coro6', 'Cort', 'Cpne7', 'Creb3l2', 'Crhbp', 'Crisp1', 'Ddx3y', 'Egfr', 'Eif2s3y', 'Elfn1', 'Ephb6', 'Ephx4', 'Ept1', 'Fam135b', 'Fxyd6', 'Gabrd', 'Galnt9', 'Gm13629', 'Gm1604b', 'Gm17171', 'Gm18668', 'Gm30015', 'Gm31329', 'Gm34257', 'Gm35290', 'Gm35736', 'Gprc5b', 'Htr3a', 'Ido2', 'Kcnk12', 'Kcnk9', 'Kit', 'Klf5', 'LOC102633357', 'LOC105244000', 'LOC105244058', 'LOC105246151', 'LOC105246187', 'LOC105247131', 'Lhx6', 'Lypd6', 'Mafb', 'Masp1', 'Moxd1', 'Mpped1', 'Necab1', 'Nxph1', 'Oprd1', 'Pde11a', 'Pla2g7', 'Plch2', 'Prox1', 'Pstpip1', 'Pthlh', 'Ptprm', 'Ranbp3l', 'Rassf5', 'Rbp4', 'Rgs12', 'Rps25-ps1', 'Samd5', 'Sash1', 'Sorcs3', 'Sox6', 'Spire2', 'Sst', 'Stxbp6', 'Tgfb3', 'Them4', 'Tmem44', 'Trhde', 'Trim66', 'Trpc6', 'Tsix', 'Uty', 'Vip', 'Xist', 'Zfp462']
print(len(approved_input_features))
output_features_to_ignore_zeros = [
	'time_to_first_spike', 'time_to_last_spike',
	'AP_height', 'AP_width',
	'sag_amplitude', 'sag_ratio1', 'sag_time_constant']

metadata_path = 'F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv'
input_data_path = 'F:\Big_MET_data\double_metadata_plus_trans.csv'
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


mean_output_feature = []
valid_inputs = []
just_trans_data = pd.read_csv(input_data_path)
for data_idx in range(len(just_trans_data['cell_specimen_id'].to_list())):
#for data_idx in range(50):
	#print(just_trans_data['ephys_path'].loc[data_idx], just_trans_data['cell_id'].loc[data_idx]) 
	
	try:
		cell_id = int(just_trans_data['cell_specimen_id'].loc[data_idx])
		
		this_cell_ephys = pd.read_csv(f'{output_data_dir}\{cell_id}.csv')
	

		this_current_ephys = this_cell_ephys[np.isclose(this_cell_ephys['current_second_edge'], input_current, atol = 0.0001)]
		
		#we want to ignore things with value 0 for certain features
		#like we don't want to inclde time to first spike when there were no spikes, ya dig?
		output_features = this_current_ephys[this_current_ephys[output_feature] != 0][output_feature].to_numpy()
		
		if len(output_features) > 0:
			mean_output_feature.append(np.mean(output_features))
			this_input_data = just_trans_data.loc[data_idx, approved_input_features].to_numpy().astype(np.float64)
			valid_inputs.append(list(this_input_data))
	except:
		pass
		#print(f'{cell_id} data not found!')
	
mean_output_feature = np.asarray(mean_output_feature)		
valid_inputs = np.asarray(valid_inputs)

valid_inputs = norm_col(valid_inputs)
valid_inputs = valid_inputs > 0 #binarize 
mean_output_feature = norm_col(mean_output_feature)
print(mean_output_feature.shape)

X_train, X_test, y_train, y_test = train_test_split(valid_inputs, mean_output_feature, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Make shape [N, 1]
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current device: {device}')

# Initialize Network
#yes there is a better way to do this, no I aparently don't know how to do it
hid_size = 256
net = nn.Sequential(
    nn.Linear(58, hid_size),
    nn.Tanh(),
	nn.Dropout(0.2),	
    
    nn.Linear(hid_size, hid_size),
    nn.Tanh(),
	nn.Dropout(0.2),
	
    nn.Linear(hid_size, hid_size),
    nn.Tanh(),
	nn.Dropout(0.2),
						    
    nn.Linear(hid_size, 1)
).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.9, 0.999))
num_epochs = 500
loss_hist = []
test_acc_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):  # loop over the dataset multiple times

	running_loss = 0.0
	for i, data in enumerate(train_loader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)

		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i == 0:	# print every 2000 mini-batches
			test_loss = 0
			total = 0
			# since we're not training, we don't need to calculate the gradients for our outputs
			with torch.no_grad():
				for j, data in enumerate(test_loader, 0):
					inputs, labels = data
					inputs = inputs.to(device)
					labels = labels.to(device)
					# calculate outputs by running images through the network
					outputs = net(inputs)
					loss = criterion(outputs, labels)
					test_loss += loss.item()
					total += batch_size

				print(f'Loss of the network during epoch {epoch} and {i}: {test_loss}')
	torch.save(net.state_dict(), f"F:\\Big_MET_data\\fresh_predict_NNs\\{epoch}_predict_derived_ephys.pth")

print('Finished Training')


net.eval()

all_outs = []
all_labels = []
with torch.no_grad():
	for i, data in enumerate(test_loader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data
		dev_inputs = inputs.to(device)
	
		# forward + backward + optimize
		dev_outputs = net(dev_inputs)
		outputs = dev_outputs.detach().cpu().numpy()
		labels = labels.numpy()
		
		all_outs += list(outputs)
		all_labels += list(labels)
	
plt.title(f'{output_feature} prediction')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.scatter(all_outs, all_labels)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('model prediction')
plt.ylabel('true value')
plt.show()

print("R² on test:", r2_score(all_labels, all_outs))
print("MSE on test:", mean_squared_error(all_labels, all_outs))

all_outs = []
all_labels = []
with torch.no_grad():
	for i, data in enumerate(train_loader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data
		dev_inputs = inputs.to(device)
	
		# forward + backward + optimize
		dev_outputs = net(dev_inputs)
		outputs = dev_outputs.detach().cpu().numpy()
		labels = labels.numpy()
		
		all_outs += list(outputs)
		all_labels += list(labels)
	
plt.title(f'{output_feature} prediction on TRAIN data')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.scatter(all_outs, all_labels)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('model prediction')
plt.ylabel('true value')
plt.show()

print("R² on train:", r2_score(all_labels, all_outs))
print("MSE on train:", mean_squared_error(all_labels, all_outs))