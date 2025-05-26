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
approved_input_features = ['hemisphere_right', 'biological_sex_M',
      'structure_VISl2/3', 'structure_VISl5', 'structure_VISl6a',
      'structure_VISp1', 'structure_VISp2/3', 'structure_VISp4',
      'structure_VISp5', 'structure_VISp6a', 'structure_VISpm2/3',
      'structure_VISpm5', 'structure_VISpm6a', 'T-type Label_Lamp5',
      'T-type Label_Pvalb', 'T-type Label_Serpinf1', 'T-type Label_Sncg',
      'T-type Label_Sst', 'T-type Label_Vip', 'cluster_0_0', 'cluster_0_1',
      'cluster_0_2', 'cluster_1_0', 'cluster_1_1', 'cluster_1_2',
      'cluster_2_0', 'cluster_2_1', 'cluster_2_2', 'cluster_3_0',
      'cluster_3_1', 'cluster_3_2', 'cluster_4_0', 'cluster_4_1',
      'cluster_4_2', 'cluster_5_0', 'cluster_5_1', 'cluster_5_2',
      'cluster_6_0', 'cluster_6_1', 'cluster_6_2', 'cluster_7_0',
      'cluster_7_1', 'cluster_7_2', 'cluster_8_0', 'cluster_8_1',
      'cluster_8_2', 'cluster_9_0', 'cluster_9_1', 'cluster_9_2',
      'cluster_10_0', 'cluster_10_1', 'cluster_10_2', 'cluster_11_0',
      'cluster_11_1', 'cluster_11_2', 'cluster_12_0', 'cluster_12_1',
      'cluster_12_2', 'cluster_13_0', 'cluster_13_1', 'cluster_13_2',
      'cluster_14_0', 'cluster_14_1', 'cluster_14_2']

print(len(approved_input_features))
output_features_to_ignore_zeros = [
	'time_to_first_spike', 'time_to_last_spike',
	'AP_height', 'AP_width',
	'sag_amplitude', 'sag_ratio1', 'sag_time_constant']

metadata_path = 'F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv'
input_data_path = 'F:\Big_MET_data\\double_meta_plus_clusters.csv'
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
hid_size = 128
net = nn.Sequential(
    nn.Linear(len(approved_input_features), hid_size),
    nn.Tanh(),
	nn.Dropout(0.2),	
    
    nn.Linear(hid_size, hid_size),
    nn.Tanh(),
	nn.Dropout(0.2),
	
    nn.Linear(hid_size, hid_size),
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

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.9, 0.999))
num_epochs = 1000
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
		if i == 0 and epoch%20 == 0:	# print every 2000 mini-batches
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