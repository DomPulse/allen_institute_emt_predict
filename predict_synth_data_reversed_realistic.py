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

def norm_col(array):
	min_by_col = np.min(array, axis = 0) 
	array = np.subtract(array, min_by_col)
	max_by_col = np.max(array, axis = 0)
	return np.divide(array, max_by_col)

data_path = r'F:\arbor_ubuntu\synth_data_one_morph_more_stims_v2.csv'
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
	'total_volume'
	]

cell_props = [
	#section_name
	'soma_Ra',
	'soma_g_pas',
	'soma_e_pas',
	'axon_g_pas',
	'axon_e_pas',
	'apic_g_pas',
	'apic_e_pas',
	'dend_g_pas',
	'dend_e_pas',
	'soma_cm',
	'axon_cm',
	'axon_Ra',
	'apic_cm',
	'apic_Ra',
	'dend_cm',
	'dend_Ra',
	'axon_gbar_NaV',
	'axon_gbar_K_T',
	'axon_gbar_Kd',
	'axon_gbar_Kv2like',
	'axon_gbar_Kv3_1',
	'axon_gbar_SK',
	'axon_gbar_Ca_HVA',
	'axon_gbar_Ca_LVA',
	'axon_gamma_CaDynamics',
	'axon_decay_CaDynamics',
	'soma_gbar_NaV',
	'soma_gbar_SK',
	'soma_gbar_Kv3_1',
	'soma_gbar_Ca_HVA',
	'soma_gbar_Ca_LVA',
	'soma_gamma_CaDynamics',
	'soma_decay_CaDynamics',
	'soma_gbar_Ih',
	'apic_gbar_NaV',
	'apic_gbar_Kv3_1',
	'apic_gbar_Im_v2',
	'apic_gbar_Ih',
	'dend_gbar_NaV',
	'dend_gbar_Kv3_1',
	'dend_gbar_Im_v2',
	'dend_gbar_Ih'
	]

ming = [
	#section_name
	'soma_Ra',
	'soma_g_pas',
	'soma_e_pas',
	'axon_g_pas',
	'axon_e_pas',
	'apic_g_pas',
	'apic_e_pas',
	'dend_g_pas',
	'dend_e_pas',
	'soma_cm',
	'axon_cm',
	'axon_Ra',
	'apic_cm',
	'apic_Ra',
	'dend_cm',
	'dend_Ra',
	'axon_gbar_K_T',
	'axon_gbar_Kd',
	'axon_gbar_Kv2like',
	'axon_gbar_Kv3_1',
	'axon_gbar_SK',
	'axon_gbar_Ca_HVA',
	'axon_gbar_Ca_LVA',
	'axon_gamma_CaDynamics',
	'axon_decay_CaDynamics',
	'soma_gbar_SK',
	'soma_gbar_Kv3_1',
	'soma_gbar_Ca_HVA',
	'soma_gbar_Ca_LVA',
	'soma_gamma_CaDynamics',
	'soma_decay_CaDynamics',
	'soma_gbar_Ih',
	'apic_gbar_Kv3_1',
	'apic_gbar_Im_v2',
	'apic_gbar_Ih',
	'dend_gbar_Kv3_1',
	'dend_gbar_Im_v2',
	'dend_gbar_Ih'
	]

pos_ephys_properties = ['steady_state_voltage', 'steady_state_voltage_stimend',
				 'time_to_first_spike', 'time_to_last_spike',
				 'spike_count', 'AP_height', 'AP_width',
				 'AHP_depth', 'AP_amplitude_from_voltagebase', 'mean_frequency', 'voltage_base', 
				 'AHP1_depth_from_peak', 'AHP2_depth_from_peak', 'AP_amplitude', 
				 'AP_peak_downstroke', 'AP_peak_upstroke', 'AP_rise_rate_change',
				 'decay_time_constant_after_stim', 'time_constant']

neg_ephys_properties = ['sag_ratio1', 'sag_time_constant', 'steady_state_voltage_stimend']

ephys_feat = []
currents_to_test = [-200, -100, -50, 50, 100, 150, 200]
for c in currents_to_test:
	props = pos_ephys_properties
	if c < 0:
		props = neg_ephys_properties

	for prop in props:
		name = f'{c}_{prop}'
		ephys_feat.append(name)


output_feature = 'soma_gbar_NaV'

input_features = ephys_feat + ming #[x for x in cell_props if x != output_feature]

#output_feature = 'soma_Ra'

df = pd.read_csv(data_path).dropna()
df = df[df['200_spike_count'] > 0]

input_data = df[input_features].to_numpy()
output_data = df[output_feature].to_numpy()

nan_mask = np.where(~np.isnan(output_data))[0]
input_data = input_data[nan_mask]
output_data = output_data[nan_mask]

bottom_percentile = np.percentile(output_data, 10)
top_percentile = np.percentile(output_data, 90)
mask = np.where((output_data > bottom_percentile)*(output_data < top_percentile))[0]

input_data = norm_col(input_data[mask])
output_data = norm_col(output_data[mask])
plt.hist(output_data)
plt.show()

test_size = 0.2
batch_size = 32
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)
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
drop_frac = 0.1
net = nn.Sequential(
    nn.Linear(len(input_features), hid_size),
    nn.Tanh(),
	nn.Dropout(drop_frac),	
    
    nn.Linear(hid_size, hid_size),
    nn.Tanh(),
	nn.Dropout(drop_frac),
	
    
    nn.Linear(hid_size, 1)
).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.9, 0.999))
num_epochs = 300
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
		if epoch%10 == 0 and i == 0:	# print every 2000 mini-batches
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