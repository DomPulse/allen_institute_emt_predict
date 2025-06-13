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

def norm_col(array):
	min_by_col = np.min(array, axis = 0) 
	array = np.subtract(array, min_by_col)
	max_by_col = np.max(array, axis = 0)
	return np.divide(array, max_by_col + 1E-8)

def norm_row(array):
	min_by_row = np.min(array, axis = 1) 
	min_by_row = min_by_row.reshape(-1,1)
	array = np.subtract(array, min_by_row)
	max_by_row = np.max(array, axis = 1)
	max_by_row = max_by_row.reshape(-1,1)
	return np.divide(array, max_by_row)

cluster_we_want = 'just_my_type_6'
pos_cols = []
embed_dim = 10
for i in range(embed_dim):
	pos_cols.append(f'{i}_dim_pos')

all_data_path = r'F:\Big_MET_data\umap_pos.csv'
all_data = pd.read_csv(all_data_path)
all_data = all_data[all_data[cluster_we_want] == 1]

umap_pos = all_data[pos_cols].to_numpy()
rest_volt = all_data['mean_resting_volt'].to_numpy()
umap_pos = norm_col(umap_pos)
rest_volt = norm_col(rest_volt)

X_train, X_test, y_train, y_test = train_test_split(umap_pos, rest_volt, test_size=0.2, random_state=42)
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
    nn.Linear(len(pos_cols), hid_size),
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
num_epochs = 2400  
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
	
plt.title('rest_volt prediction')
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
	
plt.title('rest_volt prediction on TRAIN data')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.scatter(all_outs, all_labels)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('model prediction')
plt.ylabel('true value')
plt.show()

print("R² on train:", r2_score(all_labels, all_outs))
print("MSE on train:", mean_squared_error(all_labels, all_outs))