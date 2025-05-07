import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version
from packaging import version
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

# --- Config ---
DATA_DIR = "F:\\Big_MET_data\\single_pulse_selected_genes"
norm_save_dir = "F:\\Big_MET_data\\norm_and_encode_metadata\\single_pulse"
STRING_COLUMN_INDEX = 0  # Adjusted because we drop 3 columns before indexing
MIN_FILE_SIZE_BYTES = 3072  # Skip files smaller than 1KB
test_size = 0.2
batch_size = 64

def norm_col(array):
	min_by_col = np.min(array, axis = 0) 
	array = np.subtract(array, min_by_col)
	max_by_col = np.max(array, axis = 0)

	return np.divide(array, max_by_col)

def norm_row(array):
	min_by_row = np.min(array, axis = 0) 
	array = np.subtract(array, min_by_row)
	max_by_row = np.max(array, axis = 1)
	max_by_row = max_by_row.reshape(-1,1)
	print(max_by_row.shape)
	return np.divide(array, max_by_row)

def load_and_process_data(data_dir):
	input_files = glob.glob(os.path.join(data_dir, "*_in_test.csv"))

	valid_file_pairs = []
	input_list, output_list = [], []
		
	# First pass to collect string values
	print(len(input_files))
	for in_file in input_files:
		prefix = os.path.basename(in_file).split('_')[0]
		out_file = os.path.join(data_dir, f"{prefix}_out_test.csv")

		if not os.path.exists(out_file):
			continue
		if os.path.getsize(in_file) < MIN_FILE_SIZE_BYTES or os.path.getsize(out_file) < MIN_FILE_SIZE_BYTES:
			continue

		try:
			in_df = pd.read_csv(in_file).iloc[:, 3:]
		except Exception as e:
			print(f"Error in first pass on {prefix}: {e}")
		
		valid_file_pairs.append((in_file, out_file))

		
	# Second pass
	for in_file, out_file in valid_file_pairs:
		in_df = pd.read_csv(in_file).iloc[:, 3:]
		only_positive_rows = (in_df['current_second_edge'] > 0).to_numpy()
		
		in_df = pd.read_csv(in_file).iloc[only_positive_rows, 3:]
		out_df = pd.read_csv(out_file).iloc[only_positive_rows, 3:]
		
		input_list.append(in_df.values)
		output_list.append(out_df.values)
	
	# Stack and normalize
	X = np.vstack(input_list)
	y = np.vstack(output_list)
	print(X.shape)
	X[:, 0:4] = norm_col(X[:, 0:4])
	X[:, 4:] = norm_row(X[:, 4:])
	y = norm_col(y)
	y[np.isnan(y)] = 0

	return X, y


def create_dataloaders(X, y, test_size, batch_size):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

	train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
								  torch.tensor(y_train, dtype=torch.float32))
	test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
								 torch.tensor(y_test, dtype=torch.float32))

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size)

	return train_loader, test_loader

# --- Run everything ---
X, y = load_and_process_data(DATA_DIR)
train_loader, test_loader = create_dataloaders(X, y, test_size, batch_size)

print("Data loaded and DataLoaders ready.")


# Initialize Network
#yes there is a better way to do this, no I aparently don't know how to do it
dropout_p = 0.1
hid_size = 256
act_func = nn.Sigmoid()
net = nn.Sequential(
	nn.Linear(147, hid_size),
	act_func,
	nn.Dropout(dropout_p),
	
	nn.Linear(hid_size, hid_size),
	act_func,
	nn.Dropout(dropout_p),
	
	nn.Linear(hid_size, hid_size),
	act_func,
	nn.Dropout(dropout_p),
	
	nn.Linear(hid_size, hid_size),
	act_func,
	nn.Dropout(dropout_p),
			
	nn.Linear(hid_size, 9),
	).to(device)



net.load_state_dict(torch.load("F:\\Big_MET_data\\single_pulse_predict_NNs\\10_predict_derived_ephys.pth"))

out_size = 9
all_outputs = np.zeros((0, out_size))
all_labels = np.zeros((0, out_size))

out_labels = ['voltage_base', 'time_to_first_spike', 'time_to_last_spike',
				  'sag_amplitude', 'sag_ratio1', 'sag_time_constant',
				  'minimum_voltage', 'maximum_voltage', 'spike_count']

with torch.no_grad():
	for i, data in enumerate(test_loader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data
		dev_inputs = inputs.to(device)
	
		# forward + backward + optimize
		dev_outputs = net(dev_inputs)
		outputs = dev_outputs.detach().cpu().numpy()
		labels = labels.numpy()
		
		all_outputs = np.concatenate((all_outputs, outputs), axis = 0) 
		all_labels = np.concatenate((all_labels, labels), axis = 0) 
		
	for j in range(out_size):
		plt.title(out_labels[j])
		plt.xlim(0, 1)
		plt.ylim(0, 1)
		plt.scatter(all_outputs[:, j], all_labels[:, j])
		plt.xlabel('model prediction')
		plt.ylabel('true value')
		plt.show()

