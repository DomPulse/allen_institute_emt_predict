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
DATA_DIR = "F:\\Big_MET_data\\single_pulses_only"
norm_save_dir = "F:\\Big_MET_data\\norm_and_encode_metadata\\single_pulse"
STRING_COLUMN_INDEX = 0  # Adjusted because we drop 3 columns before indexing
MIN_FILE_SIZE_BYTES = 3072  # Skip files smaller than 1KB
test_size = 0.2
batch_size = 64

# Version-safe encoder
if version.parse(sklearn_version) >= version.parse("1.2"):
	encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
else:
	encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

def minmax_scale_neg1_1(X):
	X_min = X.min(axis=0)
	X_max = X.max(axis=0)
	denom = np.where(X_max - X_min == 0, 1, X_max - X_min)
	X_scaled = 2 * (X - X_min) / denom - 1
	return X_scaled, X_min, X_max

def save_minmax_csv(min_vals, max_vals, column_names, out_path):
	df = pd.DataFrame({
		'column': column_names,
		'min': min_vals,
		'max': max_vals
	})
	df.to_csv(out_path, index=False)


class PreprocessingArtifacts:
	def __init__(self, save_dir, use_existing=True):
		self.save_dir = save_dir
		self.input_minmax_path = os.path.join(save_dir, "input_minmax.csv")
		self.output_minmax_path = os.path.join(save_dir, "output_minmax.csv")

		if use_existing:
			self.X_min, self.X_max = self._load_minmax_csv(self.input_minmax_path)
			self.y_min, self.y_max = self._load_minmax_csv(self.output_minmax_path)
		else:
			self.X_min = self.X_max = self.y_min = self.y_max = None

	def normalize_inputs(self, X, save=True):
		self.X_min = X.min(axis=0)
		self.X_max = X.max(axis=0)
		X_scaled = self._normalize(X, self.X_min, self.X_max)
		if save:
			self._save_minmax_csv(self.X_min, self.X_max, X.shape[1], self.input_minmax_path)
		return X_scaled

	def normalize_outputs(self, y, save=True):
		self.y_min = y.min(axis=0)
		self.y_max = y.max(axis=0)
		y_scaled = self._normalize(y, self.y_min, self.y_max)
		if save:
			self._save_minmax_csv(self.y_min, self.y_max, y.shape[1], self.output_minmax_path)
		return y_scaled

	def transform_inputs(self, X):
		return self._normalize(X, self.X_min, self.X_max)

	def transform_outputs(self, y):
		return self._normalize(y, self.y_min, self.y_max)

	def inverse_transform_outputs(self, y_scaled):
		return self._denormalize(y_scaled, self.y_min, self.y_max)

	def _normalize(self, arr, min_vals, max_vals):
		denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
		return 2 * (arr - min_vals) / denom - 1

	def _denormalize(self, arr, min_vals, max_vals):
		denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
		return (arr + 1) / 2 * denom + min_vals

	def _save_minmax_csv(self, min_vals, max_vals, dim, path):
		df = pd.DataFrame({
			'column': [f"col_{i}" for i in range(dim)],
			'min': min_vals,
			'max': max_vals
		})
		df.to_csv(path, index=False)

	def _load_minmax_csv(self, path):
		df = pd.read_csv(path)
		return df['min'].values, df['max'].values

def load_and_process_data(data_dir):
	input_files = glob.glob(os.path.join(data_dir, "*_in_test.csv"))

	valid_file_pairs = []
	input_list, output_list = [], []

	# Initialize with toggle
	artifacts = PreprocessingArtifacts(norm_save_dir, use_existing=False)  # set to True when reusing
	
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
		out_df = pd.read_csv(out_file).iloc[:, 3:]
		
		input_list.append(in_df.values)
		output_list.append(out_df.values)
	
	# Stack and normalize
	X = np.vstack(input_list)
	y = np.vstack(output_list)
	X = artifacts.normalize_inputs(X, save=True)
	y = artifacts.normalize_outputs(y, save=True)

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
dropout_p = 0.25
hid_size = 1024
act_func = nn.Tanh()
net = nn.Sequential(
	nn.Linear(830, hid_size),
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


criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5, betas=(0.8, 0.9))
#optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)
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
		if i % 10 == 0 and i != 0:	# print every 2000 mini-batches
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
	torch.save(net.state_dict(), f"F:\\Big_MET_data\\single_pulse_predict_NNs\\{epoch}_predict_derived_ephys.pth")

print('Finished Training')

