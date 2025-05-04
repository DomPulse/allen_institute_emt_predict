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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

# --- Config ---
DATA_DIR = "F:\Big_MET_data\my_proced_data"
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


def load_and_process_data(data_dir):
	input_files = glob.glob(os.path.join(data_dir, "*_in_test.csv"))

	valid_file_pairs = []
	all_strings = []

	# First pass: filter valid file pairs and collect string column values
	for in_file in input_files:
		prefix = os.path.basename(in_file).split('_')[0]
		out_file = os.path.join(data_dir, f"{prefix}_out_test.csv")

		if not os.path.exists(out_file):
			continue
		if os.path.getsize(in_file) < MIN_FILE_SIZE_BYTES or os.path.getsize(out_file) < MIN_FILE_SIZE_BYTES:
			continue

		try:
			in_df = pd.read_csv(in_file).iloc[:, 3:]
			if in_df.empty:
				continue
			string_col = in_df.iloc[:, STRING_COLUMN_INDEX].astype(str)
			all_strings.extend(string_col.tolist())
			valid_file_pairs.append((in_file, out_file))
		except Exception as e:
			print(f"Error in first pass on {prefix}: {e}")

	if not valid_file_pairs:
		raise ValueError("No valid file pairs found.")

	# Fit encoder on all string values
	encoder.fit(np.array(all_strings).reshape(-1, 1))

	input_list, output_list = [], []

	# Second pass: transform all data using fitted encoder
	for in_file, out_file in valid_file_pairs:
		try:
			in_df = pd.read_csv(in_file).iloc[:, 3:]
			out_df = pd.read_csv(out_file).iloc[:, 3:]

			if in_df.empty or out_df.empty:
				continue

			string_col = in_df.iloc[:, STRING_COLUMN_INDEX].astype(str)
			float_cols = in_df.drop(in_df.columns[STRING_COLUMN_INDEX], axis=1)

			string_encoded = encoder.transform(string_col.values.reshape(-1, 1))
			in_processed = np.hstack([float_cols.values, string_encoded])

			input_list.append(in_processed)
			output_list.append(out_df.values)
		except Exception as e:
			print(f"Error in second pass on {in_file}: {e}")

	X = np.vstack(input_list)
	y = np.vstack(output_list)

	# Normalize input
	X, X_min, X_max = minmax_scale_neg1_1(X)
	input_columns = [f"input_{i}" for i in range(X.shape[1])]
	save_minmax_csv(X_min, X_max, input_columns, "input_minmax.csv")

	# Normalize output
	y, y_min, y_max = minmax_scale_neg1_1(y)
	output_columns = [f"output_{i}" for i in range(y.shape[1])]
	save_minmax_csv(y_min, y_max, output_columns, "output_minmax.csv")

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
hid_size = 1024
act_func = nn.Tanh()
net = nn.Sequential(
	nn.Linear(840, hid_size),
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
	
	nn.Linear(hid_size, hid_size),
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

	nn.Linear(hid_size, 13),
	act_func
	).to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, betas=(0.8, 0.9))
#optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)
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
		if i % 100 == 0 and i != 0:	# print every 2000 mini-batches
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
	torch.save(net.state_dict(), f"F:\Big_MET_data\predict_NNs\{epoch}_predict_derived_ephys.pth")

print('Finished Training')

