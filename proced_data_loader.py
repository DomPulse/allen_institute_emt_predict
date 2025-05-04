import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version
from packaging import version
import torch
from torch.utils.data import TensorDataset, DataLoader

# --- Config ---
DATA_DIR = "F:\Big_MET_data\my_proced_data"
STRING_COLUMN_INDEX = 0  # Adjusted because we drop 3 columns before indexing
MIN_FILE_SIZE_BYTES = 3072  # Skip files smaller than 1KB
TEST_SIZE = 0.2
BATCH_SIZE = 64

# Version-safe encoder
if version.parse(sklearn_version) >= version.parse("1.2"):
	encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
else:
	encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

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
train_loader, test_loader = create_dataloaders(X, y, TEST_SIZE, BATCH_SIZE)

print("Data loaded and DataLoaders ready.")
