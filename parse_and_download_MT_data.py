import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import requests
from urllib.request import urlretrieve

def download_file(url, func_path):
	filename = url.split('/')[-1]
	filepath = os.path.join(func_path, filename)
	
	try:
		if url.startswith('ftp://'):
			urlretrieve(url, filepath)
		else:
			response = requests.get(url, timeout=60)
			response.raise_for_status()
			with open(filepath, 'wb') as f:
				f.write(response.content)
		
		print("Success")
	
	except Exception as e:
		print(url, f"Failed: {e}")

meta_data_path = r'F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv'
manifest_data_path = r'F:\Big_MET_data\2021-09-13_mouse_file_manifest.xlsx'
big_save_folder = r'F:\Big_MET_data\morpho_trans'

meta_data_df = pd.read_csv(meta_data_path)
manifest_data_df = pd.read_excel(manifest_data_path)
num_cols_meta = meta_data_df.shape[0]
num_cols_mani = manifest_data_df.shape[0]

print(meta_data_df.columns)
print(meta_data_df[['MET-type Label']])
print(meta_data_df.shape)

MET_full_idxs = []
approved_cell_spec_ids = []
for i in range(num_cols_meta):
	has_MET = 'MET' in str(meta_data_df['MET-type Label'].loc[i])
	full_recon = 'full' in str(meta_data_df['neuron_reconstruction_type'].loc[i])
	if has_MET and full_recon:
		MET_full_idxs.append(i)
		this_cell_spec_id = meta_data_df['cell_specimen_id'].loc[i]
		approved_cell_spec_ids.append(this_cell_spec_id)
		full_path = rf'{big_save_folder}\{this_cell_spec_id}'
		'''
		try:
			os.mkdir(full_path)
		except:
			pass
		'''

for i in range(num_cols_mani-4):
	this_cell_spec_id = int(manifest_data_df['cell_specimen_id'].loc[i])
	if this_cell_spec_id in approved_cell_spec_ids:
		archive = manifest_data_df['archive'].loc[i]
		if 'DANDI' not in archive:
			url = archive = manifest_data_df['archive_uri'].loc[i]
			file_type = manifest_data_df['file_type'].loc[i]
			#filename = url.split('/')[-1]
			#full_path = rf'{big_save_folder}'
			#print(full_path, this_cell_spec_id)
			download_file(url, big_save_folder)
			

