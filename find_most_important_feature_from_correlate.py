import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

full_save_path = r'F:\Big_MET_data\feature_extraction'
currents_to_check = [-0.11, -0.09, -0.07, -0.05, -0.03, 0.05, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25]
efel_features_col_names = [
	'steady_state_voltage', 'steady_state_voltage_stimend',
	'time_to_first_spike', 'time_to_last_spike',
	'spike_count', 'AP_height', 'AP_width',
	'sag_amplitude', 'sag_ratio1', 'sag_time_constant']

mutual_info_data_frames = []
for current in currents_to_check:
	mutual_info_data_frames.append(pd.read_csv(f'{full_save_path}\{current}_mutual_info.csv'))
	#print(mutual_info_data_frames[-1])

input_feature_names = mutual_info_data_frames[-1]['input_features']
votes_by_feature = np.zeros(len(input_feature_names))

for output_feature in efel_features_col_names:
	mutual_infos_for_feat = []
	for df in mutual_info_data_frames:
		try:
			mutual_infos_for_feat.append(df[output_feature].to_list())
		except:
			pass
	mutual_infos_for_feat = np.asarray(mutual_infos_for_feat)
	mean_mutual_info = np.mean(mutual_infos_for_feat, axis = 0)
	percentile_cutoff = np.percentile(mean_mutual_info, 99.75)
	important = mean_mutual_info > percentile_cutoff
	votes_by_feature += important
	print(output_feature, np.max(mean_mutual_info), np.sum(important))

print(np.max(votes_by_feature))
absolute_best_features = []
for idx, val in enumerate(votes_by_feature):
	if val > 1:
		print(val, input_feature_names[idx])
		absolute_best_features.append(input_feature_names[idx])
print(absolute_best_features, len(absolute_best_features))
		
