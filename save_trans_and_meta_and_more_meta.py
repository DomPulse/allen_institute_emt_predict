import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc

def list_all_files(root_folder):
	session_search = []
	session_to_path = {}
	for dirpath, dirnames, filenames in os.walk(root_folder):
		for filename in filenames:
			post_ses = filename.split("ses-")[-1]
			#print(post_ses.split("_")[0])
			pre_other_stuff = post_ses.split("_")[0]
			try:
				session_search.append(int(pre_other_stuff))
				session_to_path[int(pre_other_stuff)] = f'{dirpath}\\{filename}'
			except:
				pass
	return session_search, session_to_path

#thanks chat <3
prefixes = ['Vip', 'Sst', 'Pvalb', 'Sncg', 'Serpinf1', 'Lamp5']
def map_to_prefix(cell):
	for prefix in prefixes:
		if str(cell).startswith(prefix):
			return prefix
	return 'Other'  # or np.nan if you prefer to ignore unknowns

def make_age_numerical(given_age):
	age = int(given_age.split('P')[1])
	return age

save_path = r"F:\Big_MET_data"
meta_data_path = r"F:\Big_MET_data\20200711_patchseq_metadata_mouse.csv"
counts_path = r"F:\Big_MET_data\trans_data_cpm\20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
morph_path = "F:\Big_MET_data\single_pulses_only"
ephys_path = r"F:\Big_MET_data\fine_and_dandi_ephys\000020"
ephys_ses, ses_path_dict = list_all_files(ephys_path)
meta_data_df = pd.read_csv(meta_data_path)
counts_df = pd.read_csv(counts_path)

col_names = list(counts_df) #these are like the names of the cells or the experiment or whatever
valid_ephys_row = meta_data_df.index[(meta_data_df['transcriptomics_sample_id'].isin(col_names)) & (meta_data_df['ephys_session_id'].isin(ephys_ses))].tolist()
meta_data_df = meta_data_df.iloc[valid_ephys_row]
gene_names = list(counts_df[col_names[0]])
	
#just get some specific labels that will hopefully be predictive
interesting_meta = ['transcriptomics_sample_id', 'cell_specimen_id', 'ephys_session_id', 'hemisphere', 'structure', 'biological_sex', 'age', 'T-type Label']
filt_meta_data_df = meta_data_df[interesting_meta]

#filter out some of the ages that are in a different format
valid_age_rows = filt_meta_data_df.index[(filt_meta_data_df['age'].str.contains('P'))].tolist()
filt_meta_data_df = filt_meta_data_df.iloc[valid_age_rows]

#convert all those ages to being numbers
ages = filt_meta_data_df['age'].tolist()
for index, age in enumerate(ages):
	ages[index] = make_age_numerical(age)
ages = np.asarray(ages)
filt_meta_data_df['age'] = ages

#add in the full ephys path
correct_eses_paths = filt_meta_data_df['ephys_session_id'].tolist()
for index, eses_id in enumerate(correct_eses_paths):
	correct_eses_paths[index] = ses_path_dict[eses_id]
filt_meta_data_df['ephys_path'] = correct_eses_paths

#sex and hemisphere are binary and thus don't need two columns
bin_cat_cols = ['hemisphere', 'biological_sex']
decode_meta_data_df = pd.get_dummies(filt_meta_data_df, columns = bin_cat_cols, drop_first=True)

#but the others do
#but we want to optionally filter the T-type first
decode_meta_data_df['T-type Label'] = decode_meta_data_df['T-type Label'].apply(map_to_prefix)

class_cal_cols = ['structure', 'T-type Label']
decode_meta_data_df = pd.get_dummies(decode_meta_data_df, columns = class_cal_cols)
decode_meta_data_df = decode_meta_data_df.drop('T-type Label_Other', axis=1) #deletes the other column
decode_meta_data_df.index = range(decode_meta_data_df.shape[0]) #renames the rows just numerically

print(decode_meta_data_df)

transcriptomics_sample_ids = decode_meta_data_df['transcriptomics_sample_id'].tolist()
#print(transcriptomics_sample_ids)
counts_to_concat = counts_df[transcriptomics_sample_ids].T.reset_index(drop=True)

print(counts_to_concat)

the_meta_data_cols = list(decode_meta_data_df.columns)
all_columns = the_meta_data_cols + gene_names

df_new = pd.concat([decode_meta_data_df, counts_to_concat], axis=1)
df_new.columns = all_columns

print(df_new)

df_new.to_csv(os.path.join(save_path, 'double_metadata_plus_trans.csv'))

print(the_meta_data_cols)