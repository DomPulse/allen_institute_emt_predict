import json
import random
import csv
from pathlib import Path

def generate_random_json(template_file, csv_file, output_file, rename_dict=None):
	# Load template JSON
	with open(template_file, "r") as f:
		template = json.load(f)

	# Read CSV ranges into a dictionary
	ranges = {}
	with open(csv_file, "r") as f:
		reader = csv.DictReader(f)
		for row in reader:
			key = (row["section"], row["name"], row["mechanism"])
			try:
				low_val = float(row["low"])
				high_val = float(row["high"])
				ranges[key] = (low_val, high_val)
			except:
				print(f"Skipping invalid row: {row}")

	# Populate passive Ra
	for passive_entry in template.get("passive", []):
		if "ra" in passive_entry:
			key = ("soma", "Ra", "")
			if key in ranges:
				low_val, high_val = ranges[key]
				passive_entry["ra"] = random.uniform(low_val, high_val)

	# Populate genome entries and rename mechanisms/names
	for entry in template.get("genome", []):
		mech = entry.get("mechanism", "")
		name = entry.get("name", "")

		if rename_dict:
			# Check if the base mechanism should be renamed
			if mech in rename_dict:
				new_base = rename_dict[mech]
				entry["mechanism"] = new_base
				# Update the name if it contains the old base name
				if mech in name:
					entry["name"] = name.replace(mech, new_base)

		# Use updated values for range lookup
		key = (entry.get("section"), entry.get("name"), entry.get("mechanism", ""))
		if key in ranges:
			low_val, high_val = ranges[key]
			entry["value"] = str(random.uniform(low_val, high_val))

	# Save new JSON
	with open(output_file, "w") as f:
		json.dump(template, f, indent=4)


# -----------------------
# Usage example
# -----------------------
template_json = r'F:\Big_MET_data\different_all_active_model\just_genomes\fit_parameters_24.json'
csv_file = r'D:\Neuro_Sci\morph_ephys_trans_stuff\genome_stats_percentile.csv'
output_path = r'F:\arbor_ubuntu\10k_randomized_jsons'

# dictionary for renaming base mechanism names
rename_dict = {
	"K_Pst": "K_P",
	"NaTa_t": "NaTa",
	"Nap_Et2": "Nap",
	"K_Tst": "K_T",
	"NaTs2_t": "NaTs"
	
	# add more as needed
}
rename_dict = {}

num_jsons = 10000
for i in range(num_jsons):
	full_output_path = rf'{output_path}\random_genome_{i}.json'
	generate_random_json(template_json, csv_file, full_output_path, rename_dict=rename_dict)
	if i % 100 == 0:
		print(f'total of {i} saved')
