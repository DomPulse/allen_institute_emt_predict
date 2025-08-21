import json
from pathlib import Path
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def gather_genome_stats(folder):
	stats = defaultdict(list)

	for file_path in Path(folder).rglob("*.json"):
		try:
			with open(file_path, "r") as f:
				data = json.load(f)
		except Exception as e:
			print(f"Skipping {file_path}: {e}")
			continue

		# Passive Ra entries
		for passive_entry in data.get("passive", []):
			ra_val = passive_entry.get("ra")
			if ra_val is not None:
				try:
					stats[("soma", "Ra", "")].append(float(ra_val))
				except:
					pass

		# Genome entries
		for entry in data.get("genome", []):
			section = entry.get("section")
			name = entry.get("name")
			mechanism = entry.get("mechanism", "")
			value = entry.get("value")
			if section and name and value is not None:
				try:
					stats[(section, name, mechanism)].append(float(value))
				except:
					print(f"Skipping bad value in {file_path}: {value}")

	# Convert stats to min/max per key
	output_rows = []
	for (section, name, mechanism), values in stats.items():
		if values:
			output_rows.append({
				"section": section,
				"name": name,
				"mechanism": mechanism,
				"low": np.percentile(values, 10),
				"high": np.percentile(values, 90)
			})

	return stats, output_rows  # <-- return full values for plotting too

def save_stats_to_csv(stats_rows, out_file):
	fieldnames = ["section", "name", "mechanism", "low", "high"]
	with open(out_file, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in stats_rows:
			writer.writerow(row)

# Usage
folder_path = r'F:\Big_MET_data\different_all_active_model\just_genomes'
out_file = "genome_stats_percentile.csv"

stats, stats_rows = gather_genome_stats(folder_path)
save_stats_to_csv(stats_rows, out_file)

print(f"Saved stats for {len(stats_rows)} entries to {out_file}")

# ------------------------
# Plot distributions
# ------------------------
for (section, name, mechanism), values in stats.items():
	plt.figure()
	plt.hist(values, bins=30, color='skyblue', edgecolor='black')
	mech_label = f" ({mechanism})" if mechanism else ""
	plt.title(f"Distribution of {name}{mech_label} in {section}")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	plt.show()
