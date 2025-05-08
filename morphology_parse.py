# A few tools for downloading the data
from io import StringIO
import requests
from neuron_morphology.swc_io import morphology_from_swc
from neuron_morphology.feature_extractor.data import Data
from neuron_morphology.features.branching.bifurcations import num_outer_bifurcations
from neuron_morphology.feature_extractor.utilities import unnest
from neuron_morphology.feature_extractor.feature_extractor import FeatureExtractor
from neuron_morphology.features.dimension import dimension
from neuron_morphology.features.intrinsic import max_branch_order, num_tips, num_branches, num_nodes
from neuron_morphology.features.size import max_euclidean_distance, mean_diameter, total_length, total_surface_area, total_volume
from neuron_morphology.features.path import max_path_distance
import matplotlib.pyplot as plt

test_path = r"F:\Big_MET_data\morpho_trans_extracted\993283588_transformed.swc"
test_data = Data(morphology_from_swc(test_path))


nodes = test_data.morphology.nodes()
x = [node['x'] for node in nodes]
y = [node['y'] for node in nodes]
z = [node['z'] for node in nodes]

fig, ax = plt.subplots(1, 2)
ax[0].scatter(x, y, s=0.1)
ax[1].scatter(z, y, s=0.1)

# make a pipeline
def my_feat_pics(local_test_data):
	
	features = [
	    mean_diameter,
	    max_branch_order,
	    max_euclidean_distance,
		max_path_distance,
		num_outer_bifurcations,
		num_branches,
		num_nodes,
		num_tips,
		total_length,
		total_surface_area,
		total_volume
	]
	
	results = (
	    FeatureExtractor()
	    .register_features(features)
	    .extract(local_test_data)
	    .results
	)
	
	
	return unnest(results)

print(my_feat_pics(test_data))
print(my_feat_pics(test_data))
print(my_feat_pics(test_data))