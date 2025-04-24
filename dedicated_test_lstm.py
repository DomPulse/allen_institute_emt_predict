import load_patch_clamp_data as lpcd
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#kind of copying idea: https://elifesciences.org/articles/79535.pdf
#stolen lstm code: https://www.geeksforgeeks.org/long-short-term-memory-networks-using-pytorch/
#and https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

available_device = (
	"cuda"
	if torch.cuda.is_available()
	else "mps"
	if torch.backends.mps.is_available()
	else "cpu"
)

print(f"Available device: {available_device}")


folder_path = "D:\\Neuro_Sci\\morph_ephys_trans_stuff\\fine_and_dandi_ephys\\000020\\sub-610663891\\"
model_path = "10_LSTM"

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
all_times, all_currents, all_volts, orginal_lengths = lpcd.give_me_the_stuff(folder_path, files)

#should quite probably functionalize this :/
ephys_time_step = np.max(all_times)/(len(all_times[0])) #time for each data point in ms
des_time_step = 1
sequence_length = 100 #sequence length in ms
look_forward_time_steps = int(sequence_length/ephys_time_step)
data_step = int(des_time_step/ephys_time_step)

print(look_forward_time_steps)

num_experiments = len(all_times)

take_every_n = 10 #so there is a ton of data here, like way too much to fit on my gpu, so im going to take every n data points to not do that

#end of goofy thing that should be functionalized


class LSTMModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
		super(LSTMModel, self).__init__()
		self.hidden_dim = hidden_dim
		self.layer_dim = layer_dim
		self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x, h0=None, c0=None):
		if h0 is None or c0 is None:
			h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
			c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
		
		out, (hn, cn) = self.lstm(x, (h0, c0))
		out = self.fc(out[:, -1, :])
		return out, hn, cn

model = LSTMModel(input_dim=2, hidden_dim=100, layer_dim=2, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

batch_size = 1
h0, c0 = None, None


loaded_model = LSTMModel(input_dim=2, hidden_dim=100, layer_dim=2, output_dim=1)
loaded_model.load_state_dict(torch.load(model_path))
print('model reloaded successfully!')

for i in range(num_experiments):
	data_in_train = (i%2 == 0) #usually it's 80/20 train/test but there are differnt types of experiments and I want it to have data for all you see
	if data_in_train:
		pass
	else:
		test_input = []
		test_output = []
		
		stop_at = orginal_lengths[i]-1
		
		combo_volt_curr = np.zeros((stop_at, 2))
		combo_volt_curr[:, 0] = all_volts[i, :stop_at]
		combo_volt_curr[:, 1] = all_currents[i, :stop_at]
				
		for t in range(0, stop_at - look_forward_time_steps, take_every_n):
			input_data = combo_volt_curr[t:t+look_forward_time_steps:data_step]
			output_data = all_volts[i, t+look_forward_time_steps]
			test_input.append(input_data)
			test_output.append(output_data)
			
		testX = torch.tensor(test_input, dtype=torch.float32)
		testY = torch.tensor(test_output, dtype=torch.float32)
		num_data_samples = len(testX)
		
		cpu_answer = 0
		outputs = 0
		lstm_answers = []
		right_answers = []
		for j in range(0, num_data_samples):
			batch_idxs = [j, 0] #expects a 3D tensor and fuck me if i try and give it only one
			labels = testY[batch_idxs]
			
			if j == 0:
				inputs = testX[batch_idxs]
				
			else:
				inputs = inputs.roll(-1)
				not_inputs = testX[batch_idxs]
				inputs[0, -1, 0] = torch.tensor(cpu_answer, dtype=torch.float32)[0, 0]
				inputs[0, -1, 1] = not_inputs[0, -1, 1]
		
			model.eval()
		
			outputs, h0, c0 = loaded_model(inputs, h0, c0)
			cpu_answer = outputs.detach().numpy()[0, 0]
			
			right_answers.append(labels[0])
			lstm_answers.append(cpu_answer)
		
		plt.plot(right_answers)
		plt.plot(lstm_answers)
		plt.show()

			
			
			
