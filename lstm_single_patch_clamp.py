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

train_input = []
train_output = []
test_input = []
test_output = []
data_in_train = True
take_every_n = 10 #so there is a ton of data here, like way too much to fit on my gpu, so im going to take every n data points to not do that

for i in range(num_experiments):
	data_in_train = (i%2 == 0) #usually it's 80/20 train/test but there are differnt types of experiments and I want it to have data for all you see
	stop_at = orginal_lengths[i]-1

	fig, ax1 = plt.subplots(figsize=(10, 5))
	ax1.set_xlabel('Time (ms)')
	ax1.set_ylabel('Voltage (mV)')
	ax1.plot(all_times[i, :stop_at], all_volts[i, :stop_at], label="real voltage", color = [0, 0, 1])
	ax1.tick_params(axis='y')
	
	ax2 = ax1.twinx()
	ax2.set_ylabel('Current (nA)')
	ax2.plot(all_times[i, :stop_at], all_currents[i, :stop_at], label="injected current", color = [1, 0, 0])
	ax2.tick_params(axis='y')
	
	fig.tight_layout()
	plt.title("Voltage and Current Over Time")
	plt.show()
	
	combo_volt_curr = np.zeros((stop_at, 2))
	combo_volt_curr[:, 0] = all_volts[i, :stop_at]
	combo_volt_curr[:, 1] = all_currents[i, :stop_at]
	
	#print(combo_volt_curr)
	for t in range(0, stop_at - look_forward_time_steps, take_every_n):
		input_data = combo_volt_curr[t:t+look_forward_time_steps:data_step]
		output_data = all_volts[i, t+look_forward_time_steps]
		
		if data_in_train:
			train_input.append(input_data)
			train_output.append(output_data)
		else:
			test_input.append(input_data)
			test_output.append(output_data)
			
train_input = np.asarray(train_input)
train_output = np.asarray(train_output)
test_input = np.asarray(test_input)
test_output = np.asarray(test_output)
#end of goofy thing that should be functionalized

print(train_input.shape)

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


trainX = torch.tensor(train_input, dtype=torch.float32)
trainY = torch.tensor(train_output, dtype=torch.float32)
testX = torch.tensor(test_input, dtype=torch.float32)
testY = torch.tensor(test_output, dtype=torch.float32)
train_dataset = torch.utils.data.TensorDataset(trainX, trainY)
test_dataset = torch.utils.data.TensorDataset(testX, testY)
training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

model = LSTMModel(input_dim=2, hidden_dim=100, layer_dim=2, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


num_epochs = 10
batch_size = 8
num_data_samples = len(trainX)
h0, c0 = None, None

for epoch in range(num_epochs):
	for i in range(int(num_data_samples/batch_size)):
		# Every data instance is an input + label pair
		batch_idxs = np.random.randint(num_data_samples, size = batch_size)
		inputs = trainX[batch_idxs]
		labels = trainY[batch_idxs]

		model.train()
		optimizer.zero_grad()
	
		outputs, h0, c0 = model(inputs, h0, c0)
	
		loss = loss_fn(outputs, labels)
		loss.backward()
		optimizer.step()
	
		h0 = h0.detach()
		c0 = c0.detach()
	
	print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
	model_path = f"{epoch + 1}_LSTM"
	torch.save(model.state_dict(), model_path)

loaded_model = LSTMModel(input_dim=2, hidden_dim=100, layer_dim=2, output_dim=1)
loaded_model.load_state_dict(torch.load(model_path))
print('model reloaded successfully!')

#the old testeroonie
mean_loss = 0
num_data_samples = len(testX)
for i in range(int(num_data_samples/batch_size)):
	# Every data instance is an input + label pair
	batch_idxs = np.random.randint(num_data_samples, size = batch_size)
	inputs = testX[batch_idxs]
	labels = testY[batch_idxs]

	model.train()
	optimizer.zero_grad()

	outputs, h0, c0 = loaded_model(inputs, h0, c0)

	loss = loss_fn(outputs, labels)
	loss.backward()
	optimizer.step()

	h0 = h0.detach()
	c0 = c0.detach()
	mean_loss += loss.item()/num_data_samples
	if i%100 == 0:
		print(f'Loss: {loss.item():.4f}')

print(mean_loss)
