import os
from tqdm import tqdm
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

from prednet import PredNet        


def init_weights(m):
	""""
	init_weights:
	Initialize sub-module weights using xavier_uniform (for kernels) and 0. (for kernel biases)
	Arguments:
	_________
	m: nn.Module
	"""

	if isinstance(m, nn.Conv2d):
		# Reset kernels and biases
		nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			nn.init.zeros_(m.bias)


num_epochs = 50
batch_size = 10
A_channels = (1, 8, 16, 32)
R_channels = (1, 8, 16, 32)

# For PredNet-large
# A_channels = (1, 32, 64, 128)
# R_channels = (1, 32, 64, 128)



lr = 0.001 # if epoch < 75 else 0.0001
nt = 27  #5 # num of time steps
loss_mode = "L_0"
model_name ='prednet-27-8-4layer.pt'

if loss_mode == 'L_0':
	layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]).cuda())
elif loss_mode == 'L_all':
	layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.1], [0.1], [0.1]]).cuda())

time_loss_weights = 1./(nt - 1) * torch.ones(nt, 1)
time_loss_weights[0] = 0
time_loss_weights = Variable(time_loss_weights.cuda())

# Model parameters
loss_mode = 'L_0'
peephole = False
lstm_tied_bias = False
gating_mode = 'mul'
input_size = (128, 48)

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, X, output_mode='error'):
    self.X = X
    self.output_mode = output_mode
    self.p_max = np.max(X)
    self.p_min = np.min(X)

  def preprocess(self, X):
    return (X.astype(np.float32) - self.p_min) / self.p_max

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    pad = np.zeros((nt, 128, 2))
    X = np.concatenate([pad, self.X[index], pad], axis=-1)
    X = self.preprocess(X)
    return X



data_prefix = "train_data"
train_data = []
num_files = 0
for filename in tqdm(os.listdir(data_prefix)):
  curr_song = filename
  if (filename.endswith(".wav")): 
    num_files += 1
    file_location = "./" + data_prefix + "/" + filename
    y, sr = librosa.load(file_location, sr=None)
    complete_melSpec = librosa.feature.melspectrogram(y=y, sr=sr)
    complete_melSpec_db = librosa.power_to_db(complete_melSpec, ref=np.max)
    complete_melSpec_db_norm = (complete_melSpec_db * (255.0/80.0)) + 255.0
    complete_melSpec_db_norm = np.rot90(complete_melSpec_db_norm.copy(),2)
    for j in range(1): #11
      curr = []
      curr_x = 0
      WINDOW_SIZE = 44
      SHIFT = 8
      for i in range(nt):#5):
        melSpec_db_norm = complete_melSpec_db_norm[:,(curr_x):(curr_x+WINDOW_SIZE)]
        curr.append(melSpec_db_norm)
        curr_x += SHIFT
      if (len(curr) == nt): #5):
        train_data.append(np.asarray(curr))
print("Number of files:", num_files)
train_dataset = MyDataset(train_data)
train_loader_args = dict(shuffle = True, batch_size = 16, num_workers = 4, pin_memory = True)
train_loader = DataLoader(train_dataset, **train_loader_args)

model = PredNet(input_size, curr_song, R_channels, A_channels, model_name, output_mode='error', gating_mode=gating_mode,
                                peephole=peephole, lstm_tied_bias=lstm_tied_bias)
if torch.cuda.is_available():
    model.cuda()

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)
elif torch.cuda.device_count() == 1:
    print("Using 1 GPU.")
model.apply(init_weights)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss()

def lr_scheduler(optimizer, epoch):
    if epoch < num_epochs //2:
        return optimizer
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
        return optimizer
train_loss = 0.0
min_val_loss = float('inf')
target = torch.zeros(size=(1,))
for epoch in range(num_epochs):
    optimizer = lr_scheduler(optimizer, epoch)
    for i, inputs in enumerate(train_loader):
        inputs = Variable(inputs.cuda())
        target = Variable(target.cuda())
        new_shape = inputs.size() + (1,)
        inputs = inputs.view(new_shape)
        inputs = inputs.permute(0,1,4,2,3)
        errors = model(inputs) # batch x n_layers x nt
        loc_batch = errors.size(0)
        loss = torch.mm(errors.view(-1, nt), time_loss_weights) # batch*n_layers x 1
        loss = torch.mm(loss.view(loc_batch, -1), layer_loss_weights)
        loss = criterion(loss, target)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print('Epoch: {}/{}, step: {}, loss: {}'.format(epoch, num_epochs, i, loss.item()))


torch.save(model.state_dict(), 'models/{}'.format(model_name))
