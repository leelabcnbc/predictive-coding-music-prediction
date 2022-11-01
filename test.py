import matplotlib

matplotlib.use('Agg')
import os

#import hickle as hkl
import librosa
import torch.nn as nn
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy
import soundfile as sf
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import shutil
from sklearn.metrics import mean_squared_error
from prednet import PredNet


n_plot = 1 # number of plot to make (must be <= batch_size)

removenum = 0

nt = 49
wid = 432
shift=8
model_name = 'prednet-27-8-4layer-3s.pt'
model_type = 'models/' + model_name
folder = 'test_results/' + model_name + '/plots/'

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




names=[]
mse_full=[]
mse_seq=[]
mses=[]
filen = ""
# Model parameters
gating_mode = 'mul'
peephole = False
lstm_tied_bias = False

A_channels = (1, 8, 16, 32)
R_channels = (1, 8, 16, 32)

data_prefix = "test_data/"

rng = range(0,1)



# If intending to test with a resolution of one time-unit for each timelapse, use 
# rng = range(-1, 9) and see "results.xlsx" spreadsheet for formatting the output
#rng = range(-1, 9)
mses = []
mse_seq = []
idx = 0
for hint in rng:

		for filename in sorted(os.listdir(data_prefix)):
			curr_song = filename
			test_data = []
			num_files = 0
			if filename.endswith(".wav"): 
				file_location = "./" + data_prefix + "/" + filename
				num_files += 1
				y, sr = librosa.load(file_location, sr=None)
				filen = filename
				complete_melSpec = librosa.feature.melspectrogram(y=y, sr=sr, window=scipy.signal.hanning)
				complete_melSpec_db = librosa.power_to_db(complete_melSpec, ref=np.max)
				complete_melSpec_db_norm = (complete_melSpec_db * (255.0/80.0)) + 255.0
				complete_melSpec_db_norm_rot = np.rot90(complete_melSpec_db_norm.copy(),2)
				complete_melSpec_db_norm = torch.unsqueeze(torch.from_numpy(complete_melSpec_db_norm_rot.copy()),0)
				if hint >= 0:
					complete_melSpec_db_norm  = np.delete(complete_melSpec_db_norm, slice(0,hint), 2)
				if hint < 0:
					##if hint == -1:
					complete_melSpec_db_norm  = np.delete(complete_melSpec_db_norm, slice(-1+hint, -1), 2) #hacky, fix later
				    
				padmel = torch.zeros((1, 128, abs(hint)))
				padmel2 = torch.zeros((1, 128, 1))
				if hint >= 0:
					complete_melSpec_db_norm = torch.cat((complete_melSpec_db_norm, padmel), dim=2)
				else:
					complete_melSpec_db_norm = torch.cat((padmel, complete_melSpec_db_norm), dim=2)
				if complete_melSpec_db_norm.shape[2] != 432:
					complete_melSpec_db_norm = torch.cat((padmel2, complete_melSpec_db_norm), dim=2)
				complete_melSpec_db_norm_squeezed = complete_melSpec_db_norm.squeeze()
				sav_direc = 'test_results/' + model_name + '/plots/full_mel_spec/'
				if not os.path.exists(sav_direc):
					os.makedirs(sav_direc)
				plt.imsave(fname='test_results/' + model_name + '/plots/full_mel_spec/full_mel_final-{}.png'.format(filename), arr=complete_melSpec_db_norm_squeezed,  format="png", cmap='gray')
				nt = int(((wid-48)/shift)+1)
				for j in range(1): #20
					curr = []
					curr_x = 0
					WINDOW_SIZE = 44
					SHIFT = shift
					for i in range(nt):#49):
						melSpec_db_norm = complete_melSpec_db_norm[0,:,curr_x:(curr_x+WINDOW_SIZE)].numpy()
						curr.append(melSpec_db_norm)
						curr_x += SHIFT
					if (len(curr) == nt): #49):
						test_data.append(np.asarray(curr))
				test_dataset = MyDataset(test_data)
				test_loader_args = dict(shuffle = False, batch_size = 1, num_workers = 4, pin_memory = True)
				test_loader = DataLoader(test_dataset, **test_loader_args)
				input_size = (128, 48)
				curr_song = filename
				print(curr_song)
				model = PredNet(input_size, curr_song, R_channels, A_channels, model_name, output_mode='prediction', gating_mode=gating_mode,
								peephole=peephole, lstm_tied_bias=lstm_tied_bias)
				model.load_state_dict(torch.load(model_type))
				if torch.cuda.is_available():
					print('Using GPU.')
					model.cuda()
				pred_MSE = 0.0
				copy_last_MSE = 0.0
				for step, inputs in enumerate(test_loader):
					# ---------------------------- Test Loop ----------------------------
					inputs = inputs.cuda() # batch x time_steps x channel x width x height
					new_shape = inputs.size() + (1,)
					inputs = inputs.view(new_shape)
					inputs = inputs.permute(0,1,4,2,3)
					inputs = inputs.cpu()
					pred = model(inputs) # (batch_size, channels, width, height, nt)
					pred = pred.cpu()
					pred = pred.permute(0,4,1,2,3) # (batch_size, nt, channels, width, height)
					if step == len(test_loader)-1:
						inputs = inputs.detach().numpy() * 255.
						pred = pred.detach().numpy() * 255.

						inputs = np.transpose(inputs, (0, 1, 3, 4, 2))
						pred = np.transpose(pred, (0, 1, 3, 4, 2))
						
						inputs = inputs.astype(int)
						pred = pred.astype(int)
						plot_idx = np.random.permutation(inputs.shape[0])[:n_plot]

						aspect_ratio = float(pred.shape[2]) / pred.shape[3]
						fig = plt.figure(figsize = (nt, 2*aspect_ratio))
						gs = gridspec.GridSpec(2, nt)
						gs.update(wspace=0., hspace=0.)
						plot_save_dir = folder + "prediction_plots-{}/".format(shift)
						if not os.path.exists(plot_save_dir): os.makedirs(plot_save_dir)
						plot_idx = np.random.permutation(inputs.shape[0])[:n_plot]				
						names.append(filename + "-" + str(hint))
						for i in plot_idx:
							for t in range(nt):
								tmp = inputs[i,t]
								tmp = np.squeeze(tmp)
								direc = plot_save_dir + filename + "/" + str(removenum) + "/"
								if not os.path.exists(direc): os.makedirs(direc)
								plt.imsave(fname=direc + "orig" + str(t) + ".png", arr=tmp, cmap="gray", format="png")
								tmp2 = pred[i,t]
								tmp2 = np.squeeze(tmp2)
								if t == 1:
									predvals = tmp2[:,48-6]
								direc = plot_save_dir + filename + "/" + str(removenum) + "/"
								if not os.path.exists(direc): os.makedirs(direc)
								plt.imsave(fname=direc + "pred" + "-" + str(t) + ".png", arr=tmp2, cmap="gray", format="png")
								mse_curr = mean_squared_error(tmp, tmp2)	
								#Or, normalize first and compute from the (0,1) pixel intensity:
								#mse_curr = mean_squared_error(tmp/255, tmp2/255)
								mse_seq.append(mse_curr.copy())
							print('Filename: ' + str(filename) + "   pix shift: " + str(hint))
							print('MSE: ' + str(sum(mse_seq[1:])/len(mse_seq)-1))
				mse_full.append(mse_seq.copy())
				mse_seq = []
mses.append(names)
mses.append(mse_full.copy())
np.save('test_results/' + model_name + '/mses', mses)


