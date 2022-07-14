from statistics import mode
from time import sleep
import torch
import numpy as np
from train import mnist
from utils import *
import torch.nn as nn
from BM_Neuron import* 
from net import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import torch.utils.tensorboard as tb
from torch_tensorboard import *
from os import path
from datetime import datetime

import matplotlib.colors as mcolors

device = torch.device("cpu")


model = MorphSmax_NetBiased(0, (1, 28, 28))
model.load_state_dict(torch.load('models/BM_Net_1.5TEST_200_learned_all_delta_1_07-07-2022_21:50:00_trained.pt'))
model.to(device)

for key in model.state_dict():
    print(key)
    if "delta_x" in key:
        print(model.state_dict()[key])
    
    if "delta_w" in key:
        print(model.state_dict()[key])

# hist = torch.histogram(x, bins=50)

# fig, ax = plt.subplots()

# counts = hist[1]#np.array([20, 19, 40, 46, 58, 42, 23, 10, 8, 2])
# bin_edges = hist[0] #np.array([0.5, 0.55, 0.59, 0.63, 0.67, 0.72, 0.76, 0.8, 0.84, 0.89, 0.93])

# ax.bar(x=bin_edges[:-1], height=counts, width=np.diff(bin_edges), align='edge', fc='skyblue', ec='black')
# ax.set_xticks((bin_edges[:-1] + bin_edges[1:]) / 2)
# fig.set_size_inches(25.5, 15.5)
# plt.savefig('input_' + str(self.layer) + '.png')
# plt.close(fig)
# total_hist = 
# for key in model.state_dict():
#     # print(key)
#     if 'add_k2.k'in key:
#         # print(key)
#         # print(model.state_dict()[key])
#         hist = torch.histogram(model.state_dict()[key], bins=50)

#         plt.rcParams.update({'font.size': 6})
#         fig, ax = plt.subplots()

#         counts = hist[0]#np.array([20, 19, 40, 46, 58, 42, 23, 10, 8, 2])
#         bin_edges = hist[1] #np.array([0.5, 0.55, 0.59, 0.63, 0.67, 0.72, 0.76, 0.8, 0.84, 0.89, 0.93])

#         ax.bar(x=bin_edges[:-1], height=counts, width=np.diff(bin_edges), align='edge', fc='skyblue', ec='black')
#         ax.set_xticks((bin_edges[:-1] + bin_edges[1:]) / 2)
#         fig.set_size_inches(25.5, 15.5)
#         plt.savefig('weight' + str(key) + '.png')
#         plt.close(fig)