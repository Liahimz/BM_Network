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

# from train import mnist, train, device


device = torch.device("cpu")

# model = Smorph_Net_MAE(3, (1, 28, 28), alpha=16)
# model.load_state_dict(torch.load('trained.pt'))

model = Smorph_Net(3, (1, 28, 28))
model.load_state_dict(torch.load('models/Smorph_3_23-06-2022_17:56:01_trained.pt'))
model.to(device)
save_featuremap(model, mnist, SMorphLayer, "Smorph", 1)

model = CNN_Net(3, (1, 28, 28))
model.load_state_dict(torch.load('models/CNN_3_23-06-2022_14:53:03_trained.pt'))
model.to(device)
save_featuremap(model, mnist, nn.Conv2d, "CNN", 1)
# print(model.state_dict())
# for key in model.state_dict():
#     print(key)

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
#     if 'weight'in key:
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

