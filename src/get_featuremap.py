# import imp
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
# from train import mnist, train, device


device = torch.device("cpu")

# model = Smorph_Net_MAE(3, (1, 28, 28), alpha=16)
# model.load_state_dict(torch.load('trained.pt'))

ans = []

model = MorphSmax_NetBiased(0, (1, 28, 28))
model.load_state_dict(torch.load('models/BM_Net_Smax_1.5line_200_1_07-07-2022_18:28:48_trained.pt'))
model.to(device)
ans.append(save_featuremap(model, mnist, BMLayer_Smax_Biased, "SoftMax_BM_biased", 5, 6))

# model1 = MorphSmax_Net(0, (1, 28, 28))
# model1.load_state_dict(torch.load('models/BM_Net_Smax_0_05-07-2022_15:05:06_trained.pt'))
# model1.to(device)
# ans.append(save_featuremap(model1, mnist, BMLayer_Smax, "SoftMax_BM", 5, 6))

model1 = Morph_Net(0, (1, 28, 28))
model1.load_state_dict(torch.load('models/BM_Net_0_05-07-2022_14:48:33_trained.pt'))
model1.to(device)
ans.append(save_featuremap(model1, mnist, MorphLayer, "BM", 5, 6))


model2 = CNN_Net(0, (1, 28, 28))
model2.load_state_dict(torch.load('models/KDCNN_0_27-06-2022_13:36:10_trained.pt'))
model2.to(device)
ans.append(save_featuremap(model2, mnist, nn.Conv2d, "CNN", 5, 6))


min_v = 10
max_v = -10
# add_const = 0.77
for processed, names in ans:
    for item in processed:
        # item += add_const
        if item.min() < min_v:
            min_v = item.min()
        if item.max() > max_v:
            max_v = item.max()

print(min_v, max_v)


colors = plt.cm.inferno(np.linspace(min_v, max_v, 15))
mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

fig = plt.figure(figsize=(50, 50))
for j in range(len(ans)):
    processed, names = ans[j]
    print(names)
    for i in range(len(processed)):
        a = fig.add_subplot(3, 6, 6 * j + i + 1)
        # processed[i] = (processed[i] - min_v) / (max_v - min_v)
        processed[i][0][0] = min_v
        processed[i][0][1] = max_v
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        bar = plt.colorbar(orientation='horizontal')
        bar.ax.tick_params(labelsize=20) 
        a.set_title(names[i].split('(')[0], fontsize=20)
plt.savefig("SMAXBMBiased_BM_CNN", bbox_inches='tight')
plt.close(fig)


