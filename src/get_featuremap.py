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


# fig = plt.figure(figsize=(30, 30))
# import matplotlib.image as mpimg
# img1 = mpimg.imread('KD_5_feature_maps.png')
# img2 = mpimg.imread('KD_9_feature_maps.png')
# diff = img1[1:, :, :] - img2
# plt.imshow(diff)
# plt.savefig("diff", bbox_inches='tight')
# plt.close(fig)
# # print(model.state_dict())
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
