from time import sleep
import torch
import numpy as np
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

from train import mnist, train, device
from vizualize_utils import PlotLine

class SMorphLayer_MAE(nn.Module):
    def __init__(self, filters,
                 input_shape,
                 kernel_size,
                 input_shift=1e-9,
                 padding='VALID',
                 strides=(1, 1),
                 alpha = 1,
                 layer = 1,
                 bins = None,
                 **kwargs):
        super().__init__()
        self.filters = filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.input_shift = input_shift
        self.padding = padding.upper()
        self.strides = strides
        self.kernel_shape = self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.filters

        self.k1 = torch.nn.Parameter(torch.empty(self.kernel_shape, requires_grad=True))
        self.k2 = torch.nn.Parameter(torch.empty(self.kernel_shape, requires_grad=True))
        self.bias = torch.nn.Parameter(torch.zeros(self.filters, requires_grad=True))


        # self.a = torch.nn.Parameter(torch.empty(self.kernel_shape, requires_grad=True))

        torch.nn.init.xavier_uniform_(self.k1)
        torch.nn.init.xavier_uniform_(self.k2)

        self.alpha = alpha #nn.Parameter(torch.tensor(1., requires_grad=True)) #
        # self.batch_norm = nn.BatchNorm2d(filters)
        self.layer = layer

        self.bins = bins

        self.mean = 0
        self.input = torch.nn.Parameter(torch.zeros(self.input_shape, requires_grad=False))
        # self.ln = Ln(min_val=1e-9)

        # self.add_k1 = Add(filters,
        #          input_shape,
        #          kernel_size)

        # self.add_k2 = Add(filters,
        #          input_shape,
        #          kernel_size)

        # # self.exp = Exp()
        
        self.smax = Smooth_Max(alpha=self.alpha, layer=self.layer)

    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            return (self.kernel_shape[3], (self.input_shape[1] - self.kernel_shape[1] + 1) // self.strides[0],
                   (self.input_shape[2] - self.kernel_shape[0] + 1) // self.strides[1])
        else:
            
            return (self.kernel_shape[3], self.input_shape[1] // self.strides[0], self.input_shape[2] // self.strides[1])

    def forward(self, x):
        filter_height, filter_width, in_channels, out_channels = self.kernel_shape

        self.input = torch.nn.Parameter(x)

        x1_pathces = extract_image_patches(x, filter_height, self.strides[0])       
        x2_pathces = -x1_pathces


        # x1_k1 = self.add_k1(x1_pathces)
        
        x1_pathces = torch.unsqueeze(x1_pathces, 2)
        k_for_patches_1 = self.k1.view(filter_height * filter_width * in_channels, out_channels)
        x1_k1 = x1_pathces + k_for_patches_1[None, :, :, None, None]


        # x2_k2 = self.add_k2(x2_pathces)

        x2_pathces = torch.unsqueeze(x2_pathces, 2)
        k_for_patches_2 = self.k2.view(filter_height * filter_width * in_channels, out_channels)
        x2_k2 = x2_pathces + k_for_patches_2[None, :, :, None, None]

        
       
        # y11 = (self.smax(x1_k1))
        ax_1 = torch.exp(torch.mul(x1_k1, self.alpha))
        s_max_1 = torch.mul(x1_k1, ax_1 / torch.sum(ax_1, dim=1, keepdims=True))
        y11 = torch.sum(s_max_1, dim=1)


        # y21 = (self.smax(x2_k2))
        ax_2 = torch.exp(torch.mul(x2_k2, self.alpha))
        s_max_2 = torch.mul(x2_k2, ax_2 / torch.sum(ax_2, dim=1, keepdims=True))
        y21 = torch.sum(s_max_2, dim=1)



        # a = torch.nn.Parameter(x1_k1)
        # b = torch.nn.Parameter(x1_k1)

        a = torch.empty(x2_pathces.size())
        # a.cauchy_(sigma=1)
        torch.nn.init.uniform_(a, a=-4.0, b=4.0)
        # hist = torch.histogram(a, bins=50)

        # plt.rcParams.update({'font.size': 6})
        # fig, ax = plt.subplots()

        # counts = hist[0]#np.array([20, 19, 40, 46, 58, 42, 23, 10, 8, 2])
        # bin_edges = hist[1] #np.array([0.5, 0.55, 0.59, 0.63, 0.67, 0.72, 0.76, 0.8, 0.84, 0.89, 0.93])

        # ax.bar(x=bin_edges[:-1], height=counts, width=np.diff(bin_edges), align='edge', fc='skyblue', ec='black')
        # ax.set_xticks((bin_edges[:-1] + bin_edges[1:]) / 2)
        # fig.set_size_inches(25.5, 15.5)
        # plt.savefig('input_a' + '.png')
        # plt.close(fig)

        a_max = a.max(1)[0]
        a_11 = self.smax(a)
        mae = ((a_max - a_11).abs()).sum() / torch.numel(a_11)
        self.bins = mae
        self.mean = torch.mean(a_max.abs())
        # print(a.max())
        # exit(0)
        y = y11 + y21 + self.bias[None, :, None, None]
        return y


class LnExpMaxLayerMAE(nn.Module):
    def __init__(self, filters,
                 input_shape,
                 kernel_size,
                 input_shift=1e-9,
                 padding='VALID',
                 strides=(1, 1),
                 alpha = 1,
                 layer = 1,
                 **kwargs):
        super().__init__()
        self.filters = filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.input_shift = input_shift
        self.padding = padding.upper()
        self.strides = strides
        self.kernel_shape = self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.filters

        self.bias = torch.nn.Parameter(torch.zeros(self.filters, requires_grad=True))

        self.alpha =  alpha #nn.Parameter(torch.tensor(2.5, requires_grad=True)) #

        self.batch_norm = nn.BatchNorm2d(filters)

        self.add_k1 = Add(filters,
                 input_shape,
                 kernel_size)
        
        self.add_k2 = Add(filters,
                 input_shape,
                 kernel_size)

        self.lnxmax = LnExp_Max(alpha=self.alpha, layer=layer)

        self.bins = 0
        self.mean = 0

        # self.lmax = L_Max(alpha=self.alpha, layer=layer)

    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            return (self.kernel_shape[3], (self.input_shape[1] - self.kernel_shape[1] + 1) // self.strides[0],
                   (self.input_shape[2] - self.kernel_shape[0] + 1) // self.strides[1])
        else:
            
            return (self.kernel_shape[3], self.input_shape[1] // self.strides[0], self.input_shape[2] // self.strides[1])

    def forward(self, x):
        filter_height, filter_width, in_channels, out_channels = self.kernel_shape
        # print(x)
        x1_pathces = extract_image_patches(x, filter_height, self.strides[0])
        x2_pathces = -x1_pathces # extract_image_patches(-x, filter_height, self.strides[0]) #
        x1_k1 = self.add_k1(x1_pathces)
        x2_k2 = self.add_k2(x2_pathces)
        y11 = (self.lnxmax(x1_k1))
        y22 = (self.lnxmax(x2_k2))


        a = torch.empty(x2_pathces.size())
        torch.nn.init.uniform_(a, a=-4.0, b=4.0)
        a_lnmax = self.lnxmax(a)

        a_max = a.max(1)[0]
        mae = ((a_max - a_lnmax).abs()).sum() / torch.numel(a_lnmax)
        self.bins = mae
        self.mean = torch.mean(a_max.abs())

        y = y11 + y22 + self.bias[None, :, None, None]
        return y


class LSE_Net_MAE(nn.Module):

    def __init__(self, depth, shape, alpha):
        super().__init__()
        
        layers = []
        # layers.append(SMorphLayer(filters=32, kernel_size = (3, 3), input_shape=shape, alpha=2.5, layer=1))
        layers.append(LnExpMaxLayerMAE(filters=32, kernel_size = (3, 3), input_shape=shape, alpha=alpha, layer=alpha))
        # layers.append(nn.Tanh())
        layers.append(nn.ReLU())
        out_shape = shape
        for i in range(depth):
            out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
            # layers.append(SMorphLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=2, layer=i + 2))
            layers.append(LnExpMaxLayerMAE(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=2, layer=alpha))
            # layers.append(nn.Tanh())
            layers.append(nn.ReLU())
            
        out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(np.prod(out_shape), 10))
        layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Smorph_Net_MAE(nn.Module):

    def __init__(self, depth, shape, alpha):
        super().__init__()
        
        layers = []
        # layers.append(SMorphLayer(filters=32, kernel_size = (3, 3), input_shape=shape, alpha=2.5, layer=1))
        layers.append(SMorphLayer_MAE(filters=32, kernel_size = (3, 3), input_shape=shape, alpha=alpha, layer=1, bins = 0))
        # layers.append(nn.Tanh())
        layers.append(nn.ReLU())
        out_shape = shape
        for i in range(depth):
            out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
            # layers.append(SMorphLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=2, layer=i + 2))
            layers.append(SMorphLayer_MAE(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=2, layer=i + 2, bins = 0))
            # layers.append(nn.Tanh())
            layers.append(nn.ReLU())
            
        out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(np.prod(out_shape), 10))
        layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_GETMAE(alpha, epochs, batch_size=100, name="BM_NET",):
    stats = []
    mean = []
    acc = []
    d = 0
    for a in alpha:
        model = LSE_Net_MAE(d, (1, 28, 28), alpha=a)
       
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


        train_dataloader, test_dataloader = mnist(batch_size)

        train_loss, train_accuracy = train(model, train_dataloader, test_dataloader,
                                    criterion, optimizer, None,
                                    n_epochs=epochs, show=False)
        
        # torch.save(model.state_dict(), 'trained.pt')
        # torch.save(model, 'trained_model.pt')

        # acc.append(train_accuracy)
        
        print(model.net[(d+1) * 2 - 2].bins)
        if (np.isnan(model.net[(d+1) * 2 - 2].bins.item())):
            stats.append(10)
        else:
            stats.append(model.net[(d+1) * 2 - 2].bins.item())

        if (np.isnan(model.net[(d+1) * 2 - 2].mean.item())):
            mean.append(10)
        else:
            mean.append(model.net[(d+1) * 2 - 2].mean.item())

       
    PlotLine(alpha, stats, "lse_max_uniform_mae_lvl")

    PlotLine(alpha, mean, "lse_max_uniform_mean")

    # fig = plt.figure()
    # plt.plot(alpha, stats)
    # plt.savefig('mae' + str(alpha[0]) + '_' + str(alpha[-1]) + '.png')
    # plt.close(fig)

    # fig = plt.figure()
    # plt.plot(alpha, mean)
    # plt.savefig('mean' + str(alpha[0]) + '_' + str(alpha[-1]) + '.png')
    # plt.close(fig)

    # fig = plt.figure()
    # plt.plot(alpha, acc)
    # plt.savefig('acc' + str(alpha[0]) + '_' + str(alpha[-1]) + '.png')
    # plt.close(fig)
    return stats


train_GETMAE(alpha=np.arange(1, 45), epochs=2, batch_size=50, name="LSE_alpha")