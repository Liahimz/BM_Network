import torch
import numpy as np
from tmp_neuron import TMP2_SMorphLayer, TMP_SMorphLayer
from utils import *
import torch.nn as nn
from BM_Neuron import* 

from utility_layers import *


class BM_Net(nn.Module):

    def __init__(self, depth, shape):
        super().__init__()
        
        layers = []
        layers.append(BipolarMorphological2D_Torch(32, kernel_size = (3, 3), input_shape=shape))
        # layers.append(nn.Conv2d(1, 32, (3,3)))
        layers.append(nn.ReLU())
        out_shape = shape
        for i in range(depth):
            out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
            # print(out_shape)
            layers.append(BipolarMorphological2D_Torch(16, kernel_size = (3, 3), input_shape=out_shape))
            layers.append(nn.ReLU())
            
        out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
        # print(out_shape)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(np.prod(out_shape), 10))
        layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

size = [21632, 9216, 7744, 6400, 5184]
# size = [21632, 9216, 7744, 6400, 324]

class CNN_Net(nn.Module):

    def __init__(self, depth, shape):
        super().__init__()
        
        layers = []
        layers.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size = (3, 3)))
        # layers.append(Conv2d(in_channels=1, out_channels=32, kernel_size = (3, 3), layer = 1))
        layers.append(nn.ReLU())
        out_shape = 32
        for i in range(depth):
            layers.append(nn.Conv2d(in_channels=out_shape, out_channels=16, kernel_size = (3, 3)))
            # layers.append(Conv2d(in_channels=out_shape, out_channels=16, kernel_size = (3, 3), layer = i + 2))
            layers.append(nn.ReLU())
            out_shape = 16
            
        layers.append(nn.Flatten())
        layers.append(nn.Linear(np.prod(size[depth]), 10))
        layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # xk = torch.tensor(x, requires_grad=True)
        # xk.register_hook(get_hook('x'))
        return self.net(x)


class Smorph_Net(nn.Module):

    def __init__(self, depth, shape):
        super().__init__()
        
        layers = []
        # layers.append(SMorphLayer(filters=32, kernel_size = (3, 3), input_shape=shape, alpha=2.5, layer=1))
        layers.append(TMP_SMorphLayer(filters=32, kernel_size = (3, 3), input_shape=shape, alpha=2.5, layer=1))
        layers.append(nn.Tanh())
        out_shape = shape
        for i in range(depth):
            out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
            # layers.append(SMorphLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=2, layer=i + 2))
            layers.append(TMP_SMorphLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=2, layer=i + 2))
            layers.append(nn.Tanh())
            
        out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(np.prod(out_shape), 10))
        layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LnExpMax_Net(nn.Module):

    def __init__(self, depth, shape):
        super().__init__()
        
        layers = []
        layers.append(LnExpMaxLayer(filters=32, kernel_size = (3, 3), input_shape=shape, alpha=10))
        layers.append(nn.ReLU())
        out_shape = shape
        for i in range(depth):
            out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
            layers.append(LnExpMaxLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=10))
            layers.append(nn.ReLU())
            
        out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(np.prod(out_shape), 10))
        layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    

class Morph_Net(nn.Module):

    def __init__(self, depth, shape):
        super().__init__()

        coefs = []
        for i in reversed(range(depth + 1)):
            coefs.append(np.power(10, i))
       
        layers = []
        layers.append(MorphLayer(filters=32, kernel_size = (3, 3), input_shape=shape, grad_coef=coefs[0], layer=1))
        layers.append(nn.ReLU())
        out_shape = shape

        for i in range(depth):
            out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
            # print(out_shape)
            layers.append(MorphLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, grad_coef = coefs[i + 1], layer= i + 1))
            layers.append(nn.ReLU())
            
        out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(np.prod(out_shape), 10))
        # layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)