import torch
import numpy as np
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


class Smorph_Net(nn.Module):

    def __init__(self, depth, shape):
        super().__init__()
        
        layers = []
        layers.append(SMorphLayer(32, kernel_size = (3, 3), input_shape=shape))
        layers.append(nn.ReLU())
        out_shape = shape
        for i in range(depth):
            out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
            layers.append(SMorphLayer(16, kernel_size = (3, 3), input_shape=out_shape))
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
        layers.append(MorphLayer(filters=32, kernel_size = (3, 3), input_shape=shape, grad_coef=coefs[0]))
        layers.append(nn.ReLU())
        out_shape = shape

        for i in range(depth):
            out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
            # print(out_shape)
            layers.append(MorphLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, grad_coef = coefs[i + 1]))
            layers.append(nn.ReLU())
            
        out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(np.prod(out_shape), 10))
        # layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)