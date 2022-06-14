import torch
import numpy as np
from utils import *
import torch.nn as nn
from BM_Neuron import* 

from utility_layers import *

class Smorph_Net(nn.Module):

    def __init__(self, depth, shape):
        super().__init__()
        
        layers = []
        layers.append(SMorphLayer(32, kernel_size = (3, 3), input_shape=shape, alpha=2.5))
        layers.append(nn.ReLU())
        out_shape = shape
        for i in range(depth):
            out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
            layers.append(SMorphLayer(16, kernel_size = (3, 3), input_shape=out_shape, alpha=2))
            layers.append(nn.ReLU())
            
        out_shape = layers[-2].compute_output_shape(input_shape=out_shape)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(np.prod(out_shape), 10))
        layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)