import torch
import numpy as np
from utils import *
import torch.nn as nn
from torch.autograd import Function
from utility_layers import *

class BM_layer_onehalf(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 1,
                 alpha = 1,
                 **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel_shape = self.kernel_size[0], self.kernel_size[1], self.in_channels, self.out_channels

        self.bias = torch.nn.Parameter(torch.zeros(self.out_channels, requires_grad=True))

        self.k = torch.nn.Parameter(torch.empty(self.kernel_shape, requires_grad=True))
        torch.nn.init.xavier_uniform_(self.k)

        self.delta_x = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.delta_w = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.ln = Ln()

        self.lnxmax = LnExp_Max(alpha=alpha)
        self.exp = Exp()

        self.input_bias = 1
        self.weight_bias = 5

    def forward(self, x):
        filter_height, filter_width, in_channels, out_channels = self.kernel_shape

        x1_pathces = x + self.input_bias
        x1_pathces = self.ln(x1_pathces)
        x1_pathces = extract_image_patches(x1_pathces, filter_height, self.stride)
        x1_pathces = torch.unsqueeze(x1_pathces, 2)
        k_for_patches = self.k.view(filter_height * filter_width * in_channels, out_channels) + self.weight_bias
        x1_k1 = x1_pathces + k_for_patches[None, :, :, None, None]
        y11 = self.exp(self.lnxmax(x1_k1))

        x_pathces = extract_image_patches(x, filter_height, self.stride)
        k_view = self.k.view(filter_height * filter_width * in_channels, out_channels)
        x_pathces = torch.sum(x_pathces, 1, keepdim=True) * self.delta_w
        k_view = (torch.sum(k_view, 0, keepdim=False) * self.delta_x)[None, :, None, None]
        
        y = y11 - x_pathces - k_view + self.bias[None, :, None, None]
        return y