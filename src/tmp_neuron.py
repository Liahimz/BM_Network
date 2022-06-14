import re
from tkinter import N
from tkinter.messagebox import NO
import torch
import numpy as np
from utils import *
import torch.nn as nn
from torch.autograd import Function
from utility_layers import *
import torch.nn.functional as F

def layer_func(kernel_shape, stride, alpha):
    class layer_func(Function):
        @staticmethod
        def forward(ctx, x, k1, k2, bias):
            filter_height, filter_width, in_channels, out_channels = kernel_shape
            x1_pathces = extract_image_patches(x, filter_height, stride)
            x2_pathces = extract_image_patches(-x, filter_height, stride)


            x1_pathces = torch.unsqueeze(x1_pathces, 2)
            k1_for_patches = k1.view(filter_height * filter_width * in_channels, out_channels)
            x1_k1 = x1_pathces + k1_for_patches[None, :, :, None, None]


            x2_pathces = torch.unsqueeze(x2_pathces, 2)
            k2_for_patches = k2.view(filter_height * filter_width * in_channels, out_channels)
            x2_k2 = x2_pathces + k2_for_patches[None, :, :, None, None]

        
            eax11 = torch.exp(torch.mul(x1_k1, alpha))
            s11 = torch.sum(eax11, dim=1)
            s10 = torch.sum(x1_k1 * eax11, dim=1)
            y11 = s10 / s11

            eax22 = torch.exp(torch.mul(x2_k2, alpha))
            s21 = torch.sum(eax22, dim=1)
            s20 = torch.sum(x2_k2 * eax22, dim=1)
            y22 = s20 / s21

            ctx.save_for_backward(x, k1, k2, x1_k1, x2_k2, eax11, eax22, s10, s11, s20, s21)

            y = y11 + y22 + bias[None, :, None, None]
            return y

        @staticmethod
        def backward(ctx, y_grad):
            x, k1, k2, x1_k1, x2_k2, eax11, eax22, s10, s11, s20, s21 = ctx.saved_tensors
            # print(x1_k1.shape)
            # print(eax11.shape)
            # print(s10.shape)
            # print(s11.shape)
            y1 = ((eax11 + x1_k1 * alpha * eax11) * torch.unsqueeze(s11, 1) - alpha * eax11 * torch.unsqueeze(s10, 1)) / torch.unsqueeze((s11 * s11), 1)

            # print(x2_k2.shape)
            # print(eax22.shape)
            # print(s20.shape)
            # print(s21.shape)
            y2 = ((-eax22 - x2_k2 * alpha * eax22) * torch.unsqueeze(s21, 1) + alpha * eax22 * torch.unsqueeze(s20, 1)) / torch.unsqueeze((s21 * s21), 1)
            
            x = (y1+y2) * torch.unsqueeze(y_grad, 1)
            print("x ", x.shape)
            tmp = x.sum(2)
            print("tmp ",tmp.shape)

            # # tmp = tmp[:, None, None, :, :, :]
            # print((tmp.shape[0], kernel_shape[0], kernel_shape[1], tmp.shape[1] // (kernel_shape[0] * kernel_shape[1]), tmp.shape[2], tmp.shape[3]))
            # tmp = tmp.view(tmp.shape[0], kernel_shape[0], kernel_shape[1], tmp.shape[1] // (kernel_shape[0] * kernel_shape[1]), tmp.shape[2], tmp.shape[3])
            # print(tmp.shape)

            # tmp = tmp.sum(0)
            print(tmp.shape)
            out_size = tmp.shape[1] // (kernel_shape[0] * kernel_shape[1])
            tmp = F.fold(tmp, out_size, kernel_shape[0])
            print(tmp.shape)
            return tmp, (y1) * torch.unsqueeze(y_grad, 1), (y2) * torch.unsqueeze(y_grad, 1), y_grad
    return layer_func

class TMP_SMorphLayer(nn.Module):
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
        self.alpha = alpha #nn.Parameter(torch.tensor(1., requires_grad=True)) #
        # self.batch_norm = nn.BatchNorm2d(filters)
        self.layer = layer
        # self.ln = Ln(min_val=1e-9)

        # self.add_k1 = Add(filters,
        #          input_shape,
        #          kernel_size)

        # self.add_k2 = Add(filters,
        #          input_shape,
        #          kernel_size)

        
        # self.smax = Smooth_Max(alpha=self.alpha, layer=self.layer)

        self.k1 = torch.nn.Parameter(torch.empty(self.kernel_shape, requires_grad=True))
        torch.nn.init.xavier_uniform_(self.k1)

        self.k2 = torch.nn.Parameter(torch.empty(self.kernel_shape, requires_grad=True))
        torch.nn.init.xavier_uniform_(self.k2)

        self.compute = layer_func(kernel_shape=self.kernel_shape, stride=self.strides[0], alpha=self.alpha)

    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            return (self.kernel_shape[3], (self.input_shape[1] - self.kernel_shape[1] + 1) // self.strides[0],
                   (self.input_shape[2] - self.kernel_shape[0] + 1) // self.strides[1])
        else:
            
            return (self.kernel_shape[3], self.input_shape[1] // self.strides[0], self.input_shape[2] // self.strides[1])

    def forward(self, x):
        return self.compute.apply(x, self.k1, self.k2, self.bias)


class TMP_Add(nn.Module):
    def __init__(self, filters,
                 input_shape,
                 kernel_size,
                 grad_coef = 1e-2,
                 layer = 1,
                 **kwargs):
        super().__init__()
        self.filters = filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.kernel_shape = self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.filters

        self.k = torch.nn.Parameter(torch.empty(self.kernel_shape, requires_grad=True))
        torch.nn.init.xavier_uniform_(self.k)

        self.add_forward = add_func(self.kernel_shape, grad_coef, layer)
        
        
    def forward(self, x):
        # if USE_AUTOGRAD:
            filter_height, filter_width, in_channels, out_channels = self.kernel_shape
            batch_size,c,h,w = x.shape
            # x = x.unfold(2, self.kernel_shape[0], 1).unfold(3, self.kernel_shape[1], 1)
            # # print("patches ", x.shape)
            # x = x.permute(0,4,5,1,2,3)
            # # print("patches ", x.shape)
            # x = x.reshape(batch_size, -1, x.shape[-2], x.shape[-1])
            x = torch.unsqueeze(x, 2)
            k_for_patches = self.k.view(filter_height * filter_width * in_channels, out_channels)
            # print("x ", x.shape)
            # print("k ", self.k.shape)
            # print(k_for_patches[None, :, :, None, None].shape)
            xk = x + k_for_patches[None, :, :, None, None]
            return xk
        # return self.add_forward.apply(x, self.k)

class TMP2_SMorphLayer(nn.Module):
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
        self.alpha = alpha #nn.Parameter(torch.tensor(1., requires_grad=True)) #
        # self.batch_norm = nn.BatchNorm2d(filters)
        self.layer = layer
        # self.ln = Ln(min_val=1e-9)

        self.add_k1 = TMP_Add(filters,
                 input_shape,
                 kernel_size)

        self.add_k2 = TMP_Add(filters,
                 input_shape,
                 kernel_size)

        # self.exp = Exp()
        
        self.smax = Smooth_Max(alpha=self.alpha, layer=self.layer)

    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            return (self.kernel_shape[3], (self.input_shape[1] - self.kernel_shape[1] + 1) // self.strides[0],
                   (self.input_shape[2] - self.kernel_shape[0] + 1) // self.strides[1])
        else:
            
            return (self.kernel_shape[3], self.input_shape[1] // self.strides[0], self.input_shape[2] // self.strides[1])

    def forward(self, x):

        x1_k1 = self.add_k1(x)
        x2_k2 = self.add_k2(-x)
    

        y11 = (self.smax(x1_k1))
        y22 = (self.smax(x2_k2))
       
        y = y11 + y22 + self.bias[None, :, None, None]
        return y