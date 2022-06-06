import torch
import numpy as np
from utils import *
import torch.nn as nn
from torch.autograd import Function

def get_hook(name):
    def print_hook(grad):
        print(f'{name} = {torch.mean(grad)}, {torch.std(grad)}')
        return grad
    return print_hook

def ln_func(kernel_shape, input_shift, strides):
    class ln_func(Function):
        @staticmethod
        def forward(ctx, x):
            filter_height, filter_width, in_channels, out_channels = kernel_shape
            batch_size,c,h,w = x.shape
            
            patches = x.unfold(2, filter_height, strides[0]).unfold(3, filter_width, strides[0])
            patches = patches.permute(0,4,5,1,2,3)
            x1_patches = patches.reshape(batch_size,-1,patches.shape[-2], patches.shape[-1])

            x1_patches = torch.clamp(x1_patches, min=input_shift)
            x1_patches = torch.log(x1_patches)
            x1_patches =  torch.unsqueeze(x1_patches, 2)
            ctx.save_for_backward(x1_patches, x)
            return x1_patches

        @staticmethod
        def backward(ctx, y_grad):
            x1_patches, x = ctx.saved_tensors
            # print(y_grad.shape)
            # print(x.shape)
            # print(x1_patches.shape)
            return 1/x
    return ln_func

def add_func(kernel_shape):
    class add_func(Function):
        @staticmethod
        def forward(ctx, x, k):
            filter_height, filter_width, in_channels, out_channels = kernel_shape
            k_for_patches = k.view(filter_height * filter_width * in_channels, out_channels)
            xk = x + k_for_patches[None, :, :, None, None]
            ctx.save_for_backward(xk, x, k)
            return xk

        @staticmethod
        def backward(ctx, y_grad):
            xk, x, k = ctx.saved_tensors
            return torch.ones(x.shape, requires_grad=True), torch.ones(k.shape, requires_grad=True)
    return add_func

def exp_func():
    class exp_func(Function):
        @staticmethod
        def forward(ctx, x):
            y = torch.exp(x.max(1)[0])
            ctx.save_for_backward(y, x)
            return y

        @staticmethod
        def backward(ctx, y_grad):
            y, x = ctx.saved_tensors
            return torch.exp(x)
    return exp_func

def smax_func(alpha):
    class exp_func(Function):
        @staticmethod
        def forward(ctx, x):
            ax = torch.exp(torch.mul(x, alpha))
            s_max = torch.mul(x, ax / torch.sum(ax, dim=1, keepdims=True))
            return torch.sum(s_max, dim=1)

        @staticmethod
        def backward(ctx, y_grad):
            return None
    return smax_func

class Ln(nn.Module):
    def __init__(self, filters,
                 input_shape,
                 kernel_size,
                 input_shift=1e-9,
                 padding='VALID',
                 strides=(1, 1),
                 **kwargs):
        super().__init__()
        self.filters = filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.input_shift = input_shift
        self.padding = padding.upper()
        self.strides = strides
        self.kernel_shape = self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.filters

        self.ln_forward = ln_func(self.kernel_shape, self.input_shift, self.strides)
        
    def forward(self, x):
        filter_height, filter_width, in_channels, out_channels = self.kernel_shape
        batch_size,c,h,w = x.shape
        
        patches = x.unfold(2, filter_height, self.strides[0]).unfold(3, filter_width, self.strides[0]).requires_grad_(True)
        patches = patches.permute(0,4,5,1,2,3).contiguous().requires_grad_(True)
        x1_patches = patches.view(batch_size,-1,patches.shape[-2], patches.shape[-1]).requires_grad_(True)

        x1_patches = torch.clamp(x1_patches, min=self.input_shift).requires_grad_(True)
        x1_patches = torch.log(x1_patches).requires_grad_(True)
        x1_patches =  torch.unsqueeze(x1_patches, 2).requires_grad_(True)
        # h = x1_patches.register_hook(get_hook('ln_x'))
        return x1_patches
        # return self.ln_forward.apply(x)

class Add(nn.Module):
    def __init__(self, filters,
                 input_shape,
                 kernel_size,
                 input_shift=1e-9,
                 padding='VALID',
                 strides=(1, 1),
                 **kwargs):
        super().__init__()
        self.filters = filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.input_shift = input_shift
        self.padding = padding.upper()
        self.strides = strides
        self.kernel_shape = self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.filters

        self.k = torch.nn.Parameter(torch.empty(self.kernel_shape, requires_grad=True))
        torch.nn.init.xavier_uniform_(self.k)

        self.add_forward = add_func(self.kernel_shape)
        
        
    def forward(self, x):
        filter_height, filter_width, in_channels, out_channels = self.kernel_shape
        # k_for_patches = torch.reshape(self.k, [filter_height * filter_width * in_channels, out_channels])
        k_for_patches = self.k.view(filter_height * filter_width * in_channels, out_channels)
        xk = x + k_for_patches[None, :, :, None, None]
        # xk.register_hook(get_hook('x+k'))
        return xk
        # return self.add_forward.apply(x, self.k)
        


class Exp(nn.Module):
    def __init__(self, filters,
                 input_shape,
                 kernel_size,
                 input_shift=1e-9,
                 padding='VALID',
                 strides=(1, 1),
                 **kwargs):
        super().__init__()
        self.filters = filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.input_shift = input_shift
        self.padding = padding.upper()
        self.strides = strides
        self.kernel_shape = self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.filters

        self.exp_forward = exp_func()

    def forward(self, x):
        # x.register_hook(get_hook('exp_x'))
        y = torch.exp(x.max(1)[0])
        # y.register_hook(get_hook('exp_y'))
        return y
        # return self.exp_forward.apply(x)


class Smooth_Max(nn.Module):
    def __init__(self, filters,
                 input_shape,
                 kernel_size,
                 input_shift=1e-9,
                 padding='VALID',
                 strides=(1, 1),
                 alpha = 1,
                 **kwargs):
        super().__init__()
        self.filters = filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.input_shift = input_shift
        self.padding = padding.upper()
        self.strides = strides
        self.kernel_shape = self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.filters
        self.alpha = 1

        self.max_forward = smax_func(self.alpha)

    def forward(self, x):
        ax = torch.exp(torch.mul(x, self.alpha))
        s_max = torch.mul(x, ax / torch.sum(ax, dim=1, keepdims=True))
        return torch.sum(s_max, dim=1)
        # return self.exp_forward.apply(x)
        

