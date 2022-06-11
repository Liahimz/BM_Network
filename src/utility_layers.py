from time import pthread_getcpuclockid
from traceback import print_tb
import torch
import numpy as np
from utils import *
import torch.nn as nn
from torch.autograd import Function

USE_AUTOGRAD = False

def get_hook(name):
    def print_hook(grad):
        print(f'{name} = {torch.mean(grad)}, {torch.std(grad)}')
        return grad
    return print_hook

def ln_func(min_val):
    class ln_func(Function):
        @staticmethod
        def forward(ctx, x):
            y = torch.clamp(x, min=min_val)
            y = torch.log(y)
            # y =  torch.unsqueeze(y, 2)
            ctx.save_for_backward(y, x)
            return y

        @staticmethod
        def backward(ctx, y_grad):
            y, x = ctx.saved_tensors
            # x = torch.clamp(x, min=min_val)
            mask = x <= min_val
            # print(mask)
            del_x = 1/x
            del_x[mask] = 0
            # print(del_x)
            return del_x * y_grad
    return ln_func

def add_func(kernel_shape, grad_coef, layer):
    class add_func(Function):
        @staticmethod
        def forward(ctx, x, k):
            filter_height, filter_width, in_channels, out_channels = kernel_shape
            ctx.x_shape = x.shape
            ctx.k_shape = k.shape
            x = torch.unsqueeze(x, 2)
            k_for_patches = k.view(filter_height * filter_width * in_channels, out_channels)
            k = k_for_patches[None, :, :, None, None]
            
            xk = x + k_for_patches[None, :, :, None, None]
            return xk

        @staticmethod
        def backward(ctx, y_grad):
            
            k_grad = torch.sum(y_grad, [0, -1, -2])
            # print("k_grad ", k_grad.shape)
            k_grad = k_grad.view(ctx.k_shape)
            # return torch.ones(x.shape, requires_grad=True, device=y_grad.device) * y_grad, torch.ones(k.shape, requires_grad=True, device=y_grad.device) * tmp
            x_grad = torch.sum(y_grad, [2])
            # print("x_grad ", x_grad.shape)
            x_grad = x_grad.view(ctx.x_shape)

            # coef_k = torch.numel(k_grad) * 0.0000001 / torch.norm(k_grad, p=2)
            # coef_x = torch.numel(x_grad) * 0.0000001 / torch.norm(x_grad, p=2)
            sum_coef = torch.norm(k_grad, p=2) + torch.norm(x_grad, p=2)
            # print(sum_coef)
            sum_coef = (torch.numel(k_grad) + torch.numel(x_grad)) / sum_coef
            coef_x = coef_k = 1
            mean = torch.abs(torch.mean(k_grad))
            
            if mean < 10e-16:
                coef_x = coef_k = 0
            elif mean < 10e-14:
                coef_x = coef_k = 10e-9 
            elif mean > 10e-2:
                coef_x = coef_k = 10e-3 / mean
            elif mean > 10e-3:
                coef_x = coef_k = 10e-4 / mean
            elif mean > 10e-4:
                coef_x = coef_k = 10e-6 / mean

            # if layer > 4:
            #     coef = 10
            # if layer > 3:
            #     coef = 1
            coef_x = coef_k = 1

            # print(coef)
            # print(f'k_grad = {layer} {torch.mean(k_grad)} {sum_coef}') #{torch.std(k_grad)}
            return x_grad * coef_x, k_grad * coef_k
    return add_func

def exp_func():
    class exp_func(Function):
        @staticmethod
        def forward(ctx, x):
            y = torch.exp(x)
            ctx.save_for_backward(y, x)
            return y

        @staticmethod
        def backward(ctx, y_grad):
            y, x = ctx.saved_tensors
            return y * y_grad
    return exp_func

def smax_func(alpha):
    class smax_func(Function):
        @staticmethod
        def forward(ctx, x):
            eax = torch.exp(torch.mul(x, alpha))
            s1 = torch.sum(eax, dim=1)
            s0 = torch.sum(x * eax, dim=1)
            ctx.save_for_backward(x, eax, s1, s0)
            
            return s0 / s1

        @staticmethod
        def backward(ctx, y_grad):
            x, eax, s1, s0 = ctx.saved_tensors
            tmp1 = (alpha * x * eax + eax) * torch.unsqueeze(s1, 1)
            tmp2 = alpha * eax * torch.unsqueeze(s0, 1)
            res = (tmp1 - tmp2) / torch.unsqueeze(s1 * s1, 1)
            print(torch.mean(res * torch.unsqueeze(y_grad, 1)))
            return res * torch.unsqueeze(y_grad, 1) * 10e+7
    return smax_func

class Ln(nn.Module):
    def __init__(self, 
                min_val = 1e-11,
                 **kwargs):
        super().__init__()
        # self.filters = filters
        # self.input_shape = input_shape
        # self.kernel_size = kernel_size
        self.min_val = min_val
        # self.padding = padding.upper()
        # self.strides = strides
        # self.kernel_shape = self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.filters

        self.ln_forward = ln_func(min_val=min_val)
        
    def forward(self, x):
        if USE_AUTOGRAD:
            y = torch.clamp(x, min=self.min_val).requires_grad_(True)
            y = torch.log(y).requires_grad_(True)
            # y = torch.unsqueeze(y, 2).requires_grad_(True)
            return y
        return self.ln_forward.apply(x)

class Add(nn.Module):
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
        if USE_AUTOGRAD:
            filter_height, filter_width, in_channels, out_channels = self.kernel_shape
            x = torch.unsqueeze(x, 2)
            k_for_patches = self.k.view(filter_height * filter_width * in_channels, out_channels)
            xk = x + k_for_patches[None, :, :, None, None]
            return xk
        return self.add_forward.apply(x, self.k)
        


class Exp(nn.Module):
    def __init__(self, 
                 **kwargs):
        super().__init__()

        self.exp_forward = exp_func()

    def forward(self, x):
        if USE_AUTOGRAD:
            y = torch.exp(x)
            return y
        return self.exp_forward.apply(x)


class Smooth_Max(nn.Module):
    def __init__(self, 
                 alpha = 1,
                 **kwargs):
        super().__init__()
        self.alpha = alpha
        self.max_forward = smax_func(self.alpha)

    def forward(self, x):
        if USE_AUTOGRAD:
            ax = torch.exp(torch.mul(x, self.alpha))
            s_max = torch.mul(x, ax / torch.sum(ax, dim=1, keepdims=True))
            # print(torch.sum(s_max, dim=1))
            return torch.sum(s_max, dim=1)
        return self.max_forward.apply(x)
        

class Lmooth_Max(nn.Module):
    def __init__(self, 
                 alpha = 10,
                 **kwargs):
        super().__init__()
        self.alpha = alpha
        # self.max_forward = smax_func(self.alpha)

    def forward(self, x):
        if USE_AUTOGRAD:
            return torch.sum(torch.pow(x, self.alpha + 1), dim=1) / torch.sum(torch.pow(x, self.alpha), dim=1)
        return None

class LnExp_Max(nn.Module):
    def __init__(self, 
                 alpha = 10,
                 **kwargs):
        super().__init__()
        self.alpha = alpha
        # self.max_forward = smax_func(self.alpha)

    def forward(self, x):
        # if USE_AUTOGRAD:
        ax = torch.exp(torch.mul(x, self.alpha))
        return torch.log(torch.sum(ax, dim=1)) / self.alpha
        # return None
