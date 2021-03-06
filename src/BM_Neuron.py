from warnings import filters
import torch
import numpy as np
from utils import *
import torch.nn as nn
from torch.autograd import Function
from utility_layers import *

def bm_func(kernel_shape, input_shift):
    class BM_func(Function):
        @staticmethod
        def forward(ctx, x, k1, k2, bias):
            filter_height, filter_width, in_channels, out_channels = kernel_shape
            k1_for_patches = torch.tensor(torch.reshape(k1, [filter_height * filter_width * in_channels, out_channels]))
            k2_for_patches = torch.tensor(torch.reshape(k2, [filter_height * filter_width * in_channels, out_channels]))
            
            batch_size,c,h,w = x.shape
            patches = x.unfold(2, filter_height, 1).unfold(3, filter_width, 1)
           
            patches = patches.permute(0,4,5,1,2,3)
           
            x1_patches = patches.reshape(batch_size,-1,patches.shape[-2], patches.shape[-1])
           

            x1_patches = torch.clamp(x1_patches, min=input_shift)
            x1_patches = torch.log(x1_patches)
            
            x1_patches =  torch.tensor(torch.unsqueeze(x1_patches, 2))
            
            
            x1k1 = x1_patches + k1_for_patches[None, :, :, None, None]
            y11 = torch.exp(x1k1.max(1)[0])
            
            x1k2 = x1_patches + k2_for_patches[None, :, :, None, None]
            y12 = torch.exp(x1k2.max(1)[0])

            x2_patches = extract_image_patches(-x, 3)

            x2_patches = torch.clamp(x2_patches, min=input_shift)
            x2_patches = torch.log(x2_patches)
        
            x2_patches = torch.tensor(torch.unsqueeze(x2_patches, 2))
            

            x2k1 = x2_patches + k1_for_patches[None, :, :, None, None]
            y21 = torch.exp(x2k1.max(1)[0])
            x2k2 = x2_patches + k2_for_patches[None, :, :, None, None]
            y22 = torch.exp(x2k2.max(1)[0])
            # ctx.save_for_backward(input, output)
            ctx.save_for_backward(x, k1, k2, bias)
            y = torch.add(y11 - y12 - y21 + y22, bias[None, :, None, None])
            
            return y

        @staticmethod
        def backward(ctx, y_grad):
            x, k1, k2, bias = ctx.saved_tensors
            return x.grad, k1.grad, k2.grad, bias.grad
    return BM_func

class BipolarMorphological2D_Torch(nn.Module):
    def __init__(self, filters,
                 input_shape,
                 kernel_size,
                 input_shift=0.1,
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

        self.build()
        self.BM_forward = bm_func(self.kernel_shape, self.input_shift)

    def build(self):
        self.kernel_shape = self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.filters
        
        self.k1 = torch.nn.Parameter(torch.empty(self.kernel_shape, requires_grad=True))
        self.k2 = torch.nn.Parameter(torch.empty(self.kernel_shape, requires_grad=True))
        self.bias = torch.nn.Parameter(torch.zeros(self.filters, requires_grad=True))


        torch.nn.init.xavier_uniform_(self.k1)
        torch.nn.init.xavier_uniform_(self.k2)

       

    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            
            return (self.kernel_shape[3], (self.input_shape[1] - self.kernel_shape[1] + 1) // self.strides[0],
                   (self.input_shape[2] - self.kernel_shape[0] + 1) // self.strides[1])
        else:
            
            return (self.kernel_shape[3], self.input_shape[1] // self.strides[0], self.input_shape[2] // self.strides[1])

    def forward(self, x):
       
        return self.BM_forward.apply(x, self.k1, self.k2, self.bias)

class SMorphLayer(nn.Module):
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

        self.add_k1 = Add(filters,
                 input_shape,
                 kernel_size)

        self.add_k2 = Add(filters,
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
        filter_height, filter_width, in_channels, out_channels = self.kernel_shape
        # x1_pathces = self.ln(x)
        # x2_pathces = self.ln(-x)

        x1_pathces = extract_image_patches(x, filter_height, self.strides[0])
        # x2_pathces = extract_image_patches(-x, filter_height, self.strides[0])
        x2_pathces = -x1_pathces
        x1_k1 = self.add_k1(x1_pathces)
        # x1_k2 = self.add_k2(x1_pathces)

        # x2_k1 = self.add_k1(x2_pathces)
        x2_k2 = self.add_k2(x2_pathces)
       
        # y11 = self.exp(self.smax(x1_k1))
        # y12 = self.exp(self.smax(x1_k2))
        # y21 = self.exp(self.smax(x2_k1))
        # y22 = self.exp(self.smax(x2_k2))

        y11 = (self.smax(x1_k1))
        # y12 = (self.smax(x1_k2))
        y21 = (self.smax(x2_k2))

        # y11 = ((x1_k1).max(1)[0])
        # y11.register_hook(get_hook("y1"))
        # y12 = (self.smax(x1_k2))
        # y21 = ((x2_k2).max(1)[0])
        # y22 = (self.smax(x2_k2))
        y = y11 + y21 + self.bias[None, :, None, None]
        return y


class LnExpMaxLayer(nn.Module):
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
        # y11 = ((x1_k1).max(1)[0])
        # y11 = (self.lmax(x1_k1))
        y22 = (self.lnxmax(x2_k2))
        # y22 = ((x2_k2).max(1)[0])
        # y11.register_hook(get_hook("y11"))
        y = y11 + y22 + self.bias[None, :, None, None]
        return y
        

class MorphLayer(nn.Module):
    def __init__(self, filters,
                 input_shape,
                 kernel_size,
                 min_val=1e-9,
                 padding='VALID',
                 strides=(1, 1),
                 grad_coef = 1e-2,
                 layer = 1,
                 **kwargs):
        super().__init__()
        self.filters = filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.min_val = min_val
        self.padding = padding.upper()
        self.strides = strides
        self.kernel_shape = self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.filters

        self.bias = torch.nn.Parameter(torch.zeros(self.filters, requires_grad=True))

        self.layer = layer

        self.batch_norm = nn.BatchNorm2d(filters)

        self.ln = Ln(min_val=1e-12)

        self.add_k1 = Add(filters,
                 input_shape,
                 kernel_size,
                 grad_coef,
                 self.layer)

        self.add_k2 = Add(filters,
                 input_shape,
                 kernel_size,
                 grad_coef,
                 self.layer)

        self.exp = Exp()

    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            return (self.kernel_shape[3], (self.input_shape[1] - self.kernel_shape[1] + 1) // self.strides[0],
                   (self.input_shape[2] - self.kernel_shape[0] + 1) // self.strides[1])
        else:
            
            return (self.kernel_shape[3], self.input_shape[1] // self.strides[0], self.input_shape[2] // self.strides[1])

    def forward(self, x):
        filter_height, filter_width, in_channels, out_channels = self.kernel_shape
        # print(f'x = {torch.mean(x)} {torch.std(x)}')
        # print("x = ", x.shape)
        x1_pathces = self.ln(x)
        x2_pathces = self.ln(-x)

        x1_pathces = extract_image_patches(x1_pathces, filter_height, self.strides[0])
        x2_pathces = extract_image_patches(x2_pathces, filter_height, self.strides[0])

        
        # print("lnx = ", x1_pathces.shape)

        x1_k1 = self.add_k1(x1_pathces)
        x1_k2 = self.add_k2(x1_pathces)

        x2_k1 = self.add_k1(x2_pathces)
        x2_k2 = self.add_k2(x2_pathces)
        # print(f'x+ln = {torch.mean(x1_pathces)} {torch.std(x1_pathces)}')
        # print(f'x-ln = {torch.mean(x2_pathces)} {torch.std(x2_pathces)}')
        # print(f'x1_k1 = {x1_k1.shape} {x1_k1.max(1)[0].shape}')
        y11 = self.exp(x1_k1.max(1)[0])
        # print(f'y11 = {torch.mean(y11)} {torch.std(y11)}')
        y12 = self.exp(x1_k2.max(1)[0])
        # print(f'y12 = {torch.mean(y12)} {torch.std(y12)}')
        y21 = self.exp(x2_k1.max(1)[0])
        # print(f'y21 = {torch.mean(y21)} {torch.std(y21)}')
        y22 = self.exp(x2_k2.max(1)[0])
        # print(f'y22 = {torch.mean(y22)} {torch.std(y22)}')
        # print(f'y12 = {y12.shape}')
        y = y11 - y12 - y21 + y22 + self.bias[None, :, None, None]
        # print(f'y = {torch.mean(y)} {torch.std(y)}')
        # print("y = ", y.shape)
        return y
        # return self.batch_norm(y)



class BMLayer_Smax(nn.Module):
    def __init__(self, filters,
                 input_shape,
                 kernel_size,
                 min_val=1e-9,
                 padding='VALID',
                 strides=(1, 1),
                 grad_coef = 1e-2,
                 layer = 1,
                 alpha = 1,
                 **kwargs):
        super().__init__()
        self.filters = filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.min_val = min_val
        self.padding = padding.upper()
        self.strides = strides
        self.kernel_shape = self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.filters

        self.bias = torch.nn.Parameter(torch.zeros(self.filters, requires_grad=True))

        # self.k = torch.nn.Parameter(torch.empty(self.kernel_shape, requires_grad=True))
        # torch.nn.init.xavier_uniform_(self.k)

        self.layer = layer

        self.batch_norm = nn.BatchNorm2d(filters)

        self.ln = Ln(min_val=1e-12)

        self.add_k1 = Add(filters,
                 input_shape,
                 kernel_size,
                 grad_coef,
                 self.layer)

        self.add_k2 = Add(filters,
                 input_shape,
                 kernel_size,
                 grad_coef,
                 self.layer)

        self.lnxmax = LnExp_Max(alpha=alpha, layer=layer)

        self.exp = Exp()

    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            return (self.kernel_shape[3], (self.input_shape[1] - self.kernel_shape[1] + 1) // self.strides[0],
                   (self.input_shape[2] - self.kernel_shape[0] + 1) // self.strides[1])
        else:
            
            return (self.kernel_shape[3], self.input_shape[1] // self.strides[0], self.input_shape[2] // self.strides[1])

    def forward(self, x):
        filter_height, filter_width, in_channels, out_channels = self.kernel_shape
        # x = x + 1

        # print(torch.mean(x))
        # print(torch.mean(-x))

        x1_pathces = self.ln(x)
        x2_pathces = self.ln(-x)

        # print(torch.mean(x1_pathces))
        # print(torch.mean(x2_pathces))

        x1_pathces = extract_image_patches(x1_pathces, filter_height, self.strides[0])
        x2_pathces = extract_image_patches(x2_pathces, filter_height, self.strides[0])


        x1_k1 = self.add_k1(x1_pathces)
        x1_k2 = self.add_k2(x1_pathces)
        x2_k1 = self.add_k1(x2_pathces)
        x2_k2 = self.add_k2(x2_pathces)

        # print(torch.mean(x1_k1))
        # print(torch.mean(x1_k2))
        # print(torch.mean(x2_k1))
        # print(torch.mean(x2_k2))
        # exit(0)
        y11 = self.exp(self.lnxmax(x1_k1))
        y12 = self.exp(self.lnxmax(x1_k2))
        y21 = self.exp(self.lnxmax(x2_k1))
        y22 = self.exp(self.lnxmax(x2_k2))

        # print(y11.shape)
        # print((x1_pathces * delta_k).sum(1).shape)
        y = y11 - y12 - y21 + y22 + self.bias[None, :, None, None]
        # y = y11 - (x1_pathces * delta_k).sum(1) + self.bias[None, :, None, None]
        return y


class BMLayer_Smax_Biased(nn.Module):
    def __init__(self, filters,
                 input_shape,
                 kernel_size,
                 min_val=1e-9,
                 padding='VALID',
                 strides=(1, 1),
                 layer = 1,
                 alpha = 1,
                 **kwargs):
        super().__init__()
        self.filters = filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.min_val = min_val
        self.padding = padding.upper()
        self.strides = strides
        self.kernel_shape = self.kernel_size[0], self.kernel_size[1], self.input_shape[0], self.filters

        self.bias = torch.nn.Parameter(torch.zeros(self.filters, requires_grad=True))

        self.k = torch.nn.Parameter(torch.empty(self.kernel_shape, requires_grad=True))
        torch.nn.init.xavier_uniform_(self.k)
        # print(self.kernel_shape)
        self.delta_x = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.delta_w = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.layer = layer

        self.ln = Ln(min_val=1e-12)

        self.lnxmax = LnExp_Max(alpha=alpha, layer=layer)

        self.exp = Exp()

        self.input_bias = 5
        self.weight_bias = 5

    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            return (self.kernel_shape[3], (self.input_shape[1] - self.kernel_shape[1] + 1) // self.strides[0],
                   (self.input_shape[2] - self.kernel_shape[0] + 1) // self.strides[1])
        else:
            
            return (self.kernel_shape[3], self.input_shape[1] // self.strides[0], self.input_shape[2] // self.strides[1])

    def forward(self, x):
        filter_height, filter_width, in_channels, out_channels = self.kernel_shape
        x1_pathces = x + self.input_bias

        x1_pathces = self.ln(x1_pathces)

        x1_pathces = extract_image_patches(x1_pathces, filter_height, self.strides[0])

        x_pathces = extract_image_patches(x, filter_height, self.strides[0])

        x1_pathces = torch.unsqueeze(x1_pathces, 2)
        k_for_patches = self.k.view(filter_height * filter_width * in_channels, out_channels) + self.weight_bias

        # print(self.k.shape)
        # print(x1_pathces.shape)
        x1_k1 = x1_pathces + k_for_patches[None, :, :, None, None]


        k_view = self.k.view(filter_height * filter_width * in_channels, out_channels)

        y11 = self.exp(self.lnxmax(x1_k1))

        # print(y11.shape)
        # print(x.shape)
        # print(self.k.shape)
        # print(x_pathces.shape)
        x_pathces = torch.sum(x_pathces, 1, keepdim=True) * self.delta_w
        # print(x_pathces.shape)

        k_view = (torch.sum(k_view, 0, keepdim=False) * self.delta_x)[None, :, None, None]
        # k_sx_s = self.weight_bias * self.input_bias * filter_height * filter_width * in_channels

        # k_sx_s = self.delta_x * self.delta_w * filter_height * filter_width * in_channels
        # print(k_view.shape)
        y = y11 - x_pathces - k_view + self.bias[None, :, None, None]
        return y