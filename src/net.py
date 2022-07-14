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
        
        self.layers = []
        self.layers.append(BipolarMorphological2D_Torch(32, kernel_size = (3, 3), input_shape=shape))
        # self.layers.append(nn.Conv2d(1, 32, (3,3)))
        self.layers.append(nn.ReLU())
        out_shape = shape
        for i in range(depth):
            out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
            # print(out_shape)
            self.layers.append(BipolarMorphological2D_Torch(16, kernel_size = (3, 3), input_shape=out_shape))
            self.layers.append(nn.ReLU())
            
        out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
        # print(out_shape)
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(np.prod(out_shape), 10))
        self.layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)

size = [21632, 9216, 7744, 6400, 5184]
# size = [21632, 9216, 7744, 6400, 324]

class CNN_Net(nn.Module):

    def __init__(self, depth, shape):
        super().__init__()
        
        self.layers = []
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size = (3, 3)))
        # layers.append(Conv2d(in_channels=1, out_channels=32, kernel_size = (3, 3), layer = 1))
        self.layers.append(nn.ReLU())
        out_shape = 32
        for i in range(depth):
            self.layers.append(nn.Conv2d(in_channels=out_shape, out_channels=16, kernel_size = (3, 3)))
            # layers.append(Conv2d(in_channels=out_shape, out_channels=16, kernel_size = (3, 3), layer = i + 2))
            self.layers.append(nn.ReLU())
            out_shape = 16
            
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(np.prod(size[depth]), 10))
        # self.layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        # xk = torch.tensor(x, requires_grad=True)
        # xk.register_hook(get_hook('x'))
        # x1 = 
        
        return self.net(x)


class Smorph_Net(nn.Module):

    def __init__(self, depth, shape):
        super().__init__()
        
        self.layers = []
        # self.layers.append(SMorphLayer(filters=32, kernel_size = (3, 3), input_shape=shape, alpha=2.5, layer=1))
        self.layers.append(SMorphLayer(filters=32, kernel_size = (3, 3), input_shape=shape, alpha=2.5, layer=1))
        # self.layers.append(nn.Tanh())
        self.layers.append(nn.ReLU())
        out_shape = shape
        for i in range(depth):
            out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
            # self.layers.append(SMorphLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=2, layer=i + 2))
            self.layers.append(SMorphLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=2, layer=i + 2))
            # self.layers.append(nn.Tanh())
            self.layers.append(nn.ReLU())
            
        out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(np.prod(out_shape), 10))
        self.layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)

class LSE_Net(nn.Module):

    def __init__(self, depth, shape, alpha=1):
        super().__init__()
        
        self.layers = []
        self.layers.append(LnExpMaxLayer(filters=32, kernel_size = (3, 3), input_shape=shape, alpha=alpha))
        # self.layers.append(nn.ReLU())
        self.layers.append(nn.Tanh())
        out_shape = shape
        for i in range(depth):
            out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
            print(out_shape)
            self.layers.append(LnExpMaxLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=alpha))
            # self.layers.append(nn.ReLU())
            self.layers.append(nn.Tanh())
            
        out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(np.prod(out_shape), 10))
        self.layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)

    

class Morph_Net(nn.Module):

    def __init__(self, depth, shape, alpha = 1):
        super().__init__()

        coefs = []
        for i in reversed(range(depth + 1)):
            coefs.append(np.power(10, i))
       
        self.layers = []
        self.layers.append(MorphLayer(filters=32, kernel_size = (3, 3), input_shape=shape, grad_coef=coefs[0], layer=1))
        self.layers.append(nn.ReLU())
        out_shape = shape

        for i in range(depth):
            out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
            # print(out_shape)
            self.layers.append(MorphLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, grad_coef = coefs[i + 1], layer= i + 1))
            self.layers.append(nn.ReLU())
            
        out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(np.prod(out_shape), 10))
        # self.layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)

class MorphSmax_Net(nn.Module):

    def __init__(self, depth, shape, alpha = 1):
        super().__init__()

        coefs = []
        for i in reversed(range(depth + 1)):
            coefs.append(np.power(10, i))
       
        self.layers = []
        self.layers.append(BMLayer_Smax(filters=32, kernel_size = (3, 3), input_shape=shape, grad_coef=coefs[0], layer=1, alpha=alpha))
        self.layers.append(nn.ReLU())
        out_shape = shape

        for i in range(depth):
            out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
            # print(out_shape)
            self.layers.append(BMLayer_Smax(filters=16, kernel_size = (3, 3), input_shape=out_shape, grad_coef = coefs[i + 1], layer= i + 1, alpha=alpha))
            self.layers.append(nn.ReLU())
            
        out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(np.prod(out_shape), 10))
        # self.layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)

class MorphSmax_NetBiased(nn.Module):

    def __init__(self, depth, shape, alpha = 1):
        super().__init__()

        coefs = []
        for i in reversed(range(depth + 1)):
            coefs.append(np.power(10, i))
       
        self.layers = []
        self.layers.append(BMLayer_Smax_Biased(filters=32, kernel_size = (3, 3), input_shape=shape, layer=1, alpha=alpha))
        self.layers.append(nn.ReLU())
        out_shape = shape

        for i in range(depth):
            out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
            # print(out_shape)
            self.layers.append(BMLayer_Smax_Biased(filters=16, kernel_size = (3, 3), input_shape=out_shape, layer= i + 1, alpha=alpha))
            self.layers.append(nn.ReLU())
            
        out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(np.prod(out_shape), 10))
        # self.layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)


class LeNet(nn.Module):

    def __init__(self, depth, shape):
        super().__init__()
        
        self.layers = []
        self.layers.append(SMorphLayer(filters=8, kernel_size = (3, 3), input_shape=shape, alpha=2.5, layer=1))
        self.layers.append(nn.ReLU())
        out_shape = self.layers[-2].compute_output_shape(input_shape=shape)

        # print(out_shape)
        # self.layers.append(nn.AvgPool2d(kernel_size=(3, 3), stride=1))
        # out_shape = (out_shape[0], out_shape[1] - (3) + 1, out_shape[2] - (3) + 1)

        self.layers.append(SMorphLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=2.2, layer=2))
        self.layers.append(nn.ReLU())
        out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)

        print(out_shape)
        self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=1))
        out_shape = (out_shape[0], out_shape[1] - (2 - 1), out_shape[2] - (2 - 1))


        for d in range(depth - 2):
            self.layers.append(SMorphLayer(filters=32, kernel_size = (3, 3), input_shape=out_shape, alpha=2.0, layer=d + 3))
            self.layers.append(nn.ReLU())
            out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
        
        # if depth > 2:
        #     print(out_shape)
        #     self.layers.append(nn.AvgPool2d(kernel_size=(3, 3), stride=1))
        #     out_shape = (out_shape[0], out_shape[1] - (3) + 1, out_shape[2]- (3) + 1)

        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(np.prod(out_shape), 10))
        print(np.prod(out_shape))
        self.layers.append(nn.Softmax(dim=1))
        
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)


class CNNLeNet(nn.Module):

    def __init__(self, depth, shape):
        super().__init__()
        
        self.layers = []
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=8, kernel_size = (5, 5)))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.MaxPool2d(kernel_size=(3, 3), stride=1))

        self.layers.append(nn.Conv2d(in_channels=8, out_channels=16, kernel_size = (5, 5)))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=1))

       
        self.layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size = (3, 3)))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=1))

        self.layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size = (3, 3)))
        self.layers.append(nn.ReLU())
           
        
        # if depth > 2:
        #     print(out_shape)
        #     self.layers.append(nn.AvgPool2d(kernel_size=(3, 3), stride=1))
        #     out_shape = (out_shape[0], out_shape[1] - (3) + 1, out_shape[2]- (3) + 1)

        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(np.prod(4608), 2))
        self.layers.append(nn.Tanh())
        # self.layers.append(nn.LeakyReLU(negative_slope=0.1))
        # self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(2, 10))
        
        
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)

# class KDCNN_Net(nn.Module):

#     def __init__(self, depth, shape):
#         super().__init__()
        
#         self.layers = []
#         self.layers.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size = (3, 3)))

#         self.layers.append(nn.ReLU())
#         # out_shape = 32
#         for i in range(depth):
#             self.layers.append(nn.Conv2d(in_channels=out_shape, out_channels=16, kernel_size = (3, 3)))
#             # layers.append(Conv2d(in_channels=out_shape, out_channels=16, kernel_size = (3, 3), layer = i + 2))
#             self.layers.append(nn.ReLU())
#             out_shape = 16
            
#         self.layers.append(nn.Flatten())
#         self.layers.append(nn.Linear(np.prod(size[depth]), 10))
#         # self.layers.append(nn.Softmax(dim=1))
#         self.net = nn.Sequential(*self.layers)

#     def forward(self, x):
        

#         # for layer in self.layers:
#         #     x = layer(x)
        
#         return self.net(x)


# class KDSmorph_Net(nn.Module):

#     def __init__(self, depth, shape):
#         super().__init__()
        
#         self.shape = shape
#         self.depth = depth
#         self.layers = []
#         self.layers.append(SMorphLayer(filters=32, kernel_size = (3, 3), input_shape=shape, alpha=2.5, layer=1))
#         self.layers.append(nn.ReLU())
#         out_shape = shape

#         for i in range(depth):
#             out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
#             self.layers.append(SMorphLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=2, layer=i + 2))
#             self.layers.append(nn.ReLU())
            
#         out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
#         self.layers.append(nn.Flatten())
#         self.layers.append(nn.Linear(np.prod(out_shape), 10))
#         # self.layers.append(nn.Softmax(dim=1))
#         self.net = nn.Sequential(*self.layers)

#     def forward(self, x):
#         # for layer in self.layers:
#         #     x = layer(x)

#         return self.net(x)

# class KDLSE_Net(nn.Module):

#     def __init__(self, depth, shape, alpha = 1):
#         super().__init__()
        
#         self.shape = shape
#         self.depth = depth
#         self.layers = []
#         self.layers.append(LnExpMaxLayer(filters=32, kernel_size = (3, 3), input_shape=shape, alpha=alpha, layer=1))
#         self.layers.append(nn.ReLU())
#         out_shape = shape

#         for i in range(depth):
#             out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
#             self.layers.append(LnExpMaxLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=alpha, layer=i + 2))
#             self.layers.append(nn.ReLU())
            
#         out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
#         self.layers.append(nn.Flatten())
#         self.layers.append(nn.Linear(np.prod(out_shape), 10))
#         # self.layers.append(nn.Softmax(dim=1))
#         self.net = nn.Sequential(*self.layers)

#     def forward(self, x):
#         # for layer in self.layers:
#         #     x = layer(x)

#         return self.net(x)



# class Test_smorph(nn.Module):
#     def __init__(self, depth, shape):
#         super().__init__()
        
#         self.layers = []
#         self.layers.append(SMorphLayer(filters=32, kernel_size = (3, 3), input_shape=shape, alpha=2.5, layer=1))
#         self.layers.append(nn.ReLU())

#         out_shape = self.layers[-2].compute_output_shape(input_shape=shape)
#         # self.layers.append(nn.MaxPool2d(kernel_size=(3, 3), stride=1))

#         self.layers.append(SMorphLayer(filters=16, kernel_size = (3, 3), input_shape=out_shape, alpha=2, layer=1))
#         self.layers.append(nn.ReLU())
#         out_shape = self.layers[-2].compute_output_shape(input_shape=shape)
#         # self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=1))

       
#         # self.layers.append(SMorphLayer(filters=32, kernel_size = (3, 3), input_shape=out_shape, alpha=2, layer=1))
#         # self.layers.append(nn.ReLU())
#         # out_shape = self.layers[-2].compute_output_shape(input_shape=shape)
#         # # self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=1))

#         # self.layers.append(SMorphLayer(filters=32, kernel_size = (3, 3), input_shape=out_shape, alpha=2, layer=1))
#         # self.layers.append(nn.ReLU())
#         # out_shape = self.layers[-2].compute_output_shape(input_shape=shape)
        
#         # if depth > 2:
#         #     print(out_shape)
#         #     self.layers.append(nn.AvgPool2d(kernel_size=(3, 3), stride=1))
#         #     out_shape = (out_shape[0], out_shape[1] - (3) + 1, out_shape[2]- (3) + 1)

#         print(out_shape)
#         # self.layers.append(nn.Flatten())
#         self.layers.append(SMorphLayer(filters=2, kernel_size = (out_shape[1], out_shape[2]), input_shape=out_shape, alpha=2, layer=1))
#         # # self.layers.append(nn.Tanh())
#         # # self.layers.append(nn.LeakyReLU(negative_slope=0.1))
#         # self.layers.append(nn.ReLU())
#         out_shape = self.layers[-1].compute_output_shape(input_shape=shape)
#         self.layers.append(SMorphLayer(filters=10, kernel_size = (1, 1), input_shape=out_shape, alpha=2, layer=1))
#         self.layers.append(nn.Flatten())

#         # out_shape = self.layers[-2].compute_output_shape(input_shape=out_shape)
#         # print(out_shape)
#         # self.layers.append(nn.Flatten())
#         # self.layers.append(nn.Linear(np.prod(out_shape), 10))
#         self.net = nn.Sequential(*self.layers)

#     def forward(self, x):
#         return self.net(x)
