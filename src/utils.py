import math
import torch.nn.functional as F
import torch

def extract_image_patches(x, kernel, stride=1, dilation=1):
    # print(x.shape)
    batch_size,c,h,w = x.shape
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0,4,5,1,2,3).contiguous()
    return patches.view(batch_size,-1,patches.shape[-2], patches.shape[-1])

def extract_patches(x,
                    sizes,
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'):
    b,h,w,c = x.shape
    filter_height = sizes[1]
    filter_width = sizes[2]
    patches = x.unfold(1, filter_height, strides[1]).unfold(2, filter_width, strides[2])
    patches = patches.permute(0, 1, 2, 4, 5, 3).contiguous()
    x_patches =  patches.view(b, patches.shape[1], patches.shape[2], -1)
    return x_patches

def smooth_max(input_tensor, dim = 1, alpha = 1):
    ax = torch.exp(torch.mul(input_tensor, alpha))
    s_max = torch.mul(input_tensor, ax / torch.sum(ax, dim=dim, keepdims=True))
    return torch.sum(s_max, dim=dim)

