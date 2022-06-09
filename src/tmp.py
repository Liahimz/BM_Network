import torch
import numpy as np


t2 = torch.tensor([[[[[0, 1, 2, 3, 4],           [5, 6, 7, 8, 9]],
                    [[5, 6, 7, 8, 9],           [10, 11, 12, 13, 14]],
                    [[10, 11, 12, 13, 14],      [15, 16, 17, 18, 19]]], 

                   [[[0, -1, -2, -3, -4],       [-5, -6, -7, -8, -9]],
                    [[-5, -6, -7, -8, -9],      [-10, -11, -12, -13, -14]],
                    [[-10, -11, -12, -13, -14], [-15, -16, -17, -18, -19]]]],

                  [[[[20, 21, 22, 23, 24],      [25, 26, 27, 28, 29]],
                    [[25, 26, 27, 28, 29],      [30, 31, 32, 33, 34]],
                    [[30, 31, 32, 33, 34],      [35, 36, 37, 38, 39]]], 

                   [[[-20, -21, -22, -23, -24], [-25, -26, -27, -28, -29]],
                    [[-25, -26, -27, -28, -29], [-30, -31, -32, -33, -34]],
                    [[-30, -31, -32, -33, -34], [-35, -36, -37, -38, -39]]]]], dtype=torch.float32)

def smorph(x, alpha = 10, dim = 1):
    ax = torch.exp(torch.mul(x, alpha))
    s_max = torch.mul(x, ax / torch.sum(ax, dim=dim, keepdims=True))
    # print(torch.sum(s_max, dim=1))
    return torch.sum(s_max, dim=dim)

def lmorph(x, alpha = 10, dim = 1):
    return torch.sum(torch.pow(x, alpha + 1), dim=dim) / torch.sum(torch.pow(x, alpha), dim=dim)


print(t2.max(1))

print(lmorph(t2))

print(smorph(t2))