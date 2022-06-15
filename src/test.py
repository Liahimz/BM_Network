from time import sleep
import torch
import numpy as np
from utils import *
import torch.nn as nn
from BM_Neuron import* 
from net import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import torch.utils.tensorboard as tb
from torch_tensorboard import *
from os import path
from datetime import datetime

model_nottrained = Smorph_Net(0, (1, 28, 28))

model_nottrained.load_state_dict(torch.load("models/BiSmorph_0_15-06-2022_20:32:16.pt"))

model = Smorph_Net(0, (1, 28, 28))

model.load_state_dict(torch.load("models/BiSmorph_0_15-06-2022_20:32:16_trained.pt"))

for param_tensor in model.state_dict():
    x = (model.state_dict()[param_tensor] - model_nottrained.state_dict()[param_tensor]) / model.state_dict()[param_tensor]
    print(param_tensor)
    print(x)