from time import sleep
from pyparsing import col
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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from KD_train import *


def mnist(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    return train_dataloader, test_dataloader

device = torch.device("cpu")

# model = CNNLeNet(0, (1, 28, 28))
model = Test_smorph(0, (1, 28, 28))
model.load_state_dict(torch.load('models/Test_smorph_nolin_0_27-06-2022_19:31:46_trained.pt'))
model.to(device)
# save_featuremap(model, mnist, SMorphLayer, "Smorph_blend=0.0", 1)

def test(model, model_layers, test_dataloader):

    model.eval()
    container = []
    labels = []
    
    with torch.no_grad():
        with tqdm(test_dataloader, unit="batch") as tepoch:
            for x, y in tepoch:            
                x, y = x.to(device), y.to(device)            
                # pred = model(x)

                for i, layer in enumerate(model_layers):
                    x = layer(x)
                    if (i == 4):
                        model_output = x
                        # print((model_output.sum(1)[0]).shape)
                        # print(torch.flatten(model_output).shape)
                        # exit(0)

                # model_output = model_layers[6](x).sum(1)[0]
                # print((model_output).shape)
                # print(torch.flatten(model_output).shape)
                container.append(torch.flatten(model_output).numpy())
                labels.append(y)
                
  
    # total_accuracy /= total
    # # total_loss /= total
    return container, labels

train_dataloader, test_dataloader = mnist(1)

container, labels = test(model, model.layers, test_dataloader)

size = len(test_dataloader)

# X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(np.array(container))

# print(size)
# pca = PCA(n_components=2)
# pca_out = pca.fit_transform(container)
# print(pca_out.shape)

x = []
y = []
color = []
for i, item in enumerate(container):
    # print
    x.append(item[0])
    y.append(item[1])
    color.append(labels[i])

fig = plt.figure(figsize=(15, 25))
plt.scatter(x = x, y = y, c = color, linewidths = 0.4, cmap='tab10')
plt.savefig(str('smorph_pca.png'), bbox_inches='tight')
plt.close(fig)