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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device: ", device)

def loss_fn_kd(outputs, labels, teacher_outputs, params):
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

def mnist(batch_size):
    
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
    # trainset = torchvision.datasets.FashionMNIST(root='./data/f_mnist', train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
    # testset = torchvision.datasets.FashionMNIST(root='./data/f_mnist', train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    return train_dataloader, test_dataloader

def predict(pred):
    return torch.argmax(pred, axis=1)

def score(y_pred, y):
    N = len(y)
    return ((y_pred == y).sum() / N).item()

def train(model, teacher_model, train_dataloader, criterion, 
    optimizer, writer=None, n_epochs=1, show=False, verbose=10):
    
    model.train()
    teacher_model.eval()
    loss_trace = []
    accuracy_trace = []
    train_stat = []

    teacher_model_children = list(model.children())

    if writer is not None:
        if (n_epochs == 1):
            train_stat = Statistics(len(train_dataloader), writer, 'train')
        else:
            train_stat = Statistics(n_epochs, writer, 'train')

    for epoch_i in range(n_epochs):   
        mean_acc = None
        
        with tqdm(train_dataloader, unit="batch") as tepoch:     
            for iter_i, (x, y) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch_i}")
                x, y = x.to(device), y.to(device)  

                teacher_pred = teacher_model(x)
                conv_layers = []
                for i in range(len(teacher_model_children)):
                    for item in teacher_model_children[i]:
                        if type(item) == nn.Conv2d:
                            conv_layers.append(item)
                teacher_output = conv_layers[-1](x)

                optimizer.zero_grad()          
                pred = model(x)
                loss = criterion(pred, teacher_output)
                loss.backward()
                optimizer.step()
                loss_trace.append(loss.item())
                y_pred = predict(pred)
                accuracy_trace.append(score(y_pred, y))
                
                if mean_acc is not None:
                    mean_acc = 0.9 * mean_acc + 0.1 * accuracy_trace[-1]
                else:
                    mean_acc = accuracy_trace[-1]

                if (n_epochs == 1 and writer is not None):
                    train_stat.append(loss_trace[-1], accuracy_trace[-1], [0], [0], iter_i)

                tepoch.set_postfix(loss=loss_trace[-1], mean_accuracy = mean_acc, batch_accuracy=accuracy_trace[-1])
    return loss_trace[-1], accuracy_trace[-1]


def train_multilayer(depth, epochs, batch_size=100, name="BM_NET", with_logs = False, save_params = False):
    stats = {}
    for d in depth:
        model = Smorph_Net(d, (1, 28, 28))
        teacher_model = CNN_Net(d, (1, 28, 28))
        model = model.to(device)
        teacher_model.to(device)
        teacher_model.load_state_dict(torch.load('models/CNN_3_23-06-2022_14:53:03_trained.pt'))

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        dump_file = None
        writer = None

        dt = datetime.now()
        str_date_time = dt.strftime("%d-%m-%Y_%H:%M:%S")
        model_name = name + "_" + str(d) + "_" + str_date_time

        train_dataloader, test_dataloader = mnist(batch_size)

        train_loss, train_accuracy = train(model, train_dataloader, test_dataloader,
                                    criterion, optimizer, writer,
                                    n_epochs=epochs, show=False)

        if save_params:
            models = "models/"
            dump_file = path.join(models, model_name + "_trained.pt")
            torch.save(model.state_dict(), dump_file)

        stats[d] = (train_loss, train_accuracy)
    return model
