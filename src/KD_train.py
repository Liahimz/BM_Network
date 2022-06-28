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

from vizualize_utils import PlotLine

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

def train(model, model_layers, teacher_model, teacher_model_layers, train_dataloader,
    optimizer, scheduler, writer=None, n_epochs=1):
    
    model.train()
    teacher_model.eval()
    loss_trace = []
    accuracy_trace = []
    train_stat = []

    
    stat = []
    for epoch_i in range(n_epochs):   
        mean_acc = None
        
        with tqdm(train_dataloader, unit="batch") as tepoch:     
            for iter_i, (x, y) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch_i}")
                x, y = x.to(device), y.to(device)  

                model_output = model_layers[0](x)
                teacher_output = teacher_model_layers[0](x)


                optimizer.zero_grad()          
                pred = model(x)
                # teacher_pred = teacher_model(x)

                # if epoch_i > 3:
                #     print((nn.CrossEntropyLoss()(pred, y)).item())
                #     print((nn.MSELoss()(model_output, teacher_output)).item())

                mse_loss = (nn.MSELoss()(model_output, teacher_output)).item()
                if iter_i > 0:
                    stat.append(0.6 * mse_loss + 0.4 * stat[-1])
                else:
                    stat.append(mse_loss)

                alpha = 0.000001
                loss = alpha * (nn.CrossEntropyLoss()(pred, y)) + (1 - alpha) * (nn.MSELoss()(model_output, teacher_output))
                # loss = alpha * (nn.MSELoss()(pred, teacher_pred)) + (1 - alpha) * (nn.MSELoss()(model_output, teacher_output))
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

        scheduler.step()
    # fig = plt.figure(figsize=(30, 50))
    # plt.plot(stat)
    # fig = plt.figure()
    # plt.plot(stat)
    # plt.savefig("pics/" + str('_mse_raiting.png'), bbox_inches='tight')
    # plt.close(fig)
    
    return loss_trace[-1], accuracy_trace[-1], stat


def train_multilayer(depth, epochs, batch_size=100, name="BM_NET", with_logs = False, save_params = False):
    stats = {}
    #just for now depth is alpha
    for d in depth:
        model = KDLSE_Net(0, (1, 28, 28), d)
        model_layers = model.layers
        teacher_model = KDCNN_Net(0, (1, 28, 28))
        teacher_model_layers = teacher_model.layers

        model = model.to(device)
        teacher_model = teacher_model.to(device)
        teacher_model.load_state_dict(torch.load('models/KDCNN_0_27-06-2022_13:36:10_trained.pt'))

        # criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 70], gamma=0.1)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        dump_file = None
        writer = None

        dt = datetime.now()
        str_date_time = dt.strftime("%d-%m-%Y_%H:%M:%S")
        model_name = name + "_" + str(d) + "_" + str_date_time

        train_dataloader, test_dataloader = mnist(batch_size)

        train_loss, train_accuracy, stat = train(model, model_layers, teacher_model, teacher_model_layers, train_dataloader,
                                    optimizer, scheduler, writer,
                                    n_epochs=epochs)

        if save_params:
            models = "models/"
            dump_file = path.join(models, model_name + "_alpha=" + str(d) +"_trained.pt")
            torch.save(model.state_dict(), dump_file)

            PlotLine(np.arange(len(stat)), stat, model_name + "_alpha=" + str(d) + "mse_rating")

        stats[d] = (train_loss, train_accuracy)
    return model


depths = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22]
model = train_multilayer(depth=depths, epochs=50, batch_size=100, name="KDLSE_alpha_test", with_logs=False, save_params=True)