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

def mnist(batch_size):
    
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    return train_dataloader, test_dataloader


def show_progress(epoch, loss_trace, accuracy_trace, x, y, y_pred):
    clear_output(wait=True)
    
    fig, axis = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'width_ratios': [2, 2, 1]})
    
    plt.subplot(1, 3, 1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(loss_trace)
                
    plt.subplot(1, 3, 2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.plot(accuracy_trace)
    
    plt.subplot(1, 3, 3)
    img = x[0].detach().cpu().permute(1, 2, 0).numpy() / 2 + 0.5
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'true: {y[0]}    pred: {y_pred[0]}')
    plt.savefig("pics/pred_" + str(epoch) + ".png")
    plt.show()


def predict(pred):
    return torch.argmax(pred, axis=1)

def score(y_pred, y):
    N = len(y)
    return ((y_pred == y).sum() / N).item()

def test(model, test_dataloader):

    model.eval()
    total_accuracy = 0.0
    total = 0
    
    with torch.no_grad():
        for x, y in test_dataloader:            
            x, y = x.to(device), y.to(device)            
            pred = model(x)
            y_pred = predict(pred)
            total_accuracy += (y_pred == y).sum()
            total += len(y)
  
    total_accuracy /= total
    return total_accuracy.item()
    
def train(model, train_dataloader, test_dataloader, criterion, 
    optimizer, writer=None, n_epochs=1, show=False, verbose=10):
    
    model.train()
    loss_trace = []
    accuracy_trace = []
    train_stat = []
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
                optimizer.zero_grad()          
                pred = model(x)
                loss = criterion(pred, y)
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
                    train_stat.append(loss_trace[-1], accuracy_trace[-1], iter_i)

                tepoch.set_postfix(loss=loss_trace[-1], mean_accuracy = mean_acc, batch_accuracy=accuracy_trace[-1])
                # sleep(0.1)
            if show and (iter_i + 1) % verbose == 0:
                show_progress(epoch_i, loss_trace, accuracy_trace, x, y, y_pred)
        
        if (n_epochs != 1 and writer is not None):
            train_stat.append(loss_trace[-1], mean_acc, epoch_i)

        # test_accuracy = test(model, test_dataloader)
        # print(f'test  accuracy: {test_accuracy:.3f}')

    return loss_trace[-1], accuracy_trace[-1]



def train_multilayer(depth, epochs, batch_size=100, name="BM_NET", with_logs = False):
    stats = {}
    for d in depth:
        model = Smorph_Net(d, (1, 28, 28))
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        writer = None
        if with_logs:
            dt = datetime.now()
            str_date_time = dt.strftime("%d-%m-%Y_%H:%M:%S")
            model_name = name + "_" + str(d) + "_" + str_date_time

            logs = "logs/"
            writer = tb.writer.SummaryWriter(path.join(logs, model_name))

        train_dataloader, test_dataloader = mnist(batch_size)

        train_loss, train_accuracy = train(model, train_dataloader, test_dataloader,
                                    criterion, optimizer, writer,
                                    n_epochs=epochs, show=False)
        stats[d] = (train_loss, train_accuracy)
    return stats


depths = [1, 2, 3, 4]
train_multilayer(depth=depths, epochs=50, batch_size=100, name="SMORPH", with_logs=True)
