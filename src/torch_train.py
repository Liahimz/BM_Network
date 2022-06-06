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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())
print(device)

def mnist(batch_size):
    
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    return train_dataloader, test_dataloader


def show_progress(loss_trace, accuracy_trace, x, y, y_pred):
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
    plt.savefig("pics/pred.png")
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
    train_stat = Statistics(len(train_dataloader), writer, 'train')
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
                train_stat.append(loss_trace[-1], accuracy_trace[-1], iter_i)
                if mean_acc is not None:
                    mean_acc = 0.9 * mean_acc + 0.1 * accuracy_trace[-1]
                else:
                    mean_acc = accuracy_trace[-1]

                tepoch.set_postfix(loss=loss_trace[-1], mean_accuracy = mean_acc, batch_accuracy=accuracy_trace[-1])
                # sleep(0.1)
            if show and (iter_i + 1) % verbose == 0:
                show_progress(loss_trace, accuracy_trace, x, y, y_pred)

        test_accuracy = test(model, test_dataloader)
        print(f'test  accuracy: {test_accuracy:.3f}')

    return loss_trace[-1], accuracy_trace[-1]


# model = Smorph_Net(0, (1, 28, 28))
model = Morph_Net(6, (1, 28, 28))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train_dataloader, test_dataloader = mnist(100)

logs = "logs/torch/"
model_name = "BM_Net_6"
writer = tb.writer.SummaryWriter(path.join(logs, model_name))

train_loss, train_accuracy = train(model, train_dataloader, test_dataloader,
                                    criterion, optimizer, writer,
                                    n_epochs=5, show=True)


model = Morph_Net(7, (1, 28, 28))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train_dataloader, test_dataloader = mnist(100)

logs = "logs/torch/"
model_name = "BM_Net_7"
writer = tb.writer.SummaryWriter(path.join(logs, model_name))

train_loss, train_accuracy = train(model, train_dataloader, test_dataloader,
                                    criterion, optimizer, writer,
                                    n_epochs=5, show=True)


model = Morph_Net(8, (1, 28, 28))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train_dataloader, test_dataloader = mnist(100)

logs = "logs/torch/"
model_name = "BM_Net_8"
writer = tb.writer.SummaryWriter(path.join(logs, model_name))

train_loss, train_accuracy = train(model, train_dataloader, test_dataloader,
                                    criterion, optimizer, writer,
                                    n_epochs=5, show=True)
                                
                                
# writer.flush()
# writer.close()