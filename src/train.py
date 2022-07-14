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
    # trainset = torchvision.datasets.FashionMNIST(root='./data/f_mnist', train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
    # testset = torchvision.datasets.FashionMNIST(root='./data/f_mnist', train=False, download=True, transform=transform)
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

def test(model, criterion, test_dataloader):

    model.eval()
    total_accuracy = 0.0
    total_loss = []
    total = 0
    
    with torch.no_grad():
        with tqdm(test_dataloader, unit="batch") as tepoch:
            for x, y in tepoch:            
                x, y = x.to(device), y.to(device)            
                pred = model(x)
                y_pred = predict(pred)
                total_accuracy += (y_pred == y).sum()
                total += len(y)
                loss = criterion(pred, y)
                total_loss.append(loss.item())
                tepoch.set_postfix(loss=loss.item(), accuracy = (total_accuracy / total).item())
  
    total_accuracy /= total
    # total_loss /= total
    return total_accuracy.item(), sum(total_loss) / len(total_loss)
    
def train(model, train_dataloader, test_dataloader, criterion, 
    optimizer, scheduler, writer=None, n_epochs=1, show=False, verbose=10):
    
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
                    train_stat.append(loss_trace[-1], accuracy_trace[-1], [0], [0], iter_i)

                tepoch.set_postfix(loss=loss_trace[-1], mean_accuracy = mean_acc, batch_accuracy=accuracy_trace[-1])
                # sleep(0.1)
            if show and (iter_i + 1) % verbose == 0:
                show_progress(epoch_i, loss_trace, accuracy_trace, x, y, y_pred)
        
        test_accuracy, total_loss = test(model, criterion, test_dataloader)

        if (n_epochs != 1 and writer is not None):
            train_stat.append(loss_trace[-1], mean_acc, total_loss, test_accuracy, epoch_i)

        scheduler.step()
        # test_accuracy = test(model, test_dataloader)
        # print(f'test  accuracy: {test_accuracy:.3f}')

    return loss_trace[-1], accuracy_trace[-1]



def train_multilayer(depth, epochs, batch_size=100, name="BM_NET", with_logs = False, save_params = False):
    stats = {}
    for d in depth:
        # model = Smorph_Net(d, (1, 28, 28))
        model = CNN_Net(d, (1, 28, 28))
        # model = Test_smorph(d, (1, 28, 28))
        # model = Test_smorph(d, (1, 28, 28))
        # layers = model.layers
        # print(layers)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 160, 190], gamma=0.1)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        dump_file = None
        writer = None

        dt = datetime.now()
        str_date_time = dt.strftime("%d-%m-%Y_%H:%M:%S")
        model_name = name + "_" + str(d) + "_" + str_date_time
        if with_logs:
            logs = "logs/"
            writer = tb.writer.SummaryWriter(path.join(logs, model_name))

        # if save_params:
        #     # params = "params/"
        #     # dump_file = open(path.join(params, model_name), "w")

        #     models = "models/"
        #     path_file = path.join(models, model_name + ".pt")
        #     torch.save(model.state_dict(), path_file)

        train_dataloader, test_dataloader = mnist(batch_size)

        train_loss, train_accuracy = train(model, train_dataloader, test_dataloader,
                                    criterion, optimizer, scheduler, writer,
                                    n_epochs=epochs, show=False)

        if dump_file is not None:
            for i in range((d+1) * 2):
                if i % 2 == 0:
                    data = "alpha of layer " + str(i) + " = " + str(model.net[i].alpha) + "\n"
                    dump_file.write(data)
            dump_file.close()

        if save_params:
            models = "models/"
            dump_file = path.join(models, model_name + "_trained.pt")
            torch.save(model.state_dict(), dump_file)

        stats[d] = (train_loss, train_accuracy)
    return model


depths = [1, 2, 3, 4]
model = train_multilayer(depth=depths, epochs=50, batch_size=100, name="KDCNN", with_logs=False, save_params=True)
