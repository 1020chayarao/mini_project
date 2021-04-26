# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 19:26:12 2021

@author: chaya

main model mini project
"""
#importing library
import pandas as pd
import numpy as np
# open source Deep Leaning library
import tensorflow as tf 
from tensorflow import keras
# For plotting figures
import matplotlib as mpl 
import matplotlib.pyplot as plt

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

mpl.rc('axes', labelsize=15)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


#load mnist data
# Load MNIST Data
kannada_test = pd.read_csv("C:/Users/chaya/Desktop/mini/datasets/test.csv")
kannada_train = pd.read_csv("C:/Users/chaya/Desktop/mini/datasets/train.csv")

kannada_train.head(3)

print(kannada_train.shape)
print(kannada_test.shape)

num = 4
plot_num = kannada_train.iloc[num, 1:]
plot_num = np.array(plot_num).reshape(28, -1)
plt.imshow(plot_num, cmap='gray')
plt.title(f'Label: {kannada_train.iloc[num, 0]}')
plt.show()

#split train indices
def split_indices(n, val_pct):
    n_val = int(n*val_pct)
    idx = np.random.permutation(n)
    return idx[:n_val], idx[n_val:]

val_idx, train_idx = split_indices(len(kannada_train), 0.1)
#labelling
labels = kannada_train.pop('label')
labels_train = labels[train_idx]
labels_val = labels[val_idx]


#setting hyper parameters
batch_size_train = 64
batch_size_val = 1024

learning_rate = 0.01
momentum=0.5
log_interval = 10

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_sampler = SubsetRandomSampler(train_idx)
train_loader = DataLoader(kannada_train, batch_size_train, sampler = train_sampler)

val_sampler = SubsetRandomSampler(val_idx)
val_loader = DataLoader(kannada_train, batch_size_val, sampler = val_sampler)

train_X = kannada_train.iloc[train_idx, :].values
val_X = kannada_train.iloc[val_idx, :].values

train_X = torch.Tensor(train_X.reshape(train_X.shape[0],1,28,-1))
train_y = torch.Tensor(labels_train)

val_X = torch.Tensor(val_X.reshape(val_X.shape[0],1,28,-1))
val_y = torch.Tensor(labels_val.values)

train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)

train_loader = DataLoader(train_dataset, batch_size = batch_size_train, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size_val, shuffle = True)

sample = enumerate(val_loader)
idx, (sample_data, sample_labels) = next(sample)

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(sample_data[i+11].view(28, -1), cmap = 'gray', interpolation='none')
    plt.title(f'Ground truth: {sample_labels[i+11]}')
    plt.xticks([])
    plt.yticks([])
    
input_size = 28 * 28
num_classes = 10

model = nn.Linear(input_size, num_classes)    
class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size = 5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size = 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(640, 120)
        self.fc2 = nn.Linear(120, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 640)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
network = net() #instantiate network
optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum = momentum)

train_losses = [] 
train_counter = []
val_losses = []
val_counter = []
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target.type(torch.LongTensor))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] \
            {100*batch_idx/len(train_loader):.0f},\tLoss: {loss.item():.6f}')
            train_losses.append(loss.item())
            train_counter.append(batch_idx*64 + (epoch-1)*len(train_loader.dataset))

def val():
    network.eval()
    val_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            output = network(data)
            val_loss += F.nll_loss(output, target.type(torch.LongTensor), size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    print(f'Val set: Avg loss {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)}, \
    ({100*correct/len(val_loader.dataset)})')               
        
            
n_epochs = 3
val()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    val()
