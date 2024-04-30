from configparser import Interpolation
from tracemalloc import start
import torch
import torch.nn as nn
import os
import copy
from torchvision import datasets, transforms
import torchvision.models as models
from torch.hub import load_state_dict_from_url
import numpy as np
import time
import sys
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt
import re
import argparse
from torchsummary import summary

def train(model, train_loader, optimizer, epoch):
    model.train()
    train_accs = []
    train_losses = []
    epoch_loss = []
    epoch_acc = []
    for data in tqdm(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        acc = outputs.argmax(dim=-1) == labels.to(device).float().mean()
        train_accs.append(acc.detach().item())
        train_losses.append(loss.detach().item())

        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        epoch_loss.append(train_loss)
        epoch_acc.append(train_acc)


def val(model, val_loader, epoch):
    model.eval()
    loss = 0.0
    correct = 0.0
    total = 0
    with torch.no_grad():
        for data in tqdm(val_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            valid_acc = correct / total
            print(f"[ Valid | loss = {loss:.5f}, acc = {valid_acc:.5f}")

if __name__ == '__main__':
    print("start training")

    '''
    GPU usage
    '''
    if torch.cuda.is_available():
        print("==============> Using GPU")
        device = 'cuda:0'
    else:
        print("==============> Using CPU")
        device = 'cpu'

    '''
    Seed
    '''
    myseed = 7414  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    '''
    Model
    '''
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 26)
    )
    model.train()

    '''
    Optimizer and scheduler
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 5e-4, nesterov = True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,70], gamma=0.1)

    '''
    Training Start
    '''
    n_epochs = 100
    for epoch in range(n_epochs):

        train(model, train_loader, optimizer, epoch)
        val(model, val_loader)
        scheduler.step()

        torch.save(model.state_dict(), f"model_{epoch}.pth")
        print(f"Model saved at epoch {epoch}")

