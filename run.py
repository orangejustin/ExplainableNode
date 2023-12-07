import warnings
warnings.filterwarnings("ignore")

# General import
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from torchvision import datasets, transforms, models
import wandb
import os

# Import from repo
from models.SimpleMLP import SimpleMLP
from train import *
from ExplainableNode import ExplainableNode  # Import ExplainableNode

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

config = {
    'epochs': 5,
    'batch_size': 30
}

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Download data if haven't
dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms)

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=config['batch_size'],
                                         num_workers=6,
                                         pin_memory=True,
                                         shuffle=True)

# Model init
model = SimpleMLP(num_input_nodes=32*32*3, num_hidden_nodes=10).to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.01)
scheduler =  ReduceLROnPlateau(optimizer, 'max', patience=1, factor=0.9)

# Create an instance of ExplainableNode
explainable_node = ExplainableNode(matrix_size=(4, 4), plot_interval=5)

# Pass ExplainableNode instance to experiment
experiment(model=model, train_loader=dataloader, val_loader=None, test_loader=None, optimizer=optimizer,
           scheduler=scheduler, criterion=criterion, config=config, device=device, explainable_node=explainable_node)
