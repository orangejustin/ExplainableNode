import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import os
from ExplainableNode import ExplainableNode  # Import ExplainableNode
import wandb

scaler = torch.cuda.amp.GradScaler()

def train(model, dataloader, optimizer, criterion, device, explainable_node=None):
    model.train()
    if explainable_node:
        explainable_node.register_hooks(model)

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc='Training'):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        if explainable_node and batch_idx % explainable_node.plot_interval == 0:
            explainable_node.plot_values(epoch, batch_idx)
            explainable_node.clear_values()

    if explainable_node:
        for hook in model.hooks:
            hook.remove()
        del model.hooks

    return running_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return running_loss / len(dataloader), correct / total

def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total



def experiment(model, train_loader, val_loader, optimizer, scheduler,
               criterion, config, device='cuda', explainable_node=None):

    wandb.login(key="c06ce00d5f99f931dfcbf6b470908fe8de32451c")
    run = wandb.init(
        name="Resnet-Benchmark",
        reinit=True,
        # id = 'zgni61f0',
        # resume = "must",
        project="ExplainableNode",
        config=config
    )

    best_val_acc = 0

    for epoch in range(config['epochs']):
        if explainable_node:
            explainable_node.clear_values()

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, explainable_node=None)
        print(
            f'Epoch [{epoch + 1}/{config["epochs"]}] - Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')

        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(
                f'Epoch [{epoch + 1}/{config["epochs"]}] - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

            # Update the best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # Step the scheduler based on the validation loss
            if scheduler is not None:
                scheduler.step(val_loss)

            curr_lr = float(optimizer.param_groups[0]['lr'])
            wandb.log({"train_loss": train_loss, 'train_Acc': train_acc, 'validation_Acc': val_acc,
                        'validation_loss': val_loss, "learning_Rate": curr_lr})
    if explainable_node:
        explainable_node.save_plots(epoch)

    # run.finish()
    return best_val_acc
