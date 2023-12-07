import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import os

scaler = torch.cuda.amp.GradScaler()


def train(model, dataloader, optimizer, criterion, explainable_node, epoch, device='cuda', add_new_pages=False, keep_only_last_step=False):
    model.train()
    explainable_node.register_hooks(model)

    running_loss = 0
    correct, total = 0, 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc='Training', leave=False)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        if batch_idx % explainable_node.plot_interval == 0:
            explainable_node.plot_values(epoch, batch_idx, add_new_pages=add_new_pages, keep_only_last_step=keep_only_last_step)
            explainable_node.clear_values()

    # Remove hooks after training
    for hook in model.hooks:
        hook.remove()
    del model.hooks

    explainable_node.save_plots(epoch)

    accuracy = correct / total
    return running_loss / len(dataloader), accuracy


def evaluate(model, dataloader, criterion, device='cuda', mode='Validate'):
    assert mode in ['Validate', 'Test'], "Mode should be either 'Validate' or 'Test'"
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    predictions = []

    # Deactivate autograd for evaluation
    with torch.no_grad():
        for data in tqdm(dataloader, desc=mode, leave=False):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Save predictions if in test mode
            if mode == 'Test':
                predictions.extend(predicted.cpu().numpy())

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    # Print evaluation results
    print(f'{mode} Loss: {avg_loss:.4f}, {mode} Accuracy: {accuracy:.4f}')

    # Return results (and predictions if in test mode)
    return (avg_loss, accuracy, predictions) if mode == 'Test' else (avg_loss, accuracy)


def experiment(model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion, config, device='cuda', matrix_size=(4, 4), add_new_pages=False, keep_only_last_step=False):
    best_val_acc = 0
    explainable_node = ExplainableNode(matrix_size=matrix_size, plot_interval=5)

    for epoch in range(config['epochs']):
        # Clear previous values at the start of each epoch
        explainable_node.clear_values()

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, explainable_node, epoch, device, add_new_pages=add_new_pages, keep_only_last_step=keep_only_last_step)

        print('Training:', train_loss, train_acc)

        # Additional code for validation and testing, if any
        # ...

    # Additional code for the rest of the experiment function
    # ...

