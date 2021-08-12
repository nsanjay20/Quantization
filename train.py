#-------------------------------------------------------------------------------
# Author:     Namratha Sanjay
# Name:       evaluate quantized model
# Purpose:    This module evaluate quantized model.
# Copyright:   (c) Volvo cars 
# History:
#-------------------------------------------------------------------------------
import torch
import os
import torch.nn as nn
from progress.bar import Bar
import random
import time


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()

        self.num_classes = num_classes
        self.epsilon     = epsilon
        self.logsoftmax  = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets   = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets   = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss      = (-targets * log_probs).mean(0).sum()
        return loss

def train(model, train_loader, args, test_loader):
    """
    train a model on given dataset
    """
    total, correct, sum_loss = 0, 0, 0
    bar = Bar('Training', max=len(train_loader))
    model.train()

    loss_function = args.loss_function
    optimizer     = args.optimizer
    scheduler     = args.scheduler
    eval_interval = 5

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss    = loss_function(outputs, targets)
        _, predicted = outputs.max(1)
        total += targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss*targets.size(0)
        correct  += predicted.eq(targets).sum().item()
        acc       = correct / total
        loss      = sum_loss / total

        bar.suffix = f'({batch_idx + 1}/{len(train_loader)}) | ETA: {bar.eta_td} | top1: {acc} | loss:{loss}'
        bar.next()
        if (batch_idx+1) % eval_interval == 0:
            acc = test(model, test_loader)
    scheduler.step()
    bar.finish()
    return acc, loss

def test(model, test_loader):
    """
    test a model on a given dataset
    """
    total, correct = 0, 0
    bar = Bar('Testing', max=len(test_loader))
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = correct / total

            bar.suffix = f'({batch_idx + 1}/{len(test_loader)}) | ETA: {bar.eta_td} | top1: {acc}'
            bar.next()
    print('\nFinal acc: %.2f%% (%d/%d)' % (100. * acc, correct, total))
    bar.finish()
    model.train()
    return acc


def update(quantized_model, distilD):
    """
    Update activation range according to distilled data
    quantized_model: a quantized model whose activation range to be updated
    distilD: distilled data
    """
    with torch.no_grad():
        for batch_idx, inputs in enumerate(distilD):
            if isinstance(inputs, list):
                inputs = inputs[0]
            inputs = inputs.cuda()
            outputs = quantized_model(inputs)
    return quantized_model



def measure_inference_latency(model, dataset, num_samples=100, num_warmups=10, for_inception=False):

    cuda_device = torch.device("cuda:0")
    model.to(cuda_device)
    model.eval()

    if dataset == 'cifar10':
        input_size = (1, 3, 32, 32)
    elif dataset == 'imagenet':
        if not for_inception:
            input_size = (1, 3, 224, 224)
        else:
            input_size = (1, 3, 299, 299)
    
    x = torch.rand(size=input_size).to(cuda_device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model

def save_torchscript_model(model, model_dir, model_filename, device):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath,  map_location=device)


def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model


def measure_throughput(model, dataset, for_inception=False):

  cuda_device = torch.device("cuda:0")
  device = torch.device(cuda_device)
  model.to(device)

  if dataset == 'cifar10':
        input_size = (1, 3, 32, 32)
  elif dataset == 'imagenet':
        if not for_inception:
            input_size = (1, 3, 224, 224)
        else:
            input_size = (1, 3, 299, 299)
  optimal_batch_size=512
  dummy_input = torch.randn(optimal_batch_size, 3,224,224, dtype=torch.float).to(device)
  repetitions=100
  total_time = 0
  with torch.no_grad():
    for rep in range(repetitions):
      starter, ender = torch.cuda.Event(enable_timing=True),          torch.cuda.Event(enable_timing=True)
      starter.record()
      _ = model(dummy_input)
      ender.record()
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)/1000
      total_time += curr_time
  Throughput = (repetitions*optimal_batch_size)/total_time
  #print('Final Throughput':,Throughput)
  return Throughput



def evaluate_model(model, test_loader, device, criterion=None):
    #evaluate quantized model
    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)
    return eval_loss, eval_accuracy

def train_model(model, train_loader, test_loader, device, args, num_epochs=200):
    # The training configurations were not carefully selected.
    loss_function = args.loss_function
    optimizer     = args.optimizer
    scheduler     = args.scheduler

    model.to(device)

    for epoch in range(num_epochs):
        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward 
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            #backward + optimize
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model,
                                                    test_loader=test_loader,
                                                    device=device,
                                                    criterion=loss_function)

        # Set learning rate scheduler
        scheduler.step()

        print(
            "Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}"
            .format(epoch + 1, train_loss, train_accuracy, eval_loss,
                    eval_accuracy))

    return model









