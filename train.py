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