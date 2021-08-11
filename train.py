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
from torch.jit.frontend import get_jit_ast, get_default_args

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
        input_size = (3, 32, 32)
    elif dataset == 'imagenet':
        if not for_inception:
            size = (3, 224, 224)
        else:
            size = (3, 299, 299)
    
    x = torch.rand(size=input_size).to(device)

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






















"""def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg
"""


"""def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)"""


# Set the logger
"""logger = Logger('./logs')

def train_model(model, train_loader, args, num_epochs=args.epochs,args.dataset,args.save):
    since = time.time()
    
    bar = Bar('Training', max=len(train_loader))
    model.train()
    
    loss_function = args.loss_function
    optimizer     = args.optimizer
    scheduler     = args.scheduler
    eval_interval = 5

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    total, correct, sum_loss = 0, 0, 0
    
    ittr = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                ittr = ittr + 1
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss    = loss_function(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)

            optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            sum_loss += loss*targets.size(0)
            correct  += predicted.eq(targets).sum().item()
             
            if phase == 'train':
              #============ TensorBoard logging ============#
              # Log the scalar values
              info = {
                        'loss-Train': sum_loss,
                        'accuracy-Train': correct}

              step = ittr - 1
              for tag, value in info.items():
                  logger.scalar_summary(tag, value, step+1)
                
              # backward + optimize only if in training phase
              if phase == 'train':
                  loss.backward()
                  optimizer.step()

            epoch_loss = sum_loss / total
            epoch_acc = correct / total
            #print(type(epoch_loss))
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_path = os.path.join(args.dataset, args.save)
                #print(save_path)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                filename = os.path.join( save_path, "bestmodel.pth.tar")
                torch.save({'state_dict': model.state_dict(),}, filename)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    #scheduler.step()
    bar.finish()
    return model"""