#-------------------------------------------------------------------------------
# Author:     Namratha Sanjay
# Name:       perform quantization
# Purpose:    This module perform quantization.
# Copyright:   (c) Volvo cars 
# History:
#-------------------------------------------------------------------------------

import argparse
import torch
import numpy as np
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from quant_utils import *
from data_utils import *
from train import *
from quantize_model import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import gc
gc.collect()
torch.cuda.empty_cache()


# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet', 'cifar10'],
                        help='type of dataset')
    parser.add_argument('--model',
                        type=str,
                        default='mobilenetv2_w1',
                        choices=[
                            'resnet18', 'resnet50', 'inceptionv3',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'resnet56_cifar10',
                            'resnext29_32x4d_cifar10', 'sqnxt23_w2'
                        ],
                        help='model to be quantized')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=256,
                        help='batch size of test data')

    parser.add_argument('--train_batch_size',
                        type=int,
                        default=256,
                        help='batch size of train data')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help="Training epochs")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-5)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=5e-4)
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9)
    parser.add_argument("--lr-decay",
                        type=float,
                        default=0.1)
    parser.add_argument("--save",
                        type=str,
                        default="model_path")
    parser.add_argument("--eval-interval",
                        type=int,
                        default=1,
                        help="the evaluation interval during training")
    parser.add_argument("--init-test",
                        type=bool,
                        default=0,
                        help="whether test the initial result w/o QAT")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Load pretrained model
    model = ptcv_get_model(args.model, pretrained=True).cuda()
    print('****** Full precision model loaded ******')

    # Load training data
    train_loader = getTrainData(args.dataset,
                              batch_size=args.train_batch_size,
                              path='/home/jovyan/new-quant-static-vol-1/',
                              for_inception=args.model.startswith('inception'))
    # Load validation data
    test_loader = getTestData(args.dataset,
                              batch_size=args.test_batch_size,
                              path='/home/jovyan/new-quant-static-vol-1/',
                              for_inception=args.model.startswith('inception'))

    print('****** Data loaded ******')

    if args.dataset == "cifar10":
        criterion_smooth = CrossEntropyLabelSmooth(10, 0.1).cuda()
    elif args.dataset == "imagenet":
        criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1).cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[3, 5, 8],
        gamma=args.lr_decay
    )

    args.loss_function = criterion_smooth
    args.optimizer     = optimizer
    args.scheduler     = scheduler

    quan_tool = QuanModel()
    quantized_model = quan_tool.quantize_model(model, integer_only=True)
    quantized_model = nn.DataParallel(quantized_model).cuda()
    quantized_model.train()

    best_acc = 0
    if args.init_test:
        acc = test(model, test_loader)
        print('FP model accuracy', acc)

    for epoch in range(args.epochs):
        print('epoch no', epoch)
        acc, loss = train(quantized_model, train_loader, args, test_loader)
        print(f"Epoch {epoch}: loss = {loss:4f}, top1_accuracy = {acc*100:4f}, learning_rate = {args.scheduler.get_last_lr()[0]}")
        if (epoch+1) % args.eval_interval == 0:
            acc = test(quantized_model, test_loader)
            
            if acc > best_acc:
                print('acc:{}, best_acc:{}'.format(acc,best_acc))
                best_acc = acc
                save_path = os.path.join(args.dataset, args.save)
                #print(save_path)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                filename = os.path.join( save_path, "bestmodel.pth.tar")
                torch.save({'state_dict': model.state_dict(),}, filename)


