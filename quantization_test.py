#-------------------------------------------------------------------------------
# Author:     Namratha Sanjay
# Name:       test quantization
# Purpose:    This module defines all the functions used for generation of synthetic data.
# Copyright:   (c) Volvo cars 
# History:
#-------------------------------------------------------------------------------
import argparse
import random
import time
import numpy as np
#import plotly
#import plotly.graph_objects as go
#import plotly.io as pio
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchcv.model_provider import get_model as ptcv_get_model
from quantize_model import *
from data_utils import *
from train import *
from quant_utils import *
from visualization import *
#from torchsummary import summary


# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
                description='This repository contains the PyTorch implementation for the Quantization Framework.')
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet', 'cifar10'],
                        help='type of dataset')

    parser.add_argument('--data-source', type=str, default='distill',
                        choices=['distill', 'random', 'train'],
                        help='whether to use distill data, this will take some minutes')

    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet50', 'inceptionv3',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'sqnxt23_w2', 'vgg19'
                        ],
                        help='model to be quantized')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')
    args = parser.parse_args()
    return args

def kl_divergence(P, Q):
    return (P * (P / Q).log()).sum() / P.size(0) # batch size
    # F.kl_div(Q.log(), P, None, None, 'sum')
def symmetric_kl(P, Q):
    return (kl_divergence(P, Q) + kl_divergence(Q, P)) / 2

def plot_sen(sen, arch):
    trace0 = go.Scatter(
      y = sen[0],
      mode = 'lines + markers',
      name = '2bit'   )
    trace1 = go.Scatter(
        y = sen[1],
        mode = 'lines + markers',
        name = '4bit'   )
    trace2 = go.Scatter(
        y = sen[2],
        mode = 'lines + markers',
        name = '8bit'   )
    data = [trace0, trace1, trace2]

    #layout = go.Layout(
    #    title='{}'.format(arch),
    #    xaxis=dict(
    #        title='{} layer id'.format(arch),
    #    ),
    #    yaxis=dict(
    #        title='sensitivity of quantization',
    #        type='log'
    #    )
    #)
    #fig = go.Figure(data, layout)
    #if not os.path.exists('workspace/images'):
    #    os.makedirs('workspace/images')
    #fig.write_image('workspace/images/{}_sen.png'.format(arch))

def random_sample(sen_result, quan_weight, weight_num):         #quan_weight: no of quantized weight layers, weight_num: no of weight elements in each quantized layer
    bit_ = [2,4,8]
    random_code = [random.randint(0,2) for i in range(len(quan_weight))]
    #print(random_code)
    sen = 0
    size = 0
    for i, bit in enumerate(random_code):
        sen += sen_result[bit][i]
    size = sum(weight_num[l] * bit_[i] / 8 / 1024 / 1024 for (l, i) in enumerate(random_code))
    return size, sen

class Node:
    def __init__(self, cost=0, profit=0, bit=None, parent=None, left=None, middle=None, right=None, position='middle'):
        self.parent = parent
        self.left = left
        self.middle = middle
        self.right = right
        self.position = position
        self.cost = cost
        self.profit = profit
        self.bit = bit
    def __str__(self):
        return 'cost: {:.2f} profit: {:.2f} bit: {:.2f}'.format(self.cost, self.profit,self.bit)
    def __repr__(self):
        return self.__str__()
    

def get_FrontierFrontier(sen_result, layer_num, weight_num, constraint=1000):
    bits = [2, 4, 8]
    cost = [2, 4, 8]
    prifits = []
    for line in sen_result:
        prifits.append([-i for i in line])
    #print(sen_result)
    root = Node(cost=0, profit=0, parent=None)
    current_list = [root]
    for layer_id in range(layer_num):
        # 1. split
        next_list = []
        for n in current_list:
            n.left = Node(n.cost + cost[0] * weight_num[layer_id] / 8 / 1024 / 1024, n.profit + prifits[0][layer_id], bit=bits[0], parent=n, position='left')
            n.middle = Node(n.cost + cost[1] * weight_num[layer_id] / 8 / 1024 / 1024, n.profit + prifits[1][layer_id], bit=bits[1], parent=n, position='middle')
            n.right = Node(n.cost + cost[2] * weight_num[layer_id] / 8 / 1024 / 1024, n.profit + prifits[2][layer_id], bit=bits[2], parent=n, position='right')
            next_list.extend([n.left, n.middle, n.right])
        #print(next_list)
        # 2. sorting the frontier according to optimal solution
        next_list.sort(key=lambda x:x.cost, reverse=False)
        # 3. remove the least optimal solutions
        pruned_list = []
        for node in next_list:
            if (len(pruned_list) == 0 or pruned_list[-1].profit < node.profit) and node.cost <= constraint:
                pruned_list.append(node)
            else:
                node.parent.__dict__[node.position] = None
        # 4. loop 
        current_list = pruned_list
    return current_list

def sensitivity_anylysis(quan_act, quan_weight, dataloader, quantized_model, args, weight_num):
    # 1. get the baseline value for comparison
    for l in quan_act:
        l.full_precision_flag = True
    for l in quan_weight:
        l.full_precision_flag = True
    inputs = None
    gt_output = None
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            if isinstance(inputs, list):
                inputs = inputs[0]
            inputs = inputs.cuda()
            gt_output = quantized_model(inputs)
            gt_output = F.softmax(gt_output, dim=1) #outputs in range [0,1]
            #print(gt_output)
            break
    # 2. change bitwidth layer by layer and get the sensitivity
    sen_result = [[0 for i in range(len(quan_weight))] for j in range(3)]
    for i in range(len(quan_weight)):
        for j, bit in enumerate([2,4,8]):
            quan_weight[i].full_precision_flag = False
            quan_weight[i].bit = bit
            #print(quan_weight[i].bit)
            with torch.no_grad():
                tmp_output = quantized_model(inputs)
                tmp_output = F.softmax(tmp_output, dim=1)
                #print('temp_',len(tmp_output))
                kl_div = symmetric_kl(tmp_output, gt_output)
                #print(kl_div)
            sen_result[j][i] = kl_div.item()
            quan_weight[i].full_precision_flag = True
    #print(sen_result)
    #plot_sen(sen_result, args.model)
    # 3. Pareto Frontier
    ## random
    sizes = []
    sens = []
    for i in range(1000):
        size, sen = random_sample(sen_result, quan_weight, weight_num)
        sizes.append(size)
        sens.append(sen)
    #trace_random = go.Scatter(x=sizes, y=sens, mode='markers', name='random')
    #layout = go.Layout(
    #    title='{}'.format(args.model),
    #    xaxis=dict(
    #        title='{} size (MB)'.format(args.model),
    #    ),
    #    yaxis=dict(
    #        title='sensitivity',
    #        type='log'
    #    )
    #)
    begin = time.time()
    ## DP
    node_list = get_FrontierFrontier(sen_result, len(quan_weight), weight_num)
    print('dp cost: {:.2f}s'.format(time.time() - begin))
    sizes = [x.cost for x in node_list]
    sens = [ -x.profit for x in node_list]
    #trace = go.Scatter(x=sizes, y=sens, mode='markers+lines', name='Frontier Frontier', marker={"size": 3})
    #data = [trace, trace_random]
    #fig = go.Figure(data, layout)
    #fig.write_image('workspace/images/{}_Pareto.png'.format(args.model))
    #fig.write_image('workspace/images/{}_Pareto.pdf'.format(args.model))
    return node_list

#def plot_bits(bits, name):
#    trace = go.Scatter(y=bits, mode='markers+lines')
#    layout = go.Layout(
#        title=name,
#        xaxis=dict(title='size (MB)'),
#        yaxis=dict(title='bits of weight'))
#    data = [trace]
#    fig = go.Figure(data, layout)
#    fig.write_image('workspace/images/{}_bit.png'.format(name))
#    fig.write_image('workspace/images/{}_bit.pdf'.format(name))
    


if __name__ == '__main__':
    args = arg_parse()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    # Load pretrained model
    model = ptcv_get_model(args.model, pretrained=True)
    print('****** Baseline model loaded ******')

    # Load validation data
    test_loader = getTestData(args.dataset,
                              batch_size=args.test_batch_size,
                              path='/home/jovyan/new-quant-static-vol-1/',
                              for_inception=args.model.startswith('inception'))
    begin = time.time()
    # Load training data
    if args.data_source == 'train':
        print('Get train data')
        dataloader = getTrainData(args.dataset,
                                  batch_size=args.batch_size,
                                  path='/home/jovyan/new-quant-static-vol-1/',
                                  for_inception=args.model.startswith('inception'))

    if args.data_source == 'random':
        print('load random data from imagenet dataset')
        dataloader = getRandomData(dataset=args.dataset,
                                   batch_size=args.batch_size,
                                   for_inception=args.model.startswith('inception'))
    
    print('****** Data loaded ****** cost {:.2f}s'.format(time.time() - begin))
    begin = time.time()
    # Quantize single-precision model to 8-bit model
    
    quan_tool = QuanModel()
    quantized_model = quan_tool.quantize_model(model)
    # Freeze BatchNorm statistics
    quantized_model.eval()
    quantized_model = quantized_model.cuda()
    #freeze_model(quantized_model)
    #quantized_model = nn.DataParallel(quantized_model).cuda()
    #test(quantized_model, test_loader)
    #print(quantized_model)
    node_list = sensitivity_anylysis(quan_tool.quan_act_layers, quan_tool.quan_weight_layers, dataloader, quantized_model, args, quan_tool.weight_num)
    config = {
        'resnet18': [(8, 8),(6,6),(4, 8), (4,4)], # representing MP6 for weights and 6bit for activation
        'resnet50': [(4,4)],
        'resnet20_cifar10':[(8, 8),(6,6),(4, 8), (4,4)],
        'mobilenetv2_w1': [(8, 8),(6,6),(4, 8), (4,4)],
        'shufflenet_g1_w1': [(4,4)],
        'inceptionv3': [(8, 8),(6,6),(4, 8), (4,4)],
        'sqnxt23_w2': [(8, 8),(6,6),(4, 8), (4,4)]
    }
    for (bit_w, bit_a) in config[args.model]:
        for l in quan_tool.quan_act_layers:
            l.full_precision_flag = False
            l.bit = bit_a
            #print('activation bit',l.bit)
        constraint = sum(quan_tool.weight_num) * bit_w / 8 / 1024 / 1024
        #print(sum(quan_tool.weight_num))
        meet_list = []
        for node in node_list:
            #print('node----',node)
            if node.cost <= constraint:
                meet_list.append(node)
        bits = []
        node = meet_list[-1] #last element
        #print('node', node)
        while(node is not None):
            bits.append(node.bit)
            #print(bits)
            #print(len(bits))
            node = node.parent
            #print('node---->',node)
        bits.reverse() #remove None
        bits = bits[1:] #remove None from list
        #plot_bits(bits, '{}_MP{}A{}'.format(args.model, bit_w, bit_a))
        for i, l in enumerate(quan_tool.quan_weight_layers):
            l.full_precision_flag = False
            l.bit = bits[i]
            #print(i,l.bit)
        # Update activation range according to distilled data
        unfreeze_model(quantized_model)
        update(quantized_model, dataloader)
        print('****** Quantization Finished ****** cost {:.2f}s'.format(time.time() - begin))
        #print(quantized_model)
        #visual_weight(quantized_model)
        #visual_weight(model)
        # Freeze activation range during test
        freeze_model(quantized_model)
        quantized_model = nn.DataParallel(quantized_model).cuda()
        #print(quantized_model)

        # Test the final quantized model
        print('size: {:.2f} MB Wmp{}A{}'.format(constraint, bit_w, bit_a))
        test(quantized_model, test_loader)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #test inference latency
        #fp32_gpu_inference_latency = measure_inference_latency(model=model,
        # device=cuda_device,
        # input_size=(1, 3, 224, 224),
        # num_samples=100)

        #int8_gpu_inference_latency = measure_inference_latency(model=quantized_model,
        # device=cuda_device,
        # input_size=(1, 3, 224, 224),
        # num_samples=100)

        #print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(
        #fp32_cpu_inference_latency * 1000))
        #print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(
        #fp32_gpu_inference_latency * 1000))
        #print("INT8 CUDA Inference Latency: {:.2f} ms / sample".format(
        #int8_gpu_inference_latency * 1000))
         #print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(
        #int8_jit_cpu_inference_latency * 1000))

        
        
        
