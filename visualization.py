import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock
import quant_modules
from quant_utils import *
#import tensorflow.keras as keras
#from tensorflow.keras.utils import plot_model

def visual_weight(model):
    print('entering visual weight function')
    #print(model)
    with torch.no_grad():
        for name, layer in model.named_modules():
            #print(type(layer))
            #if isinstance(layer, nn.Conv2d):
            if isinstance(layer, quant_modules.Quant_Conv2d):
                #print(layer)
                weight = np.array(Parameter(layer.weight.data.clone()).cpu().detach().numpy())
                #=print(weight)
                #b,h,w,in_channel,out_channel = weight.shape
                out_channel,in_channel, h, w = weight.shape
                #print(b,h,w,in_channel,out_channel)
                #print(weight.shape)
                fig = plt.figure()
                for i in range(out_channel):
                    c = '#%02X%02X%02X' % (random.randint(0, 255),
                                        random.randint(0, 255),
                                        random.randint(0, 255))
                    ax = fig.add_subplot()
                    n, bins, patches = ax.hist(x=weight[:,i,:,:].reshape(in_channel*h*w), bins='auto', color=c,
                                                alpha=0.7, rwidth=0.8)
                    ax.grid(axis='y', alpha=0.75)
                    ax.set_xlabel('Weight Value')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Weight visualization for {} layer'.format(name))
                    #fig.savefig('/home/jovyan/Master_Thesis/Master_Thesis/classification/utils/plots/{}.png'.format(name))
                    #plt.cla()
                #plt.show()
                #plt.savefig(f'/home/jovyan/Samesame/image/weight.png')
                #fig.savefig(f'/home/jovyan/Master_Thesis/Master_Thesis/classification/utils/plots/weight_visualize.png')


