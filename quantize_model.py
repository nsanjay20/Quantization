#-------------------------------------------------------------------------------
# Author:     Namratha Sanjay
# Name:       perform quantization
# Purpose:    This module perform quantization.
# Copyright:   (c) Volvo cars 
# History:
#-------------------------------------------------------------------------------
import torch
import torch.nn as nn
import copy
from quant_modules import *
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock

class QuanModel():
    def __init__(self):
        # collect convq, linearq and actq for sensitivity analysis.
        self.quan_act_layers = []
        self.quan_weight_layers = []
        self.weight_num = [] # TODO
        #self.percentile = percentile

    def quantize_model(self, model, integer_only=True):
        """
        Recursively quantize a pretrained single-precision model to int8 quantized model
        model: pretrained single-precision model
        """
        # quantize convolutional and linear layers to 8-bit
        if type(model) == nn.Conv2d:
            if integer_only:
                print('integer_only')
                quant_mod = Quant_Conv2d_Int(weight_bit=8)
            else:
                print('not QAT')
                quant_mod = Quant_Conv2d(weight_bit=8)
            quant_mod.set_param(model)
            self.quan_weight_layers.append(quant_mod)
            self.weight_num.append(quant_mod.weight.numel())
            return quant_mod
            
        elif type(model) == nn.Linear:
            if integer_only:
                quant_mod = Quant_Linear_Int(weight_bit=8)
            else:
                quant_mod = Quant_Linear(weight_bit=8)
            quant_mod.set_param(model)
            self.quan_weight_layers.append(quant_mod)
            self.weight_num.append(quant_mod.weight.numel())
            return quant_mod

        # quantize all the activation to 8-bit
        if type(model) == nn.ReLU or type(model) == nn.ReLU6:
            if integer_only:
                return nn.Sequential(*[model, QuantAct(activation_bit=8)])
            else:
                return nn.Sequential(*[model, QuantAct(activation_bit=8)])
            self.quan_act_layers.append(quant_mod)
            return nn.Sequential(*[model, quant_mod])

        # recursively use the quantized module to replace the single-precision module
        elif type(model) == nn.Sequential:
            mods = []
            for n, m in model.named_children():
                mods.append(self.quantize_model(m))
            return nn.Sequential(*mods)
        else:
            q_model = copy.deepcopy(model)
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    setattr(q_model, attr, self.quantize_model(mod))
            return q_model


def freeze_model(model):
    """
    freeze the activation range
    """
    if type(model) == QuantAct:
        model.fix()
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            freeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                freeze_model(mod)
        return model


def unfreeze_model(model):
    """
    unfreeze the activation range
    """
    if type(model) == QuantAct:
        model.unfix()
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            unfreeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                unfreeze_model(mod)
        return model
