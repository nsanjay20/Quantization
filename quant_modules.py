#-------------------------------------------------------------------------------
# Author:     Namratha Sanjay
# Name:       perform quantization
# Purpose:    This module perform quantization.
# Copyright:   (c) Volvo cars 
# History:
#-------------------------------------------------------------------------------

import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from quant_utils import *
import sys


class QuantAct(Module):
    """
    Class to quantize given activations
    """
    def __init__(self,
                 activation_bit,
                 full_precision_flag=False,
                 running_stat=True,
                 integer_only=True):
        """
        activation_bit: bit-setting for activation
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.act_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2}, running_stat={3}, Act_min: {4:.2f}, Act_max: {5:.2f})".format(
            self.__class__.__name__, self.activation_bit,
            self.full_precision_flag, self.running_stat, self.x_min.item(),
            self.x_max.item())

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def unfix(self):
        self.running_stat = True

    def forward(self, x):
        """
        quantize given activation x
        """
        if self.running_stat:
            x_min = x.data.min()
            x_max = x.data.max()
            # in-place operation used on multi-gpus
            self.x_min += -self.x_min + min(self.x_min, x_min)
            self.x_max += -self.x_max + max(self.x_max, x_max)

        if self.full_precision_flag:
            return x
        else:
            quant_act = self.act_function(x, self.activation_bit, self.x_min,
                                          self.x_max)
            return quant_act


class Quant_Linear(Module):
    """
    Class to quantize given linear layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(Quant_Linear, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        s = super(Quant_Linear, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.detach()
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        if self.full_precision_flag:
            w = self.weight
        else:
            w = self.weight_function(self.weight, self.weight_bit, w_min, w_max)
        
        return F.linear(x, weight=w, bias=self.bias)


class Quant_Conv2d(Module):
    """
    Class to quantize given convolutional layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        super(Quant_Conv2d, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        s = super(Quant_Conv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.contiguous().view(self.out_channels, -1)
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        if self.full_precision_flag:
            w = self.weight
        else:
            w = self.weight_function(self.weight, self.weight_bit, w_min,
                                     w_max)   

        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)



# Integer Only Implementation
class QuantAct_Int(Module):
    """
    Class to quantize given activations
    """
    def __init__(self,
                 activation_bit,
                 full_precision_flag=False,
                 running_stat=False,
                 integer_only=True):
        """
        activation_bit: bit-setting for activation
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(QuantAct_Int, self).__init__()
        self.activation_bit = activation_bit
        self.momentum = 0.99
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))

    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2}, running_stat={3}, Act_min: {4:.2f}, Act_max: {5:.2f})".format(
            self.__class__.__name__, self.activation_bit,
            self.full_precision_flag, self.running_stat, self.x_min.item(),
            self.x_max.item())

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def forward(self, x):
        """
        quantize given activation x
        """
        if self.running_stat:
            x_min = x.data.min()
            x_max = x.data.max()
            # in-place operation used on multi-gpus
            self.x_min += -self.x_min + min(self.x_min, x_min)
            self.x_max += -self.x_max + max(self.x_max, x_max)

        if self.full_precision_flag:
            return x
        else:
            return x, self.activation_bit, self.x_min, self.x_max
        
            


class Quant_Linear_Int(Module):
    """
    Class to quantize given linear layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(Quant_Linear_Int, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit          = weight_bit

    def __repr__(self):
        s = super(Quant_Linear_Int, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None
        self.quantfunc   = LinearQuantizeModule()
        self.dequantfunc = LinearDequantizeModule()

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.detach()
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values

        x_max = x.data.detach().max()
        n = 2**(self.weight_bit)
        scale_x = (n-1) / torch.clamp(x_max, min=1e-8)

        if self.full_precision_flag:
            w = self.weight
            return F.linear(x, weight=w, bias=self.bias)
        else:
            # x is asymmetric quantization with range [0,255]
            # new_quant_x = linear_quantize(x, scale_x, torch.zeros(1).cuda())
            new_quant_x = self.quantfunc(x, scale_x, torch.zeros(1).cuda())
            new_quant_w, scale_w, zero_point_w = quantize_int(self.weight, self.weight_bit, w_min, w_max)

            # bias is symmetric quantization with range [-128,127]
            # new_quant_b = linear_quantize(self.bias, scale_w*scale_x, 0)
            new_quant_b = self.quantfunc(self.bias, scale_w*scale_x, 0)
            new_quant_b = torch.clamp(new_quant_b, -n - 1, n)

            # TODO bias quantization
            # print(x.shape, new_quant_x.shape, new_quant_w.shape, new_quant_b.shape)
            mult_res = F.linear(new_quant_x, weight=new_quant_w, bias=new_quant_b)

            res = mult_res + zero_point_w * new_quant_x.sum(-1).unsqueeze(-1).expand_as(mult_res)

            return res / scale_x / scale_w
            # return self.dequantfunc(res, scale_x, scale_w)

class Quant_Conv2d_Int(Module):
    """
    Class to quantize given convolutional layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        super(Quant_Conv2d_Int, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        s = super(Quant_Conv2d_Int, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None
        self.quantfunc   = LinearQuantizeModule()
        self.dequantfunc = LinearDequantizeModule()

    def forward(self, x):

        """
        using quantized weights to forward activation x
        """
        calc_w = self.weight
        x_transform = calc_w.data.contiguous().view(self.out_channels, -1)
        w_max = x_transform.max(dim=1).values
        w_min = x_transform.min(dim=1).values
        # range[-127,127]
        n_w = (2**(self.weight_bit-1) - 1)
        scale_w = n_w / torch.clamp( torch.max(abs(w_max), abs(w_min)), min=1e-8)
        # scale_w = torch.ones(scale_w.shape).cuda()

        # n_x = (2**(self.weight_bit) - 1)
        n_x = n_w
        x_max = x.data.detach().max()
        x_min = x.data.detach().min()
        scale_x = n_x / torch.clamp( max(abs(x_max), abs(x_min)), min=1e-8)


        if self.full_precision_flag:
            w = self.weight
            return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
        else:
            # scale-only asymmetric
            # new_quant_x = linear_quantize(x, scale_x, torch.zeros(1).cuda())
            new_quant_x = self.quantfunc(x, scale_x, torch.zeros(1).cuda())
            new_quant_x = torch.clamp(new_quant_x, -n_x, n_x)
            # scale-only symmetric

            # new_quant_w = linear_quantize(self.weight, scale_w, torch.zeros(1).cuda())
            new_quant_w = self.quantfunc(self.weight, scale_w, torch.zeros(1).cuda())
            new_quant_w = torch.clamp(new_quant_w, -n_w, n_w)

            if self.bias is not None:
                new_quant_b = self.quantfunc(self.bias, scale_w*scale_x, torch.zeros(1).cuda())
                new_quant_b = torch.clamp(new_quant_b, -n_w, n_w)
            else:
                new_quant_b = None

            # TODO bias quantization
            # print(x.shape, new_quant_x.shape, new_quant_w.shape, new_quant_b.shape)
            mult_res = F.conv2d(new_quant_x, new_quant_w, new_quant_b, self.stride, self.padding,
                        self.dilation, self.groups)

            return mult_res / (scale_w.view(1,-1,1,1).expand_as(mult_res)) / scale_x
            # return self.dequantfunc(mult_res, scale_x, scale_w.view(1,-1,1,1) )

