#-------------------------------------------------------------------------------
# Author:     Namratha Sanjay
# Name:       linear quantization helper functions
# Purpose:    linear quantization helper functions
# Copyright:   (c) Volvo cars 
# History:
#-------------------------------------------------------------------------------

import math
import numpy as np
from torch.autograd import Function, Variable
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def clamp(input, min, max, inplace=False):
    """
    Clamp tensor input to (min, max).
    input: input tensor to be clamped
    """

    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping single-precision input to integer values with the given scale and zeropoint
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_dequantize(input, scale, zero_point, inplace=False):
    """
    Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
    input: integer input tensor to be mapped
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale

def symmetric_linear_quantization_params(num_bits,
                                        saturation_min,
                                        saturation_max,
                                        per_channel=False):
    """ Compute the scaling factor and zero-point with given quantization range """
    '''max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2.**(num_bits-1) - 1.

    scale = max_val / qmax

    return scale, 0'''
    
    n = 2 ** (num_bits - 1) - 1
    if per_channel:
        scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
        scale = torch.clamp(scale, min=1e-8) / n
    else:
        scale = max(abs(saturation_min), abs(saturation_min))
        scale = torch.clamp(scale, min=1e-8) / n

    return scale, 0

def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True,
                                          signed=True):
    """
    Compute the scaling factor and zeropoint with the given quantization range.
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    n = 2**num_bits - 1
    scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2**(num_bits - 1)
    return scale, zero_point


class AsymmetricQuantFunction_Int(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """

        if x_min is None or x_max is None or (sum(x_min == x_max) == 1
                                              and x_min.numel() == 1):
            x_min, x_max = x.min(), x.max()
        scale, zero_point = asymmetric_linear_quantization_params(
            k, x_min, x_max)
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2**(k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize(new_quant_x,
                                    scale,
                                    zero_point,
                                    inplace=False)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        ''' straight through estimator '''
        return grad_output.clone(), None, None, None

class AsymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """

        if x_min is None or x_max is None or (sum(x_min == x_max) == 1
                                              and x_min.numel() == 1):
            x_min, x_max = x.min(), x.max()
        scale, zero_point = asymmetric_linear_quantization_params(
            k, x_min, x_max)
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2**(k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize(new_quant_x,
                                    scale,
                                    zero_point,
                                    inplace=False)
        visualise(x, axs)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError



"""---------- Integer only quantization with back-propogation ---------"""
def quantize_int(x, k, x_min=None, x_max=None):
    if x_min is None or x_max is None or (sum(x_min == x_max) == 1 and x_min.numel() == 1):
        x_min, x_max = x.min(), x.max()
    scale, zero_point = asymmetric_linear_quantization_params(k, x_min, x_max)
    quantfunc = LinearQuantizeModule()
    #new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
    new_quant_x = quantfunc(x, scale, zero_point, inplace=False)
    n = 2**(k - 1)
    # new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
    if len(scale.shape) == 0:
        return new_quant_x, torch.Tensor([scale]), torch.Tensor([zero_point])
    return new_quant_x, scale, zero_point


class LinearDequantizeModule(nn.Module):
    def __init__(self):
        super(LinearDequantizeModule, self).__init__()

    def forward(self, x, scale_x, scale_w):
        print('lineardequant_forward')
        self.M = 1 / (scale_x * scale_w)
        M_0 = torch.round(self.M * 2**31)
        res = ((x * M_0) << 31)
        #print(f"M:{self.M}, M_0:{M_0}")
        #print(res.sum())
        #print((x*self.M).sum())
        #exit()
        return res

    def backward(self, grad_output):
        #print('backpropogate scale',self.M * grad_output.clone())
        return self.M * grad_output.clone(), None, None, None


class LinearQuantizeModule(nn.Module):
    def __init__(self):
        super(LinearQuantizeModule, self).__init__()

    def forward(self, input, scale, zero_point, inplace=False):
        """
        Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
        input: single-precision input tensor to be quantized
        scale: scaling factor for quantization
        zero_point: shift for quantization
        """

        # reshape scale and zeropoint for convolutional weights and activation
        self.scale      = scale
        self.zero_point = zero_point
        if len(input.shape) == 4:
            scale      = scale.view(-1, 1, 1, 1)
            zero_point = zero_point.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(input.shape) == 2:
            scale      = scale.view(-1, 1)
            zero_point = zero_point.view(-1, 1)
        # mapping single-precision input to integer values with the given scale and zeropoint
        if inplace:
            input.mul_(scale).sub_(zero_point).round_()
            return input
        return torch.round(scale * input - zero_point)

    def backward(self, grad_output):
        return self.scale * grad_output.clone(), None, None, None




def plot_quant_float(input_x, quant_x):
    input_x = input_x.cpu().detach().numpy()
    quant_x = quant_x.cpu().detach().numpy()
    #fig = plt.figure()
    fig, ax = plt.subplots()
    ax.plot(input_x, quant_x, marker='+', color='b')
    ax.set_xlabel('floating point')
    ax.set_ylabel('quantized integer')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.grid(True)
    fig.savefig('/home/jovyan/new-quant-static-vol-3/plots/plot_linear.png')
    #plt.show()

def plot_quant_float_conv(input_x, quant_x):
  print('plot_quant_float_conv')
  input_x = input_x.cpu().detach().numpy()
  quant_x = quant_x.cpu().detach().numpy()
  in_channel,out_channel, h, w = input_x.shape
  in_, out_, h_, w_ = quant_x.shape
  fig = plt.figure()
  x = input_x.reshape(out_channel*h, in_channel*w)
  quant_x_ = quant_x.reshape(out_*h_, in_* w_)
  #print(x.shape)
  fig, ax = plt.subplots()
  ax.plot(x, quant_x_, marker='+', color='b')
  ax.set_xlabel('floating point')
  ax.set_ylabel('quantized integer')
  ax.axhline(y=0, color='k')
  ax.axvline(x=0, color='k')
  ax.grid(True)
  fig.savefig('/home/jovyan/new-quant-static-vol-3/plots/plot_conv_{}.png'.format(input_x.shape))
    
    

def plot_quantization_loss_linear(input_x, dequant_x):
    input_x = input_x.cpu().detach().numpy()
    #print('input', input_x.shape)
    dequant_x = dequant_x.cpu().detach().numpy()
    #print('dequant',dequant_x.shape)
    error = np.reshape(input_x - dequant_x, (-1) )
    #print('error',error.shape)
    #print("error", np.min(error), np.max(error))
    fig = plt.figure()
    fig, ax = plt.subplots()
    ax.scatter(input_x, error)
    ax.set_ylim([1.2*min(error), 1.2*max(error)])
    ax.set_xlabel("real value")
    ax.set_ylabel("quantization error")
    fig.savefig('/home/jovyan/new-quant-static-vol-3/plots/quant_error.png')


def plot_quantization_loss_conv(input_x, dequant_x):
    input_x = input_x.cpu().detach().numpy()
    #print('input', input_x.shape)
    dequant_x = dequant_x.cpu().detach().numpy()
    #print('dequant',dequant_x.shape)
    error = np.reshape(input_x - dequant_x, (-1) )
    #print('error',error.shape)
    #print("error", np.min(error), np.max(error))
    fig = plt.figure()
    fig, ax = plt.subplots()
    ax.scatter(input_x, error)
    ax.set_ylim([1.2*min(error), 1.2*max(error)])
    ax.set_xlabel("real value")
    ax.set_ylabel("quantization error")
    fig.savefig('/home/jovyan/new-quant-static-vol-3/plots/quant_error_conv_{}.png'.format(input_x.shape))


def visualise(x, axs):
  x = x.view(-1).cpu().numpy()
  axs.hist(x) 

