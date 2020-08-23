import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable

import math
import numpy as np
from numpy import prod


class OutputTrainingHook(nn.Module):
    #This training hook captures and handles the gradients at the output of the network
    def __init__(self):
        super(OutputTrainingHook, self).__init__()

    def forward(self, input, grad_at_output):
        return OutputHookFunction.apply(input, grad_at_output)

class OutputHookFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_at_output):
        ctx.save_for_backward(input)
        ctx.in1 = grad_at_output
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_at_output = ctx.in1
        input = ctx.saved_variables

        grad_at_output.data.copy_(grad_output.data)

        return grad_output, None


class DFATrainingHook(nn.Module):
    #This training hook calculates and injects the gradients made by DKP or DFA
    def __init__(self, train_mode):
        super(DFATrainingHook, self).__init__()
        self.train_mode = train_mode
        self.is_not_initialized = True
        self.backward_weights = nn.Parameter(requires_grad=True)

    def init_weights(self, dim):
        self.backward_weights = nn.Parameter(torch.Tensor(torch.Size(dim)).to(torch.cuda.current_device()))
        torch.nn.init.kaiming_uniform_(self.backward_weights)
        if self.train_mode == 'DKP':
            self.backward_weights.requires_grad = True
        elif self.train_mode == 'DFA':
            self.backward_weights.requires_grad = False

    def forward(self, input, grad_at_output, network_output):
        if self.is_not_initialized and self.train_mode in ['DKP', 'DFA']:
            if len(input.shape) > 2:
                dim = [grad_at_output.shape[1], input.shape[1], input.shape[2], input.shape[3]]
            else:
                dim = [grad_at_output.shape[1], input.shape[1]]
            self.init_weights(dim)
            self.is_not_initialized = False

        return DFAHookFunction.apply(input, self.backward_weights, grad_at_output, network_output, self.train_mode)

class DFAHookFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, backward_weights, grad_at_output, network_output, train_mode):
        ctx.save_for_backward(input, backward_weights)
        ctx.in1 = grad_at_output
        ctx.in2 = network_output
        ctx.in3 = train_mode
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_at_output            = ctx.in1
        network_output            = ctx.in2
        train_mode                = ctx.in3
        input, backward_weights   = ctx.saved_variables

        if train_mode == 'DFA':
            B_view = backward_weights.view(-1, prod(backward_weights.shape[1:]))
            grad_output_est = grad_at_output.mm(B_view).view(grad_output.shape)
            return grad_output_est, None, None, None, None

        elif train_mode == 'DKP':
            layer_out_view = input.view(-1, prod(input.shape[1:]))
            B_view = backward_weights.view(-1, prod(backward_weights.shape[1:]))
            output_view = network_output.view(-1, prod(network_output.shape[1:]))

            grad_output_est = grad_at_output.mm(B_view).view(grad_output.shape)
            grad_weights_B = grad_at_output.t().mm(layer_out_view).view(backward_weights.shape)

            return grad_output_est, grad_weights_B, None, None, None

        return grad_output, None, None, None, None

