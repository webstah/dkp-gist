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

        grad_at_output[:grad_output.shape[0], :].data.copy_(grad_output.data)

        return grad_output, None


class TestTrainingHook(nn.Module):
    #Used for handling the gradients for the DFA plus single layer backprop method
    def __init__(self):
        super(TestTrainingHook, self).__init__()

    def forward(self, input):
        return TestHookFunction.apply(input)

class TestHookFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        print(torch.mean(grad_output))
        return grad_output, None


class DFATrainingHook(nn.Module):
    #This training hook calculates and injects the gradients made by DFA
    def __init__(self, train_mode, dim):
        super(DFATrainingHook, self).__init__()
        self.train_mode = train_mode
        self.backward_weights = None
        if self.train_mode in ['DKP', 'DFA']:
            self.backward_weights = nn.Parameter(torch.Tensor(torch.Size(dim)), requires_grad=True)
        if self.train_mode in ['BP', 'DFA']:
            self.backward_weights.requires_grad = False

    def forward(self, input, grad_at_output):
        return DFAHookFunction.apply(input, self.backward_weights, grad_at_output, self.train_mode)

class DFAHookFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, backward_weights, grad_at_output, train_mode):
        ctx.save_for_backward(input, backward_weights)
        ctx.in1 = grad_at_output
        ctx.in2 = train_mode
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_at_output            = ctx.in1
        train_mode                = ctx.in2
        input, backward_weights   = ctx.saved_variables

        grad_at_output = grad_at_output[:grad_output.shape[0], :]

        if train_mode == 'DFA':
            B_view = backward_weights.view(-1, prod(backward_weights.shape[1:]))
            grad_output_est = grad_at_output.mm(B_view).view(grad_output.shape)
            return grad_output_est, None, None, None

        elif train_mode == 'DKP':
            layer_out_view = input.view(-1, prod(input.shape[1:]))
            B_view = backward_weights.view(-1, prod(backward_weights.shape[1:]))

            grad_output_est = grad_at_output.mm(B_view).view(grad_output.shape)
            grad_weights_B = grad_at_output.t().mm(layer_out_view).view(backward_weights.shape)

            return grad_output_est, grad_weights_B, None, None

        return grad_output, None, None, None
