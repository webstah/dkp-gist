import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.gradfunctions import DFATrainingHook, OutputTrainingHook, TestTrainingHook

class ConvNetwork(nn.Module):
    def __init__(self, batch_size, train_mode, device):
        super(ConvNetwork, self).__init__()
        self.batch_size = batch_size
        self.train_mode = train_mode
        self.device = device

        self.grad_at_output = torch.zeros([batch_size, 10], requires_grad=False).to(device)
        self.network_output = torch.zeros([batch_size, 10], requires_grad=False).to(device)

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv1_dfa = DFATrainingHook(train_mode)
        self.conv1_out = torch.zeros([batch_size, 32, 30, 30], requires_grad=False).to(device)

        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv2_dfa = DFATrainingHook(train_mode)
        self.conv2_out = torch.zeros([batch_size, 32, 28, 28], requires_grad=False).to(device)

        self.fc1 = nn.Linear(6272, 128)
        self.fc1_dfa = DFATrainingHook(train_mode)
        self.fc1_out = torch.zeros([batch_size, 128], requires_grad=False).to(device)

        self.fc2 = nn.Linear(128, 10)

        self.output_hook = OutputTrainingHook()


    def forward(self, x, target):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.conv1_dfa(x, self.grad_at_output, self.network_output)
        if x.requires_grad:
            self.conv1_out[:x.shape[0], :, :, :].data.copy_(x.data)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.conv2_dfa(x, self.grad_at_output, self.network_output)
        if x.requires_grad:
            self.conv2_out[:x.shape[0], :, :, :].data.copy_(x.data)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1_dfa(x, self.grad_at_output, self.network_output)
        if x.requires_grad:
            self.fc1_out[:x.shape[0], :].data.copy_(x.data)

        x = self.fc2(x)

        if x.requires_grad:
            x = self.output_hook(x, self.grad_at_output)
            self.network_output[:x.shape[0], :].data.copy_(x.data)

        return x
