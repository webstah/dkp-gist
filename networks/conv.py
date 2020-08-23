import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DFATrainingHook, OutputTrainingHook, TestTrainingHook


class ConvNetwork(nn.Module):
    def __init__(self, batch_size, train_mode, device):
        super(ConvNetwork, self).__init__()
        self.batch_size = batch_size
        self.train_mode = train_mode

        self.network_output = torch.zeros([batch_size, 10], requires_grad=False).to(device)
        self.grad_at_output = torch.zeros([batch_size, 10], requires_grad=False).to(device)

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv1_dfa = DFATrainingHook(train_mode)

        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv2_dfa = DFATrainingHook(train_mode)

        self.fc1 = nn.Linear(6272, 128)
        self.fc1_dfa = DFATrainingHook(train_mode)

        self.fc2 = nn.Linear(128, 10)

        self.output_hook = OutputTrainingHook()


    def forward(self, x, target):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.conv1_dfa(x, self.grad_at_output, self.network_output)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.conv2_dfa(x, self.grad_at_output, self.network_output)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1_dfa(x, self.grad_at_output, self.network_output)

        x = self.fc2(x)

        if x.requires_grad:
            x = self.output_hook(x, self.grad_at_output)
            self.network_output.data.copy_(x.data)


        return x
