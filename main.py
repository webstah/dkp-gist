from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from networks.conv import ConvNetwork
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import math
import numpy as np
from numpy import prod

from torch.utils.tensorboard import SummaryWriter

from networks.conv import ConvNetwork
from utils.multioptimizer import MultipleOptimizer
from utils.logger import CSVLogger


def train(args, model, device, train_loader, optimizer, epoch, batch_size, writer):
    model.train()

    training_loss = 0
    alignment_fc1 = 0
    alignment_conv1 = 0
    alignment_conv2 = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        one_hot = torch.nn.functional.one_hot(target, 10).float()
        output = model(data, one_hot)
        loss = F.cross_entropy(output, target)
        loss.backward()

        align_fc1 = torch.mean(F.cosine_similarity(output.detach(), model.fc1_out[:output.shape[0], :].mm(model.fc1_dfa.backward_weights.t())))
        align_conv1 = torch.mean(F.cosine_similarity(output.detach(), model.conv1_out[:output.shape[0], :].view(-1, prod(model.conv1_out[:output.shape[0], :].shape[1:])).mm(model.conv1_dfa.backward_weights.view(-1, prod(model.conv1_dfa.backward_weights.shape[1:])).t())))
        align_conv2 = torch.mean(F.cosine_similarity(output.detach(), model.conv2_out[:output.shape[0], :].view(-1, prod(model.conv2_out[:output.shape[0], :].shape[1:])).mm(model.conv2_dfa.backward_weights.view(-1, prod(model.conv2_dfa.backward_weights.shape[1:])).t())))
        alignment_fc1 += align_fc1.item()
        alignment_conv1 += align_conv1.item()
        alignment_conv2 += align_conv2.item()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            if args.dry_run:
                break

        training_loss += loss

    training_loss /= (batch_idx + 1)

    writer.add_scalar('loss/training_loss', training_loss.item(), epoch)
    writer.add_scalar('cos_alignment/fc1_to_output', alignment_fc1 / (batch_idx + 1), epoch)
    writer.add_scalar('cos_alignment/conv1_to_output', alignment_conv1 / (batch_idx + 1), epoch)
    writer.add_scalar('cos_alignment/conv2_to_output', alignment_conv2 / (batch_idx + 1), epoch)

    writer.close()

    return training_loss.item()



def test(model, device, test_loader, train_loader, epoch, batch_size, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data, None)
            one_hot = torch.nn.functional.one_hot(target, 10).float()
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= batch_idx + 1
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.2f}%)\n'.format(
        test_loss, test_accuracy))

    if epoch is not None:
        writer.add_scalar('loss/test_loss', test_loss, epoch)
        writer.add_scalar('accuracy/test_accuracy', test_accuracy, epoch)
        writer.close()

    return test_loss, test_accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Pytorch Implementation of DFA and Learned Direct Connections')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--b-lr', type=float, default=1e-4, metavar='BLR',
                        help='learning rate for backward parameters (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='M',
                        help='Learning rate step gamma (default: 0.8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--train-mode', choices = ['BP', 'DKP', 'DFA'], default='DKP',
                        help='Choose between backpropagation (BP), Direct Kolen Pollack (DKP), or Direct Feedback Alignment (DFA).')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True},
                     )


    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.CIFAR10(root='/data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **kwargs)

    test_data = datasets.CIFAR10(root='/data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **kwargs)

    model = ConvNetwork(args.batch_size, args.train_mode, device).to(device)

    writer = SummaryWriter(log_dir='results/conv_dkp_4')
    logger = CSVLogger(['Epoch', 'Training Loss', 'Test Loss', 'Test Accuracy'], args)

    if args.train_mode == 'DKP':
        # we need to run through the network once to properly initialize the backward weights
        test(model, device, test_loader, train_loader, None, None, None)

        forward_params = []
        backward_params = []
        for name, param in model.named_parameters():
            print(name, type(param.data), param.size(), param.is_leaf, param.requires_grad)
            if "backward" in name:
                backward_params.append(param)
            else:
                forward_params.append(param)

        forward_optimizer = optim.SGD([{'params': forward_params}], lr=args.lr, weight_decay=1e-6, momentum=0.9, nesterov=True)
        backward_optimizer = optim.Adam([{'params': backward_params}], lr=args.b_lr, weight_decay=1e-6)

        optimizer = MultipleOptimizer(forward_optimizer, backward_optimizer)
        scheduler = StepLR(backward_optimizer, step_size=1, gamma=args.gamma)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-6, momentum=0.9, nesterov=True)

    for epoch in range(1, args.epochs + 1):
        training_loss = train(args, model, device, train_loader, optimizer, epoch, args.batch_size, writer)
        test_loss, test_accuracy = test(model, device, test_loader, train_loader, epoch, args.batch_size, writer)

        logger.save_values(epoch, training_loss, test_loss, test_accuracy)

        if args.train_mode == 'DKP':
            scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_conv.pt")


if __name__ == '__main__':
    main()

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def state_dict(self):
        for op in self.optimizers:
            op.state_dict()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        one_hot = torch.nn.functional.one_hot(target, 10).float()
        output = model(data, one_hot)
        loss = F.cross_entropy(output, target)
        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, train_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data, None)
            one_hot = torch.nn.functional.one_hot(target, 10).float()
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.2f}%)\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Pytorch Implementation of DFA and Learned Direct Connections')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default:50)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--b-lr', type=float, default=1e-4, metavar='BLR',
                        help='learning rate for backward parameters (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='M',
                        help='Learning rate step gamma (default: 0.8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--train-mode', choices = ['BP', 'DKP', 'DFA'], default='DKP',
                        help='Choose between backpropagation (BP), Direct Kolen Pollack (DKP), or Direct Feedback Alignment (DFA).')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True},
                     )


    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **kwargs)

    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **kwargs)

    model = ConvNetwork(args.batch_size, args.train_mode, device).to(device)
    if args.train_mode == 'DKP':
        # we need to run through the network once to properly initialize the backward weights
        test(model, device, test_loader, train_loader)

        forward_params = []
        backward_params = []
        for name, param in model.named_parameters():
            print(name, type(param.data), param.size(), param.is_leaf, param.requires_grad)
            if "backward" in name:
                backward_params.append(param)
            else:
                forward_params.append(param)

        forward_optimizer = optim.SGD([{'params': forward_params}], lr=args.lr, weight_decay=1e-6, momentum=0.9, nesterov=True)
        backward_optimizer = optim.Adam([{'params': backward_params}], lr=args.b_lr, weight_decay=1e-6)

        optimizer = MultipleOptimizer(forward_optimizer, backward_optimizer)
        scheduler = StepLR(backward_optimizer, step_size=1, gamma=args.gamma)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-6, momentum=0.9, nesterov=True)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, train_loader)
        if args.train_mode == 'DKP':
            scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_cnn.pt")


if __name__ == '__main__':
    main()
