import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DFATrainingHook, OutputTrainingHook

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, train_mode, batch_size, num_classes=100):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.train_mode = train_mode
        self.batch_size = batch_size

        self.grad_at_output = nn.Parameter(torch.zeros([batch_size, num_classes]), requires_grad=False)

        self.features = conv_features(train_mode)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = linear_classifier(train_mode, num_classes)

        self.output_hook = OutputTrainingHook()

    def forward(self, x, target):
        x = self.features(x, self.grad_at_output)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x, self.grad_at_output)

        if x.requires_grad:
            x = self.output_hook(x, self.grad_at_output)

        return x

class conv_features(nn.Module):
    def __init__(self, train_mode):
        super(conv_features, self).__init__()
        self.train_mode = train_mode

        self.conv_1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn_1 = nn.BatchNorm2d(64)
        self.conv_1_dfa = DFATrainingHook(train_mode, [100, 64, 55, 55])

        self.conv_2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.pool_2  = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn_2 = nn.BatchNorm2d(192)
        self.conv_2_dfa = DFATrainingHook(train_mode, [100, 192, 27, 27])

        self.conv_3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn_3 = nn.BatchNorm2d(384)
        self.conv_3_dfa = DFATrainingHook(train_mode, [100, 384, 13, 13])

        self.conv_4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn_4 = nn.BatchNorm2d(256)
        self.conv_4_dfa = DFATrainingHook(train_mode, [100, 256, 13, 13])

        self.conv_5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_5 = nn.BatchNorm2d(256)
        self.pool_5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv_5_dfa = DFATrainingHook(train_mode, [100, 256, 13, 13])

        self.act = nn.ReLU(inplace=True)

    def forward(self, x, grad_at_output):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act(x)
        x = self.conv_1_dfa(x, grad_at_output)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act(x)
        x = self.conv_2_dfa(x, grad_at_output)
        x = self.pool_2(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.act(x)
        x = self.conv_3_dfa(x, grad_at_output)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.act(x)
        x = self.conv_4_dfa(x, grad_at_output)

        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.act(x)
        x = self.conv_5_dfa(x, grad_at_output)
        x = self.pool_5(x)

        return x


class linear_classifier(nn.Module):
    def __init__(self, train_mode, num_classes):
        super(linear_classifier, self).__init__()
        self.drop = nn.Dropout()
        self.act = nn.ReLU(inplace=True)

        self.linear_1 = nn.Linear(256 * 6 * 6, 4096)
        self.linear_1_dfa = DFATrainingHook(train_mode, [100, 4096])

        self.linear_2 = nn.Linear(4096, 4096)
        self.linear_2_dfa = DFATrainingHook(train_mode, [100, 4096])

        self.linear_3 = nn.Linear(4096, num_classes)

    def forward(self, x, grad_at_output):
        x = self.drop(x)

        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_1_dfa(x, grad_at_output)

        x = self.drop(x)

        x = self.linear_2(x)
        x = self.act(x)
        x = self.linear_2_dfa(x, grad_at_output)

        x = self.linear_3(x)

        return x

def alexnet(train_mode, batch_size, pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(train_mode, batch_size, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
