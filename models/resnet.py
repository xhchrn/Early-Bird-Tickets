'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .channel_selection import channel_selection


__all__ = ['resnet_cifar']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, cfg=None, downsample=None):
        assert cfg is not None
        super(BasicBlock, self).__init__()

        self.select = channel_selection(in_planes)

        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.shortcut = nn.Sequential()
        self.downsample = downsample
        self.stride = stride
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )

    def forward(self, x):
        out = self.select(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        residual = self.downsample(x) if self.downsample else x

        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_method='standard', cfg=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if cfg is None:
            # Construct config variable.
            cfg = [
                [64],
                [64, 64] * num_blocks[0],
                [128, 128] * num_blocks[1],
                [256, 256] * num_blocks[2],
                [512, 512] * num_blocks[3],
            ]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.select = channel_selection(64 * block.expansion)

        self.layer1 = self._make_layer(block, 64 , num_blocks[0], stride=1, cfg=cfg[0                    :2*sum(num_blocks[:1])])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, cfg=cfg[2*sum(num_blocks[:1]):2*sum(num_blocks[:2])])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, cfg=cfg[2*sum(num_blocks[:2]):2*sum(num_blocks[:3])])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, cfg=cfg[2*sum(num_blocks[:3]):2*sum(num_blocks[:4])])

        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.reset_conv_parameters(init_method)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def reset_parameters(self, module, init_method="kaiming_uniform") -> None:
        if init_method == "kaiming_constant_signed":
            fan = nn.init._calculate_correct_fan(module.weight, "fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                module.weight.data = module.weight.data.sign() * std
        elif init_method == "kaiming_constant_unsigned":
            fan = nn.init._calculate_correct_fan(module.weight, "fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                module.weight.data = torch.ones_like(module.weight.data) * std
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
        elif init_method == "kaiming_laplace":
            fan = nn.init._calculate_correct_fan(module.weight, "fan_in")
            gain = nn.init.calculate_gain("relu")
            scale = gain / math.sqrt(2.0 * fan)
            with torch.no_grad():
                new_weight = np.random.laplace(loc=0.0, scale=scale, size=module.weight.shape)
                module.weight.data = module.weight.data.new_tensor(torch.from_numpy(new_weight).clone().detach())
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif init_method == "xavier_constant":
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            with torch.no_grad():
                module.weight.data = module.weight.data.sign() * std
        elif init_method == "standard":
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        else:
            raise ValueError(f"{init_method} is not an initialization option!")

    def reset_conv_parameters(self, init_method="standard") -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.reset_parameters(m, init_method)
    
    def _make_layer(self, block, planes, num_blocks, stride, cfg):
        downsample = None
        if stride != 1 or planes != block.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_plains, block.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(block.expansion * planes)
            )

        strides = [stride] + [1]*(num_blocks-1)

        layers = []
        for i, stride in enumerate(strides):
            ds = downsample if i == 0 else None
            layers.append(block(self.in_planes, planes, stride, cfg[2*i:2*(i+1)], ds))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.select(self.bn1(self.conv1(x))))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet_cifar(dataset='cifar10', depth=18, cfg=None):
    assert dataset == 'cifar10'
    if depth == 18:
        return ResNet18(init_method='kaiming_normal', cfg=cfg)
    elif depth == 34:
        return ResNet34(init_method='kaiming_normal', cfg=cfg)
    elif depth == 50:
        return ResNet50(init_method='kaiming_normal', cfg=cfg)
    elif depth == 101:
        return ResNet101(init_method='kaiming_normal', cfg=cfg)
    elif depth == 152:
        return ResNet152(init_method='kaiming_normal', cfg=cfg)
    else:
        raise ValueError('invalid arch type')


def ResNet18(init_method='standard', cfg=None):
    return ResNet(BasicBlock, [2,2,2,2], init_method=init_method, cfg=cfg)

def ResNet34(init_method='standard', cfg=None):
    return ResNet(BasicBlock, [3,4,6,3], init_method=init_method, cfg=cfg)

def ResNet50(init_method='standard', cfg=None):
    return ResNet(Bottleneck, [3,4,6,3], init_method=init_method, cfg=cfg)

def ResNet101(init_method='standard', cfg=None):
    return ResNet(Bottleneck, [3,4,23,3], init_method=init_method, cfg=cfg)

def ResNet152(init_method='standard', cfg=None):
    return ResNet(Bottleneck, [3,8,36,3], init_method=init_method, cfg=cfg)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()

