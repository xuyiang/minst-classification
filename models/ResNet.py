# coding:utf8
from torch import nn
from torch.nn import functional as F

from .BasicModel import BasicModule


class ResidualBlock(nn.Module):
    """
    实现子module: Residual Block
    """

    def __init__(self, inchannel=1, outchannel=10, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(BasicModule):
    """
    实现主module：ResNet34
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    """

    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        self.model_name = 'resnet34'

        # 前几层: 图像转换 - 修改卷积核大小和步长以适应28x28的输入
        self.pre = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),  # 改为3x3卷积核，步长1，通道数减少
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        # 重复的layer，分别有3，4，6，3个residual block，但通道数减少
        self.layer1 = self._make_layer(32, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建layer,包含多个residual block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)  # 28x28 -> 28x28

        x = self.layer1(x)  # 28x28 -> 28x28
        x = self.layer2(x)  # 28x28 -> 14x14
        x = self.layer3(x)  # 14x14 -> 7x7
        x = self.layer4(x)  # 7x7 -> 4x4

        x = F.adaptive_avg_pool2d(x, 1)  # 4x4 -> 1x1
        x = x.view(x.size(0), -1)
        return self.fc(x)
