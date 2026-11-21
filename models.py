"""经典模型保留原结构，仅替换末层；MyNet 使用紧凑卷积与全局池化。"""
from torch import nn
from torchvision import models


def activation(name):
    factories = {'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU}
    if name not in factories:
        raise ValueError(f'未知激活函数: {name}')
    return factories[name]()


class MyNet(nn.Module):
    def __init__(self, num_classes=10, channels=(32, 64, 128, 192),
                 blocks_per_stage=2, act='relu', dropout=0.25, head_channels=None):
        super().__init__()
        layers = []
        incoming = 3
        depths = ([blocks_per_stage] * len(channels) if isinstance(blocks_per_stage, int) else blocks_per_stage)
        for width, depth in zip(channels, depths):
            for block in range(depth):
                # 每阶段首层降采样，控制激活显存；后续卷积提取局部特征
                layers += [nn.Conv2d(incoming, width, 3, stride=2 if block == 0 else 1,
                                     padding=1, bias=False),
                           nn.BatchNorm2d(width), activation(act)]
                incoming = width
        self.features = nn.Sequential(*layers)
        # 1×1 卷积压缩通道；全局池化避免大规模全连接层
        head_width = head_channels if head_channels is not None else channels[-1] // 2
        self.head = nn.Sequential(nn.Conv2d(incoming, head_width, 1, bias=False),
                                  nn.BatchNorm2d(head_width), activation(act),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(head_width, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def create_model(config):
    config.validate()
    if config.model == 'resnet18':
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if config.pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, config.num_classes)
    elif config.model == 'vgg16':
        net = models.vgg16(weights=models.VGG16_Weights.DEFAULT if config.pretrained else None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, config.num_classes)
    else:
        net = MyNet(config.num_classes, config.channels, config.blocks_per_stage,
                    config.activation, config.dropout, config.head_channels)
    return net


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
