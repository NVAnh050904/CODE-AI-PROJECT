"""
Các kiến trúc mạng trích xuất đặc trưng (Backbone Feature Extractor) cho bài toán Nhận diện thuộc tính người đi bộ (PAR) tại D:\AI DATASET.
Hỗ trợ ResNet50 (sử dụng Bottleneck được huấn luyện trước trên ImageNet và kiến trúc tự Custom Scratch).
"""
import torch
import torch.nn as nn
import torchvision.models as models


class IdentityBlock(nn.Module):
    def __init__(self, in_channels: int, filters: list):
        super(IdentityBlock, self).__init__()
        f1, f2, f3 = filters
        self.conv2a = nn.Conv2d(in_channels, f1, kernel_size=1, bias=False)
        self.bn2a = nn.BatchNorm2d(f1)
        self.conv2b = nn.Conv2d(f1, f2, kernel_size=3, padding=1, bias=False)
        self.bn2b = nn.BatchNorm2d(f2)
        self.conv2c = nn.Conv2d(f2, f3, kernel_size=1, bias=False)
        self.bn2c = nn.BatchNorm2d(f3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn2a(self.conv2a(x)))
        out = self.relu(self.bn2b(self.conv2b(out)))
        out = self.bn2c(self.conv2c(out))
        out += residual
        return self.relu(out)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, filters: list, stride: int = 2):
        super(ConvBlock, self).__init__()
        f1, f2, f3 = filters
        self.conv2a = nn.Conv2d(in_channels, f1, kernel_size=1, stride=stride, bias=False)
        self.bn2a = nn.BatchNorm2d(f1)
        self.conv2b = nn.Conv2d(f1, f2, kernel_size=3, padding=1, bias=False)
        self.bn2b = nn.BatchNorm2d(f2)
        self.conv2c = nn.Conv2d(f2, f3, kernel_size=1, bias=False)
        self.bn2c = nn.BatchNorm2d(f3)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, f3, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(f3)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn2a(self.conv2a(x)))
        out = self.relu(self.bn2b(self.conv2b(out)))
        out = self.bn2c(self.conv2c(out))
        out += residual
        return self.relu(out)


class CustomResNet50Backbone(nn.Module):
    def __init__(self):
        super(CustomResNet50Backbone, self).__init__()
        self.stage1 = nn.Sequential(
            nn.ZeroPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, [64, 64, 256], stride=1),
            IdentityBlock(256, [64, 64, 256]),
            IdentityBlock(256, [64, 64, 256])
        )
        self.stage3 = nn.Sequential(
            ConvBlock(256, [128, 128, 512], stride=2),
            IdentityBlock(512, [128, 128, 512]),
            IdentityBlock(512, [128, 128, 512]),
            IdentityBlock(512, [128, 128, 512])
        )
        self.stage4 = nn.Sequential(
            ConvBlock(512, [256, 256, 1024], stride=2),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024])
        )
        self.stage5 = nn.Sequential(
            ConvBlock(1024, [512, 512, 2048], stride=2),
            IdentityBlock(2048, [512, 512, 2048]),
            IdentityBlock(2048, [512, 512, 2048])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x


def build_backbone(name: str = 'resnet50', pretrained: bool = True):
    name = name.lower()
    if name == 'custom_resnet50':
        backbone = CustomResNet50Backbone()
        feature_dim = 2048
    else:
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            base = models.resnet50(weights=weights)
            backbone = nn.Sequential(*list(base.children())[:-2])
        else:
            backbone = CustomResNet50Backbone()
        feature_dim = 2048

    return backbone, feature_dim
