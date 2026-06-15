from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .data import IMG_SIZE, NUM_CLASSES


class MLPNet(nn.Module):
    def __init__(self, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMG_SIZE * IMG_SIZE, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class TinyCNN(nn.Module):
    def __init__(self, width: int = 32, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 2 * 12 * 12, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _vgg_block(in_ch: int, out_ch: int, use_bn: bool) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers += [nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
    return nn.Sequential(*layers)


class DeepCNN(nn.Module):
    def __init__(self, n_blocks: int = 3, base_width: int = 64,
                 use_bn: bool = True, dropout: float = 0.4):
        super().__init__()
        chans = [1] + [base_width * (2 ** i) for i in range(n_blocks)]
        self.features = nn.Sequential(
            *[_vgg_block(chans[i], chans[i + 1], use_bn) for i in range(n_blocks)]
        )
        feat_size = IMG_SIZE // (2 ** n_blocks)
        flat = chans[-1] * feat_size * feat_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_resnet(pretrained: bool = False, dropout: float = 0.0) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    in_feat = net.fc.in_features
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, NUM_CLASSES))
    return net


def build_model(name: str, **cfg) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLPNet(hidden=cfg.get("hidden", 512), dropout=cfg.get("dropout", 0.0))
    if name == "tinycnn":
        return TinyCNN(width=cfg.get("width", 32), dropout=cfg.get("dropout", 0.0))
    if name == "deepcnn":
        return DeepCNN(
            n_blocks=cfg.get("n_blocks", 3), base_width=cfg.get("base_width", 64),
            use_bn=cfg.get("use_bn", True), dropout=cfg.get("dropout", 0.4),
        )
    if name == "resnet":
        return make_resnet(pretrained=cfg.get("pretrained", False),
                           dropout=cfg.get("dropout", 0.0))
    raise ValueError(f"unknown model: {name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
