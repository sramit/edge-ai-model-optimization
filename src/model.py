import torch.nn as nn
from torchvision import models


NUM_CLASSES = 10


def build_model():
    model = models.mobilenet_v3_small(weights=None)

    # Adapt MobileNetV3 for CIFAR-10's 32x32 input.
    model.features[0][0] = nn.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # Remove ImageNet-style early downsampling.
    model.features[0][1] = nn.BatchNorm2d(16)

    # CIFAR-10 classifier.
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        NUM_CLASSES,
    )

    return model
