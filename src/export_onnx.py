import os

import torch
import torch.nn as nn
from torchvision import models


CHECKPOINT_PATH = "./results/mobilenetv3_cifar10_fp32.pth"
OUTPUT_PATH = "./results/mobilenetv3_cifar10_fp32.onnx"


def build_model():
    model = models.mobilenet_v3_small(weights=None)

    model.features[0][0] = nn.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.features[0][1] = nn.BatchNorm2d(16)

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        10,
    )

    return model


def main():
    device = torch.device("cpu")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
    )

    print(f"Exported ONNX model: {OUTPUT_PATH}")
    print(
        f"Model size: "
        f"{os.path.getsize(OUTPUT_PATH) / (1024 ** 2):.2f} MB"
    )


if __name__ == "__main__":
    main()
