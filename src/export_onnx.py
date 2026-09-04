import os

import torch

from model import build_model


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "mobilenetv3_cifar10_fp32.pth",
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "mobilenetv3_cifar10_fp32.onnx",
)


def get_model_size(path):
    total_size = os.path.getsize(path)

    data_path = path + ".data"

    if os.path.exists(data_path):
        total_size += os.path.getsize(data_path)

    return total_size / (1024 ** 2)


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
    print(f"Total model size: {get_model_size(OUTPUT_PATH):.2f} MB")


if __name__ == "__main__":
    main()
