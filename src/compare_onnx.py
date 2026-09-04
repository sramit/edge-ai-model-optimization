import os

import numpy as np
import torch
import onnxruntime as ort
from torchvision import datasets, transforms

from model import build_model


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "mobilenetv3_cifar10_fp32.pth",
)

ONNX_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "mobilenetv3_cifar10_fp32.onnx",
)


def main():
    # Load PyTorch model
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load the same CIFAR-10 test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    # Use first 100 samples
    images = torch.stack(
        [dataset[i][0] for i in range(100)]
    )

    # PyTorch inference
    with torch.inference_mode():
        torch_output = model(images)

    torch_output = torch_output.numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(
        ONNX_PATH,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    onnx_output = session.run(
        None,
        {
            input_name: images.numpy().astype(np.float32)
        },
    )[0]

    # Compare raw outputs
    absolute_difference = np.abs(
        torch_output - onnx_output
    )

    max_difference = absolute_difference.max()
    mean_difference = absolute_difference.mean()

    # Compare predictions
    torch_predictions = torch_output.argmax(axis=1)
    onnx_predictions = onnx_output.argmax(axis=1)

    matching_predictions = (
        torch_predictions == onnx_predictions
    ).mean() * 100

    print("\nPyTorch vs ONNX Runtime")
    print("=" * 45)

    print(f"Samples tested       : {len(images)}")
    print(f"Max output difference: {max_difference:.8f}")
    print(f"Mean output difference: {mean_difference:.8f}")
    print(
        f"Prediction agreement : "
        f"{matching_predictions:.2f}%"
    )


if __name__ == "__main__":
    main()
