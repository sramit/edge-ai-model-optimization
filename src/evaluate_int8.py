import numpy as np
import torch
import onnxruntime as ort

from torchvision import datasets, transforms


FP32_PATH = "./results/mobilenetv3_cifar10_fp32.onnx"
INT8_PATH = "./results/mobilenetv3_cifar10_int8.onnx"

DATA_DIR = "./data"

BATCH_SIZE = 128


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])


def evaluate(model_path, name):

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    correct = 0
    total = 0

    for start in range(0, len(dataset), BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            len(dataset),
        )

        images = torch.stack(
            [dataset[i][0] for i in range(start, end)]
        )

        labels = np.array(
            [dataset[i][1] for i in range(start, end)]
        )

        outputs = session.run(
            None,
            {
                input_name:
                images.numpy().astype(np.float32)
            },
        )[0]

        predictions = outputs.argmax(axis=1)

        correct += (
            predictions == labels
        ).sum()

        total += len(labels)

    accuracy = 100.0 * correct / total

    print(
        f"{name} accuracy: "
        f"{accuracy:.2f}%"
    )

    return accuracy


if __name__ == "__main__":

    print("INT8 Accuracy Evaluation")
    print("=" * 40)

    fp32_accuracy = evaluate(
        FP32_PATH,
        "FP32",
    )

    int8_accuracy = evaluate(
        INT8_PATH,
        "INT8",
    )

    print("\nQuantization Impact")
    print("-" * 40)

    print(
        f"FP32 accuracy : "
        f"{fp32_accuracy:.2f}%"
    )

    print(
        f"INT8 accuracy : "
        f"{int8_accuracy:.2f}%"
    )

    print(
        f"Accuracy drop : "
        f"{fp32_accuracy - int8_accuracy:.2f} "
        f"percentage points"
    )
