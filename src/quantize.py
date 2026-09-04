import os

import numpy as np
import onnx
from torchvision import datasets, transforms
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ONNX_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "mobilenetv3_cifar10_fp32.onnx",
)

INT8_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "mobilenetv3_cifar10_int8.onnx",
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Number of calibration images.
CALIBRATION_SAMPLES = 500


class CIFAR10CalibrationDataReader(CalibrationDataReader):

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])

        dataset = datasets.CIFAR10(
            root=DATA_DIR,
            train=True,
            download=True,
            transform=self.transform,
        )

        self.data = []

        for index in range(CALIBRATION_SAMPLES):
            image, _ = dataset[index]

            # Add batch dimension.
            image = image.unsqueeze(0)

            self.data.append(
                {
                    "images": image.numpy().astype(
                        np.float32
                    )
                }
            )

        self.index = 0

    def get_next(self):
        if self.index >= len(self.data):
            return None

        item = self.data[self.index]

        self.index += 1

        return item

    def rewind(self):
        self.index = 0


def main():

    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(
            f"FP32 ONNX model not found: {ONNX_PATH}"
        )

    print("Starting INT8 post-training quantization...")
    print(f"Calibration samples: {CALIBRATION_SAMPLES}")

    calibration_data_reader = (
        CIFAR10CalibrationDataReader()
    )

    quantize_static(
        model_input=ONNX_PATH,
        model_output=INT8_PATH,
        calibration_data_reader=calibration_data_reader,

        # INT8 activations.
        activation_type=QuantType.QInt8,

        # INT8 weights.
        weight_type=QuantType.QInt8,

        # Per-channel weight quantization.
        per_channel=True,

        # Min/max calibration for activations.
        calibrate_method=CalibrationMethod.MinMax,

        # QOperator representation.
        #quant_format=QuantFormat.QOperator,
        quant_format=QuantFormat.QDQ,

    )

    size_mb = (
        os.path.getsize(INT8_PATH)
        / (1024 ** 2)
    )

    print("\nINT8 quantization complete.")
    print(f"Output: {INT8_PATH}")
    print(f"Model size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
