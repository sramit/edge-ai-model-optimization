import argparse
import os
import time

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

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Number of calibration images (default; overridable with --calibration-samples).
CALIBRATION_SAMPLES = 1000

# Output filename within results/ (default; overridable with --output).
DEFAULT_OUTPUT_NAME = "mobilenetv3_cifar10_int8.onnx"

# Activation calibration method choices (default; overridable with --calibrate-method).
CALIBRATION_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}


class CIFAR10CalibrationDataReader(CalibrationDataReader):

    def __init__(self, num_samples=CALIBRATION_SAMPLES):
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

        for index in range(num_samples):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="INT8 post-training static quantization (QDQ)."
    )

    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=CALIBRATION_SAMPLES,
        help="Number of CIFAR-10 training images used for calibration.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Output filename, written under results/.",
    )

    parser.add_argument(
        "--calibrate-method",
        type=str,
        choices=sorted(CALIBRATION_METHODS),
        default="percentile",
        help="Activation calibration method.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    int8_path = os.path.join(RESULTS_DIR, args.output)

    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(
            f"FP32 ONNX model not found: {ONNX_PATH}"
        )

    print("Starting INT8 post-training quantization...")
    print(f"Calibration samples: {args.calibration_samples}")
    print(f"Calibration method: {args.calibrate_method}")

    calibration_start = time.perf_counter()

    calibration_data_reader = CIFAR10CalibrationDataReader(
        num_samples=args.calibration_samples
    )

    quantize_static(
        model_input=ONNX_PATH,
        model_output=int8_path,
        calibration_data_reader=calibration_data_reader,

        # INT8 activations.
        activation_type=QuantType.QInt8,

        # INT8 weights.
        weight_type=QuantType.QInt8,

        # Per-channel weight quantization.
        per_channel=True,

        # Activation calibration method.
        calibrate_method=CALIBRATION_METHODS[args.calibrate_method],

        # QOperator representation.
        #quant_format=QuantFormat.QOperator,
        quant_format=QuantFormat.QDQ,

    )

    calibration_time = time.perf_counter() - calibration_start

    size_mb = (
        os.path.getsize(int8_path)
        / (1024 ** 2)
    )

    print("\nINT8 quantization complete.")
    print(f"Output: {int8_path}")
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Calibration + quantization time: {calibration_time:.2f} s")


if __name__ == "__main__":
    main()
