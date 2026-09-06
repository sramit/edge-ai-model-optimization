"""
Quantization error analysis for FP32 vs INT8 ONNX models.

Phase 2 Experiment 3:
Measure numerical differences introduced by INT8 quantization
at the final model output while using identical CIFAR-10 inputs.

Standalone investigative tooling — does not modify the production
training, export, quantization, or evaluation pipeline.

Usage:
    python benchmarks/quantization_error_experiment.py
"""

import json
import os

import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FP32_PATH = os.path.join(
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

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "quantization_error_results",
)

NUM_SAMPLES = 10000
BATCH_SIZE = 128


def get_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    return datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )


def create_session(model_path):
    return ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )


def collect_outputs(session, dataset, num_samples):
    input_name = session.get_inputs()[0].name

    outputs = []

    num_samples = min(num_samples, len(dataset))

    for start in range(0, num_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_samples)

        images = np.stack([
            dataset[i][0].numpy()
            for i in range(start, end)
        ]).astype(np.float32)

        batch_output = session.run(
            None,
            {input_name: images},
        )[0]

        outputs.append(batch_output)

    return np.concatenate(outputs, axis=0)


def calculate_metrics(fp32_output, int8_output):
    difference = int8_output - fp32_output
    absolute_difference = np.abs(difference)

    mae = float(np.mean(absolute_difference))

    rmse = float(
        np.sqrt(np.mean(np.square(difference)))
    )

    max_absolute_error = float(
        np.max(absolute_difference)
    )

    p95_absolute_error = float(
        np.percentile(absolute_difference, 95)
    )

    p99_absolute_error = float(
        np.percentile(absolute_difference, 99)
    )

    # Avoid division by zero when calculating relative error.
    denominator = np.maximum(np.abs(fp32_output), 1e-12)

    relative_error = (
        absolute_difference / denominator
    )

    mean_relative_error = float(
        np.mean(relative_error)
    )

    # Cosine similarity across the flattened output tensors.
    fp32_flat = fp32_output.reshape(-1)
    int8_flat = int8_output.reshape(-1)

    denominator = (
        np.linalg.norm(fp32_flat)
        * np.linalg.norm(int8_flat)
    )

    if denominator == 0:
        cosine_similarity = 0.0
    else:
        cosine_similarity = float(
            np.dot(fp32_flat, int8_flat) / denominator
        )

    fp32_predictions = fp32_output.argmax(axis=1)
    int8_predictions = int8_output.argmax(axis=1)

    prediction_agreement = float(
        np.mean(
            fp32_predictions == int8_predictions
        ) * 100.0
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "max_absolute_error": max_absolute_error,
        "mean_relative_error": mean_relative_error,
        "cosine_similarity": cosine_similarity,
        "prediction_agreement_percent": prediction_agreement,
        "p95_absolute_error": p95_absolute_error,
        "p99_absolute_error": p99_absolute_error,
    }


def calculate_accuracy(outputs, dataset, num_samples):
    labels = np.array([
        dataset[i][1]
        for i in range(num_samples)
    ])

    predictions = outputs.argmax(axis=1)

    return float(
        np.mean(predictions == labels) * 100.0
    )


def print_report(metrics, fp32_accuracy, int8_accuracy):
    print("\nQuantization Error Analysis")
    print("=" * 60)

    print(f"MAE                    : {metrics['mae']:.8f}")
    print(f"RMSE                   : {metrics['rmse']:.8f}")
    print(
        f"Max absolute error    : "
        f"{metrics['max_absolute_error']:.8f}"
    )
    print(
    f"P95 absolute error    : "
    f"{metrics['p95_absolute_error']:.8f}"
    )

    print(
    f"P99 absolute error    : "
    f"{metrics['p99_absolute_error']:.8f}"  
    )
    print(
        f"Mean relative error   : "
        f"{metrics['mean_relative_error']:.8f}"
    )
    print(
        f"Cosine similarity     : "
        f"{metrics['cosine_similarity']:.8f}"
    )
    print(
        f"Prediction agreement  : "
        f"{metrics['prediction_agreement_percent']:.2f}%"
    )

    print("\nAccuracy")
    print("-" * 60)

    print(f"FP32 accuracy         : {fp32_accuracy:.2f}%")
    print(f"INT8 accuracy         : {int8_accuracy:.2f}%")
    print(
        f"Accuracy drop         : "
        f"{fp32_accuracy - int8_accuracy:.2f} "
        f"percentage points"
    )


def save_results(metrics, fp32_accuracy, int8_accuracy, num_samples):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "experiment": "Phase 2 Experiment 3",
        "description": (
            "End-to-end FP32 vs INT8 output error analysis"
        ),
        "fp32_model": FP32_PATH,
        "int8_model": INT8_PATH,
        "num_samples": num_samples,
        "batch_size": BATCH_SIZE,
        "metrics": metrics,
        "accuracy": {
            "fp32_percent": fp32_accuracy,
            "int8_percent": int8_accuracy,
            "drop_percentage_points": (
                fp32_accuracy - int8_accuracy
            ),
        },
    }

    output_path = os.path.join(
        OUTPUT_DIR,
        "quantization_error_summary.json",
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    print("Phase 2 Experiment 3")
    print("FP32 vs INT8 Quantization Error Analysis")
    print("=" * 60)

    dataset = get_dataset()

    num_samples = min(
        NUM_SAMPLES,
        len(dataset),
    )

    print(f"Samples evaluated      : {num_samples}")
    print(f"Batch size             : {BATCH_SIZE}")
    print("Execution provider     : CPUExecutionProvider")

    print("\nLoading FP32 model...")
    fp32_session = create_session(FP32_PATH)

    print("Loading INT8 model...")
    int8_session = create_session(INT8_PATH)

    print("\nRunning FP32 inference...")
    fp32_output = collect_outputs(
        fp32_session,
        dataset,
        num_samples,
    )

    print("Running INT8 inference...")
    int8_output = collect_outputs(
        int8_session,
        dataset,
        num_samples,
    )

    print("\nCalculating error metrics...")
    metrics = calculate_metrics(
        fp32_output,
        int8_output,
    )

    fp32_accuracy = calculate_accuracy(
        fp32_output,
        dataset,
        num_samples,
    )

    int8_accuracy = calculate_accuracy(
        int8_output,
        dataset,
        num_samples,
    )

    print_report(
        metrics,
        fp32_accuracy,
        int8_accuracy,
    )

    save_results(
        metrics,
        fp32_accuracy,
        int8_accuracy,
        num_samples,
    )


if __name__ == "__main__":
    main()
