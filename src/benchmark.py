import os
import time

import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# Fixed CIFAR-10 test-set index used as the deterministic benchmark input.
FIXED_SAMPLE_INDEX = 0

WARMUP_RUNS = 50
BENCHMARK_RUNS = 500
NUM_ROUNDS = 5


def get_model_size(path):
    total_size = os.path.getsize(path)

    data_path = path + ".data"

    if os.path.exists(data_path):
        total_size += os.path.getsize(data_path)

    return total_size / (1024 ** 2)


def get_fixed_input():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    image, _ = test_dataset[FIXED_SAMPLE_INDEX]

    # Add batch dimension.
    image = image.unsqueeze(0)

    return image.numpy().astype(np.float32)


def benchmark(model_path, name, fixed_input):

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    latencies = []

    for round_index in range(NUM_ROUNDS):

        # Warmup
        for _ in range(WARMUP_RUNS):
            session.run(
                None,
                {input_name: fixed_input},
            )

        # Benchmark
        for _ in range(BENCHMARK_RUNS):

            start = time.perf_counter()

            session.run(
                None,
                {input_name: fixed_input},
            )

            end = time.perf_counter()

            latencies.append(
                (end - start) * 1000
            )

    latencies = np.array(latencies)

    median = np.median(latencies)
    mean = np.mean(latencies)
    std = np.std(latencies)
    p95 = np.percentile(latencies, 95)
    model_size = get_model_size(model_path)

    print(f"\n{name}")
    print("-" * 45)

    print(f"Model size     : {model_size:.2f} MB")
    print(f"Median latency : {median:.3f} ms")
    print(f"Mean latency   : {mean:.3f} ms")
    print(f"Std deviation  : {std:.3f} ms")
    print(f"P95 latency    : {p95:.3f} ms")
    print(f"Throughput     : {1000 / median:.2f} FPS")

    return {
        "median": median,
        "mean": mean,
        "std": std,
        "p95": p95,
        "model_size": model_size,
    }


if __name__ == "__main__":

    print("ONNX Runtime FP32 vs INT8 Benchmark")
    print("=" * 50)

    print(
        f"Rounds: {NUM_ROUNDS} x {BENCHMARK_RUNS} runs "
        f"(warmup: {WARMUP_RUNS} per round)"
    )

    fixed_input = get_fixed_input()

    fp32_stats = benchmark(FP32_PATH, "FP32", fixed_input)
    int8_stats = benchmark(INT8_PATH, "INT8", fixed_input)

    latency_improvement = (
        (fp32_stats["median"] - int8_stats["median"])
        / fp32_stats["median"]
        * 100
    )

    size_reduction = (
        (fp32_stats["model_size"] - int8_stats["model_size"])
        / fp32_stats["model_size"]
        * 100
    )

    print("\nOptimization Impact")
    print("-" * 45)

    print(f"Latency improvement : {latency_improvement:.2f}%")
    print(f"Speedup             : {fp32_stats['median'] / int8_stats['median']:.2f}x")
    print(f"Model size reduction: {size_reduction:.2f}%")
