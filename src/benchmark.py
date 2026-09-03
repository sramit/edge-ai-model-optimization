import os
import time

import torch
import torch.nn as nn
from torchvision import models


CHECKPOINT_PATH = "./results/mobilenetv3_cifar10_fp32.pth"
INPUT_SHAPE = (1, 3, 32, 32)

WARMUP_RUNS = 20
BENCHMARK_RUNS = 100


def build_model():
    model = models.mobilenet_v3_small(weights=None)

    # Same CIFAR-10 adaptations used during training.
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


def load_model(device):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def benchmark_cpu():
    device = torch.device("cpu")
    model = load_model(device)

    dummy_input = torch.randn(INPUT_SHAPE, device=device)

    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):
            _ = model(dummy_input)

        latencies = []

        for _ in range(BENCHMARK_RUNS):
            start = time.perf_counter()

            _ = model(dummy_input)

            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    latencies.sort()

    median = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    print("\nCPU Benchmark")
    print("-" * 40)
    print(f"Median latency : {median:.3f} ms")
    print(f"P95 latency    : {p95:.3f} ms")
    print(f"Throughput     : {1000 / median:.2f} FPS")


def benchmark_gpu():
    if not torch.cuda.is_available():
        print("\nCUDA is not available.")
        return

    device = torch.device("cuda")
    model = load_model(device)

    dummy_input = torch.randn(INPUT_SHAPE, device=device)

    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):
            _ = model(dummy_input)

        torch.cuda.synchronize()

        latencies = []

        for _ in range(BENCHMARK_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(dummy_input)

            torch.cuda.synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

    latencies.sort()

    median = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    print("\nGPU Benchmark")
    print("-" * 40)
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Median latency : {median:.3f} ms")
    print(f"P95 latency    : {p95:.3f} ms")
    print(f"Throughput     : {1000 / median:.2f} FPS")


def model_statistics():
    model = build_model()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    checkpoint_size_mb = (
        os.path.getsize(CHECKPOINT_PATH) / (1024 ** 2)
    )

    print("\nModel Statistics")
    print("-" * 40)
    print(f"Parameters     : {parameter_count:,}")
    print(f"Checkpoint size: {checkpoint_size_mb:.2f} MB")


if __name__ == "__main__":
    print("FP32 MobileNetV3-Small Benchmark")
    print("=" * 40)

    model_statistics()
    benchmark_cpu()
    benchmark_gpu()
