import os
import time

import numpy as np
import onnxruntime as ort


FP32_PATH = "./results/mobilenetv3_cifar10_fp32.onnx"
INT8_PATH = "./results/mobilenetv3_cifar10_int8.onnx"

INPUT_SHAPE = (1, 3, 32, 32)

WARMUP_RUNS = 50
BENCHMARK_RUNS = 500


def get_model_size(path):
    total_size = os.path.getsize(path)

    data_path = path + ".data"

    if os.path.exists(data_path):
        total_size += os.path.getsize(data_path)

    return total_size / (1024 ** 2)


def benchmark(model_path, name):

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    dummy_input = np.random.randn(
        *INPUT_SHAPE
    ).astype(np.float32)

    # Warmup
    for _ in range(WARMUP_RUNS):
        session.run(
            None,
            {input_name: dummy_input},
        )

    latencies = []

    # Benchmark
    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        session.run(
            None,
            {input_name: dummy_input},
        )

        end = time.perf_counter()

        latencies.append(
            (end - start) * 1000
        )

    latencies = np.array(latencies)

    median = np.median(latencies)
    p95 = np.percentile(latencies, 95)

    print(f"\n{name}")
    print("-" * 45)

    print(
        f"Model size     : "
        f"{get_model_size(model_path):.2f} MB"
    )

    print(
        f"Median latency : "
        f"{median:.3f} ms"
    )

    print(
        f"P95 latency    : "
        f"{p95:.3f} ms"
    )

    print(
        f"Throughput     : "
        f"{1000 / median:.2f} FPS"
    )

    return median, p95


if __name__ == "__main__":

    print("ONNX Runtime FP32 vs INT8 Benchmark")
    print("=" * 50)

    fp32_median, fp32_p95 = benchmark(
        FP32_PATH,
        "FP32",
    )

    int8_median, int8_p95 = benchmark(
        INT8_PATH,
        "INT8",
    )

    latency_improvement = (
        (fp32_median - int8_median)
        / fp32_median
        * 100
    )

    print("\nOptimization Impact")
    print("-" * 45)

    print(
        f"Latency improvement: "
        f"{latency_improvement:.2f}%"
    )

    print(
        f"Speedup: "
        f"{fp32_median / int8_median:.2f}x"
    )