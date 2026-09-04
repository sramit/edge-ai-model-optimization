"""
Operator-level latency profiling for FP32 vs INT8 (QDQ) ONNX models.

Standalone investigative tooling — separate from the production pipeline
(src/). Read-only with respect to model.py, train.py, quantize.py,
evaluate_int8.py and the model artifacts under results/. Uses ONNX
Runtime's built-in profiler with CPUExecutionProvider, matching the
input/session setup already used in src/benchmark.py, so results are
directly comparable to the existing benchmark numbers.

Usage:
    python benchmarks/profile_operators.py
"""

import json
import os
from collections import defaultdict

import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FP32_PATH = os.path.join(
    PROJECT_ROOT, "results", "mobilenetv3_cifar10_fp32.onnx"
)

INT8_PATH = os.path.join(
    PROJECT_ROOT, "results", "mobilenetv3_cifar10_int8.onnx"
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "profiling_results"
)

# Matches src/benchmark.py's deterministic single-sample setup.
FIXED_SAMPLE_INDEX = 0
WARMUP_RUNS = 50
PROFILE_RUNS = 300

# Operators of specific interest per the investigation.
OPS_OF_INTEREST = [
    "Conv",
    "QLinearConv",
    "QuantizeLinear",
    "DequantizeLinear",
    "Gemm",
    "QGemm",
    "Mul",
    "QLinearMul",
    "Add",
    "QLinearAdd",
]


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
    image = image.unsqueeze(0)

    return image.numpy().astype(np.float32)


def run_profiling(model_path, name, fixed_input):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    sess_options.profile_file_prefix = os.path.join(
        OUTPUT_DIR, f"{name}_profile"
    )

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    for _ in range(WARMUP_RUNS):
        session.run(None, {input_name: fixed_input})

    for _ in range(PROFILE_RUNS):
        session.run(None, {input_name: fixed_input})

    trace_path = session.end_profiling()

    return trace_path


def parse_trace(trace_path):
    """
    Aggregate per-node ONNX Runtime kernel-time events by op type.

    Only events with cat == "Node" and a name ending in "_kernel_time"
    are counted as compute time; fence/memcpy bookkeeping events are
    excluded to avoid inflating totals.
    """
    with open(trace_path) as f:
        events = json.load(f)

    if isinstance(events, dict):
        events = events.get("traceEvents", [])

    op_time_us = defaultdict(float)
    op_count = defaultdict(int)

    for event in events:
        if event.get("cat") != "Node":
            continue

        name = event.get("name", "")

        if not name.endswith("_kernel_time"):
            continue

        args = event.get("args", {})
        op_type = args.get("op_name")

        if op_type is None:
            continue

        op_time_us[op_type] += event.get("dur", 0)
        op_count[op_type] += 1

    return op_time_us, op_count


def build_report(op_time_us, op_count, num_runs):
    total_time_us = sum(op_time_us.values())

    rows = []

    for op_type, time_us in op_time_us.items():
        rows.append({
            "op_type": op_type,
            "total_time_us": time_us,
            "pct_of_total": (
                100.0 * time_us / total_time_us if total_time_us else 0.0
            ),
            "count": op_count[op_type],
            "avg_time_per_call_us": time_us / op_count[op_type],
        })

    rows.sort(key=lambda r: r["total_time_us"], reverse=True)

    return {
        "total_time_us": total_time_us,
        "total_time_ms": total_time_us / 1000.0,
        "avg_inference_time_ms": total_time_us / 1000.0 / num_runs,
        "num_runs": num_runs,
        "operators": rows,
    }


def print_report(name, report):
    print(f"\n{name}")
    print("-" * 70)
    print(f"Profiled runs            : {report['num_runs']}")
    print(f"Total op-level time      : {report['total_time_ms']:.3f} ms "
          f"(summed across all runs)")
    print(f"Avg op-level time/run    : {report['avg_inference_time_ms']:.4f} ms")

    print(f"\nTop 10 operators by cumulative execution time:")
    print(f"{'Op Type':<20}{'Time (ms)':>12}{'% Total':>10}{'Count':>10}{'Avg/call (us)':>16}")

    for row in report["operators"][:10]:
        print(
            f"{row['op_type']:<20}"
            f"{row['total_time_us'] / 1000.0:>12.3f}"
            f"{row['pct_of_total']:>9.2f}%"
            f"{row['count']:>10}"
            f"{row['avg_time_per_call_us']:>16.3f}"
        )

    print(f"\nOperators of interest:")
    by_type = {row["op_type"]: row for row in report["operators"]}

    for op_type in OPS_OF_INTEREST:
        if op_type in by_type:
            row = by_type[op_type]
            print(
                f"  {op_type:<20}"
                f"{row['total_time_us'] / 1000.0:>10.3f} ms"
                f"{row['pct_of_total']:>9.2f}%"
                f"{row['count']:>8} calls"
            )
        else:
            print(f"  {op_type:<20}  (not present in this model)")


def profile_model(model_path, name, fixed_input):
    trace_path = run_profiling(model_path, name, fixed_input)
    op_time_us, op_count = parse_trace(trace_path)
    report = build_report(op_time_us, op_count, PROFILE_RUNS)

    report["trace_path"] = trace_path

    print_report(name, report)

    summary_path = os.path.join(OUTPUT_DIR, f"{name}_summary.json")

    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nRaw trace   : {trace_path}")
    print(f"Summary JSON: {summary_path}")

    return report


def compare_reports(fp32_report, int8_report):
    print("\n\nFP32 vs INT8 Operator Composition")
    print("=" * 70)

    print(
        f"FP32 total op-level time : {fp32_report['total_time_ms']:.3f} ms "
        f"over {fp32_report['num_runs']} runs "
        f"({fp32_report['avg_inference_time_ms']:.4f} ms/run avg)"
    )
    print(
        f"INT8 total op-level time : {int8_report['total_time_ms']:.3f} ms "
        f"over {int8_report['num_runs']} runs "
        f"({int8_report['avg_inference_time_ms']:.4f} ms/run avg)"
    )

    fp32_op_count = len(fp32_report["operators"])
    int8_op_count = len(int8_report["operators"])

    print(f"\nDistinct op types  : FP32={fp32_op_count}  INT8={int8_op_count}")

    fp32_node_count = sum(r["count"] for r in fp32_report["operators"])
    int8_node_count = sum(r["count"] for r in int8_report["operators"])

    print(f"Total node executions/run: "
          f"FP32={fp32_node_count / fp32_report['num_runs']:.1f}  "
          f"INT8={int8_node_count / int8_report['num_runs']:.1f}")

    # Q/DQ contribution within INT8.
    by_type = {row["op_type"]: row for row in int8_report["operators"]}

    qdq_time_us = sum(
        by_type[op]["total_time_us"]
        for op in ("QuantizeLinear", "DequantizeLinear")
        if op in by_type
    )
    qdq_count = sum(
        by_type[op]["count"]
        for op in ("QuantizeLinear", "DequantizeLinear")
        if op in by_type
    )
    qdq_pct = (
        100.0 * qdq_time_us / int8_report["total_time_us"]
        if int8_report["total_time_us"] else 0.0
    )

    print(f"\nQuantizeLinear + DequantizeLinear in INT8:")
    print(f"  Combined time : {qdq_time_us / 1000.0:.3f} ms "
          f"({qdq_pct:.2f}% of INT8 op-level total)")
    print(f"  Combined calls: {qdq_count}")

    print(
        "\nNote: this reports the measured share of profiled time spent in "
        "Q/DQ nodes. Whether that share is large enough to *explain* the "
        "wall-clock gap vs FP32 should be read against the numbers above, "
        "not asserted independently of them."
    )


if __name__ == "__main__":
    print("ONNX Runtime Operator-Level Profiling: FP32 vs INT8")
    print("=" * 70)
    print(f"Warmup runs per model: {WARMUP_RUNS}")
    print(f"Profiled runs per model: {PROFILE_RUNS}")
    print("Execution provider: CPUExecutionProvider")

    fixed_input = get_fixed_input()

    fp32_report = profile_model(FP32_PATH, "fp32", fixed_input)
    int8_report = profile_model(INT8_PATH, "int8", fixed_input)

    compare_reports(fp32_report, int8_report)
