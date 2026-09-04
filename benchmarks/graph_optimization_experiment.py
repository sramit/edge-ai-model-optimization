"""
Phase 2 / Experiment 1: does explicitly enabling ONNX Runtime graph
optimization change INT8 CPU inference latency vs. the current
src/benchmark.py behavior?

This script does not modify src/benchmark.py or src/evaluate_int8.py, does
not retrain, does not touch quantization parameters/calibration, does not
change the execution provider, and does not modify any canonical model file.

It:
  A. Calls the real, unmodified benchmark()/get_fixed_input() from
     src/benchmark.py to establish the authoritative baseline.
  B/C. Adds a local benchmark_with_options() that reuses the exact same
     WARMUP_RUNS / BENCHMARK_RUNS / NUM_ROUNDS / latency statistics from
     src/benchmark.py, but accepts an explicit onnxruntime.SessionOptions,
     so graph optimization can be controlled without touching the canonical
     script. Both the implicit-default config and the explicit
     ORT_ENABLE_ALL config are run through this same harness, repeated
     across multiple independent sessions, to characterize run-to-run noise.
  I. Saves the ORT-optimized graph via optimized_model_filepath and diffs
     its node/op-type histogram against the original INT8 graph to verify
     whether any transformation/fusion actually occurred.
  Accuracy check reuses evaluate_int8.py's exact methodology (not modified)
     to confirm accuracy is unchanged.

Outputs are written to benchmarks/graph_opt_experiment/ (gitignored).
"""

import json
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision import datasets, transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from benchmark import (  # noqa: E402
    FP32_PATH,
    INT8_PATH,
    WARMUP_RUNS,
    BENCHMARK_RUNS,
    NUM_ROUNDS,
    get_fixed_input,
    get_model_size,
    benchmark as canonical_benchmark,
)
from evaluate_int8 import (  # noqa: E402
    DATA_DIR,
    BATCH_SIZE,
    transform as eval_transform,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "graph_opt_experiment")

# Independent session creations per configuration, used only to measure
# session-to-session noise. Does not change WARMUP_RUNS/BENCHMARK_RUNS/
# NUM_ROUNDS/latency statistics from src/benchmark.py.
REPEAT_SESSIONS = 8


def benchmark_with_options(model_path, name, fixed_input, sess_options):
    """Same methodology as src.benchmark.benchmark(), but with an explicit
    SessionOptions and per-round median tracking for noise analysis."""

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    latencies = []
    round_medians = []

    for _round_index in range(NUM_ROUNDS):

        for _ in range(WARMUP_RUNS):
            session.run(None, {input_name: fixed_input})

        round_latencies = []
        for _ in range(BENCHMARK_RUNS):
            start = time.perf_counter()
            session.run(None, {input_name: fixed_input})
            end = time.perf_counter()
            lat = (end - start) * 1000
            latencies.append(lat)
            round_latencies.append(lat)

        round_medians.append(float(np.median(round_latencies)))

    latencies = np.array(latencies)
    median = float(np.median(latencies))
    mean = float(np.mean(latencies))
    std = float(np.std(latencies))
    p95 = float(np.percentile(latencies, 95))
    model_size = get_model_size(model_path)

    print(f"\n{name}")
    print("-" * 45)
    print(f"Model size     : {model_size:.2f} MB")
    print(f"Median latency : {median:.3f} ms")
    print(f"Mean latency   : {mean:.3f} ms")
    print(f"Std deviation  : {std:.3f} ms")
    print(f"P95 latency    : {p95:.3f} ms")
    print(f"Throughput     : {1000 / median:.2f} FPS")
    print(f"Per-round medians (ms): {[round(m, 3) for m in round_medians]}")

    return {
        "median": median,
        "mean": mean,
        "std": std,
        "p95": p95,
        "model_size": model_size,
        "round_medians": round_medians,
    }


def interleaved_sessions(
    model_path,
    fixed_input,
    configs,
    n=REPEAT_SESSIONS,
):
    """Runs benchmark_with_options() for each named (label, sess_options_factory)
    config in `configs`, interleaving configs run-by-run (A,B,A,B,...) rather
    than running all of A then all of B. This controls for time-based drift
    (CPU frequency scaling, thermal state) that could otherwise masquerade as
    a difference between configs. Returns {label: {...}} per config."""

    results = {label: [] for label, _ in configs}

    for i in range(n):
        for label, factory in configs:
            result = benchmark_with_options(
                model_path,
                f"{label} (interleaved round {i + 1}/{n})",
                fixed_input,
                factory(),
            )
            results[label].append(result)

    summary = {}
    for label, runs in results.items():
        session_medians = [r["median"] for r in runs]
        print(
            f"\n{label}: across-session medians (ms) = "
            f"{[round(m, 4) for m in session_medians]}"
        )
        print(
            f"{label}: mean of session medians = {np.mean(session_medians):.4f} ms, "
            f"std of session medians = {np.std(session_medians):.4f} ms"
        )
        summary[label] = {
            "runs": runs,
            "session_medians": session_medians,
            "session_median_mean": float(np.mean(session_medians)),
            "session_median_std": float(np.std(session_medians)),
        }

    return summary


def evaluate_accuracy(session, name):
    input_name = session.get_inputs()[0].name

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=eval_transform,
    )

    correct = 0
    total = 0
    for start in range(0, len(dataset), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(dataset))
        images = torch.stack([dataset[i][0] for i in range(start, end)])
        labels = np.array([dataset[i][1] for i in range(start, end)])
        outputs = session.run(
            None,
            {input_name: images.numpy().astype(np.float32)},
        )[0]
        predictions = outputs.argmax(axis=1)
        correct += (predictions == labels).sum()
        total += len(labels)

    accuracy = 100.0 * correct / total
    print(f"{name} accuracy: {accuracy:.2f}%")
    return float(accuracy)


def inspect_graph(path, label):
    model = onnx.load(path)
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    print(f"\n{label}")
    print(f"  Path: {path}")
    print(f"  Total nodes: {len(model.graph.node)}")
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count}")

    return op_counts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Phase 2 / Experiment 1: ONNX Runtime Graph Optimization (INT8)")
    print("=" * 60)
    print(f"onnxruntime version: {ort.__version__}")

    # --- Step 0: verify (not assume) the current default behavior ---
    default_so = ort.SessionOptions()
    print(
        "\nDefault SessionOptions().graph_optimization_level = "
        f"{default_so.graph_optimization_level}"
    )
    print(
        "src/benchmark.py calls ort.InferenceSession(model_path, "
        "providers=[...]) with no sess_options argument, so it relies on "
        "this default value."
    )

    fixed_input = get_fixed_input()

    # --- A. Baseline via the existing benchmark.py, unmodified ---
    print("\n\n[A] BASELINE — src/benchmark.py's own benchmark(), unmodified")
    fp32_canonical = canonical_benchmark(
        FP32_PATH, "FP32 (canonical src/benchmark.py)", fixed_input
    )
    int8_canonical = canonical_benchmark(
        INT8_PATH,
        "INT8 (canonical src/benchmark.py, implicit default SessionOptions)",
        fixed_input,
    )

    # --- B/C/H. Harness-matched comparison: implicit-default vs explicit ORT_ENABLE_ALL ---
    optimized_model_path = os.path.join(
        OUTPUT_DIR, "int8_optimized_ORT_ENABLE_ALL.onnx"
    )

    def default_so_factory():
        return ort.SessionOptions()

    def explicit_so_factory():
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Only the final session's save call matters for graph inspection;
        # cheap to set on every repeat.
        so.optimized_model_filepath = optimized_model_path
        return so

    print(
        "\n\n[B/C] Harness-matched INT8 runs: implicit default SessionOptions "
        "vs explicit graph_optimization_level=ORT_ENABLE_ALL, interleaved to "
        "control for time-based drift (thermal/frequency scaling)"
    )

    interleaved = interleaved_sessions(
        INT8_PATH,
        fixed_input,
        [
            ("INT8 default SessionOptions()", default_so_factory),
            ("INT8 explicit ORT_ENABLE_ALL", explicit_so_factory),
        ],
        n=REPEAT_SESSIONS,
    )

    default_repeats = interleaved["INT8 default SessionOptions()"]
    explicit_repeats = interleaved["INT8 explicit ORT_ENABLE_ALL"]

    baseline_median = default_repeats["session_median_mean"]
    experimental_median = explicit_repeats["session_median_mean"]

    abs_diff = experimental_median - baseline_median
    pct_diff = abs_diff / baseline_median * 100

    noise_default = default_repeats["session_median_std"]
    noise_explicit = explicit_repeats["session_median_std"]
    noise_estimate = max(noise_default, noise_explicit)

    # Paired per-round difference (default vs explicit within the same
    # interleaved round) — the statistically appropriate comparison since
    # both configs were interleaved to share the same time-varying
    # conditions.
    paired_diffs = [
        e - d
        for d, e in zip(
            default_repeats["session_medians"], explicit_repeats["session_medians"]
        )
    ]
    paired_mean = float(np.mean(paired_diffs))
    paired_std = float(np.std(paired_diffs))

    print("\n\n[G] Harness-matched default vs explicit-ORT_ENABLE_ALL comparison")
    print("-" * 60)
    print(f"Default-config mean-of-session-medians   : {baseline_median:.4f} ms")
    print(f"Explicit-config mean-of-session-medians  : {experimental_median:.4f} ms")
    print(f"Absolute difference (mean of means)      : {abs_diff:+.4f} ms")
    print(f"Percentage difference                    : {pct_diff:+.3f}%")

    print("\n[H] Run-to-run noise")
    print(f"  Default config  — session-to-session std: {noise_default:.4f} ms")
    print(f"  Explicit config — session-to-session std: {noise_explicit:.4f} ms")
    print(
        f"  Paired per-round diffs (explicit - default), ms: "
        f"{[round(x, 4) for x in paired_diffs]}"
    )
    print(f"  Paired mean diff: {paired_mean:+.4f} ms, paired std: {paired_std:.4f} ms")
    print(
        f"  |paired mean diff| = {abs(paired_mean):.4f} ms vs "
        f"paired std (noise) = {paired_std:.4f} ms"
    )
    meaningful = abs(paired_mean) > paired_std
    print(
        f"  => Observed difference is "
        f"{'LARGER' if meaningful else 'NOT larger'} than paired run-to-run noise."
    )

    # --- I. Inspect whether the optimized graph actually changed ---
    print("\n\n[I] Graph inspection: did ORT apply any fusions/transformations?")
    original_ops = inspect_graph(INT8_PATH, "Original canonical INT8 graph")

    graph_changed = None
    if os.path.exists(optimized_model_path):
        optimized_ops = inspect_graph(
            optimized_model_path, "ORT-saved optimized graph (ORT_ENABLE_ALL)"
        )
        graph_changed = original_ops != optimized_ops
        if not graph_changed:
            print(
                "\nResult: node op-type histogram is IDENTICAL between the "
                "original and ORT-optimized graph. No fusions/transformations "
                "were applied to this graph at ORT_ENABLE_ALL."
            )
        else:
            print(
                "\nResult: node op-type histogram DIFFERS. Graph "
                "transformations occurred:"
            )
            all_ops = set(original_ops) | set(optimized_ops)
            for op in sorted(all_ops):
                o = original_ops.get(op, 0)
                n = optimized_ops.get(op, 0)
                if o != n:
                    print(f"  {op}: {o} -> {n}")
    else:
        print(
            f"WARNING: optimized model file was not written to "
            f"{optimized_model_path}"
        )

    print(
        "\nNote: this diff is against the on-disk (pre-optimization) INT8 "
        "file. Since default SessionOptions() ALSO runs at ORT_ENABLE_ALL "
        "internally, these fusions occur identically for both the default "
        "and explicit configurations at inference time — the diff shows "
        "that ORT_ENABLE_ALL performs real fusions on this graph in "
        "general, not that the explicit setting differs from default."
    )

    # --- Accuracy check (evaluate_int8.py methodology, file not modified) ---
    print(
        "\n\n[Accuracy] Verifying INT8 accuracy is unchanged under explicit "
        "graph optimization (evaluate_int8.py methodology, file not modified)"
    )

    default_session = ort.InferenceSession(
        INT8_PATH, providers=["CPUExecutionProvider"]
    )
    explicit_session = ort.InferenceSession(
        INT8_PATH, sess_options=explicit_so_factory(), providers=["CPUExecutionProvider"]
    )

    acc_default = evaluate_accuracy(default_session, "INT8 (default SessionOptions)")
    acc_explicit = evaluate_accuracy(explicit_session, "INT8 (explicit ORT_ENABLE_ALL)")

    print(f"\nAccuracy delta: {acc_explicit - acc_default:+.4f} percentage points")

    # --- Save machine-readable summary ---
    summary = {
        "ort_version": ort.__version__,
        "default_graph_optimization_level": str(default_so.graph_optimization_level),
        "fp32_canonical_benchmark": fp32_canonical,
        "int8_canonical_benchmark": int8_canonical,
        "int8_default_repeats": default_repeats,
        "int8_explicit_ORT_ENABLE_ALL_repeats": explicit_repeats,
        "latency_abs_diff_ms": abs_diff,
        "latency_pct_diff": pct_diff,
        "noise_default_session_std_ms": noise_default,
        "noise_explicit_session_std_ms": noise_explicit,
        "paired_diffs_ms": paired_diffs,
        "paired_mean_diff_ms": paired_mean,
        "paired_std_ms": paired_std,
        "diff_larger_than_noise": meaningful,
        "graph_changed": graph_changed,
        "accuracy_default_pct": acc_default,
        "accuracy_explicit_pct": acc_explicit,
        "accuracy_delta_pct_points": acc_explicit - acc_default,
    }

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
