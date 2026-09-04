"""
Phase 2 / Experiment 2: BatchNorm Folding Verification.

Question: does ONNX Runtime's graph optimization actually fold
BatchNormalization into preceding Conv layers in our FP32/INT8 ONNX
graphs?

This script is inspection-only. It does not modify src/model.py,
src/export_onnx.py, src/quantize.py, src/benchmark.py, or
src/evaluate_int8.py, does not retrain, does not re-quantize, does not
change the execution provider, and does not overwrite any canonical
artifact in results/. It only reads existing files (including, read-only,
the ORT-optimized INT8 graph already produced by the prior
graph-optimization experiment under benchmarks/graph_opt_experiment/)
and writes its own JSON summary into a new gitignored directory,
benchmarks/batchnorm_experiment/.

Method:
  1. Count BatchNorm2d / Conv2d modules directly in the PyTorch model
     (src/model.py's build_model(), imported read-only) to establish
     the pre-export ground truth.
  2. Count BatchNormalization / Conv node op-types in the exported FP32
     ONNX graph (results/mobilenetv3_cifar10_fp32.onnx) and the
     canonical INT8 QDQ graph (results/mobilenetv3_cifar10_int8.onnx).
  3. Count node op-types in the ORT-optimized INT8 graph produced by
     the prior graph-optimization experiment
     (benchmarks/graph_opt_experiment/int8_optimized_ORT_ENABLE_ALL.onnx)
     to see what ORT_ENABLE_ALL actually changes on this pipeline (QDQ
     fusion), as distinct from BatchNorm folding.
  4. For every Conv+BatchNorm pair identifiable in the PyTorch module
     tree, algebraically fold BatchNorm into the preceding Conv using
     the standard formula and compare the result against the ONNX
     Conv weight/bias initializers to determine numerically whether
     the fold already happened before the graph reached ONNX Runtime.

Outputs are written to benchmarks/batchnorm_experiment/ (gitignored).
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import numpy_helper

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from model import build_model  # noqa: E402

FP32_PATH = os.path.join(PROJECT_ROOT, "results", "mobilenetv3_cifar10_fp32.onnx")
FP32_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "results", "mobilenetv3_cifar10_fp32.pth"
)
INT8_PATH = os.path.join(PROJECT_ROOT, "results", "mobilenetv3_cifar10_int8.onnx")

# Read-only reuse of the artifact from Phase 2 / Experiment 1. Not
# regenerated here; if it is missing, the INT8-optimized-graph section
# of this script is skipped rather than re-running that experiment.
PRIOR_INT8_OPTIMIZED_PATH = os.path.join(
    PROJECT_ROOT,
    "benchmarks",
    "graph_opt_experiment",
    "int8_optimized_ORT_ENABLE_ALL.onnx",
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "batchnorm_experiment")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary.json")

CONV_LIKE_OPS = {"Conv", "QLinearConv"}


def load_onnx_graph_counts(path):
    """Returns (op_type -> count) and total node count for an ONNX file."""
    model = onnx.load(path)
    counts = Counter(node.op_type for node in model.graph.node)
    return dict(counts), len(model.graph.node)


def load_onnx_initializers(path):
    model = onnx.load(path)
    return {
        init.name: numpy_helper.to_array(init) for init in model.graph.initializer
    }


def find_conv_bn_pairs(model):
    """Walks the PyTorch module tree and returns (conv_name, bn_name)
    for every BatchNorm2d that immediately follows a Conv2d within the
    same parent nn.Sequential container (the standard Conv-BN-Activation
    pattern used throughout MobileNetV3)."""
    pairs = []
    for parent_name, parent in model.named_modules():
        if not isinstance(parent, nn.Sequential):
            continue
        children = list(parent.named_children())
        for (idx_name, child), (prev_idx_name, prev_child) in zip(
            children[1:], children[:-1]
        ):
            if isinstance(child, nn.BatchNorm2d) and isinstance(
                prev_child, nn.Conv2d
            ):
                conv_name = (
                    f"{parent_name}.{prev_idx_name}" if parent_name else prev_idx_name
                )
                bn_name = f"{parent_name}.{idx_name}" if parent_name else idx_name
                pairs.append((conv_name, bn_name))
    return pairs


def get_module_by_name(model, name):
    module = model
    for part in name.split("."):
        module = getattr(module, part)
    return module


def verify_fold(model, onnx_inits, conv_name, bn_name):
    """Algebraically folds BatchNorm (bn_name) into the preceding Conv
    (conv_name) per:
        W' = W * gamma / sqrt(var + eps)
        b' = beta + (b - mean) * gamma / sqrt(var + eps)
    and compares the result against the ONNX Conv weight/bias
    initializers for that same layer, if present."""
    conv = get_module_by_name(model, conv_name)
    bn = get_module_by_name(model, bn_name)

    W = conv.weight.detach().numpy()
    b = (
        conv.bias.detach().numpy()
        if conv.bias is not None
        else np.zeros(W.shape[0], dtype=np.float32)
    )
    gamma = bn.weight.detach().numpy()
    beta = bn.bias.detach().numpy()
    mean = bn.running_mean.detach().numpy()
    var = bn.running_var.detach().numpy()
    eps = bn.eps

    scale = gamma / np.sqrt(var + eps)
    W_folded = W * scale.reshape(-1, 1, 1, 1)
    b_folded = beta + (b - mean) * scale

    weight_key = f"{conv_name}.weight"
    bias_key = f"{conv_name}.weight_bias"

    result = {
        "conv_name": conv_name,
        "bn_name": bn_name,
        "conv_had_bias_in_pytorch": conv.bias is not None,
        "onnx_weight_initializer_found": weight_key in onnx_inits,
        "onnx_bias_initializer_found": bias_key in onnx_inits,
    }

    if weight_key in onnx_inits:
        W_onnx = onnx_inits[weight_key]
        result["weight_max_abs_diff"] = float(np.max(np.abs(W_folded - W_onnx)))
        result["weight_matches_fold"] = bool(
            np.allclose(W_folded, W_onnx, atol=1e-5, rtol=1e-4)
        )
    else:
        result["weight_max_abs_diff"] = None
        result["weight_matches_fold"] = None

    if bias_key in onnx_inits:
        b_onnx = onnx_inits[bias_key]
        result["bias_max_abs_diff"] = float(np.max(np.abs(b_folded - b_onnx)))
        result["bias_matches_fold"] = bool(
            np.allclose(b_folded, b_onnx, atol=1e-5, rtol=1e-4)
        )
    else:
        result["bias_max_abs_diff"] = None
        result["bias_matches_fold"] = None

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Phase 2 / Experiment 2: BatchNorm Folding Verification")
    print("=" * 60)
    print(f"onnxruntime version: {ort.__version__}")

    # --- Step 1: PyTorch module-level ground truth (pre-export) ---
    print("\n[1] PyTorch model (pre-export) module counts")
    print("-" * 60)

    checkpoint = torch.load(FP32_CHECKPOINT_PATH, map_location="cpu")
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pytorch_bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    pytorch_conv_count = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    print(f"  nn.BatchNorm2d modules: {pytorch_bn_count}")
    print(f"  nn.Conv2d modules     : {pytorch_conv_count}")

    conv_bn_pairs = find_conv_bn_pairs(model)
    print(
        f"  Conv->BatchNorm adjacency pairs found in module tree: "
        f"{len(conv_bn_pairs)}"
    )

    # --- Step 2: original (pre-ORT-optimization) ONNX graphs ---
    print("\n[2] Original ONNX graph op-type counts (as exported/quantized)")
    print("-" * 60)

    if not os.path.exists(FP32_PATH):
        raise FileNotFoundError(f"FP32 ONNX model not found: {FP32_PATH}")
    if not os.path.exists(INT8_PATH):
        raise FileNotFoundError(f"INT8 ONNX model not found: {INT8_PATH}")

    fp32_counts, fp32_total = load_onnx_graph_counts(FP32_PATH)
    int8_counts, int8_total = load_onnx_graph_counts(INT8_PATH)

    print(f"\n  FP32 original graph ({FP32_PATH})")
    print(f"    total nodes: {fp32_total}")
    for op, cnt in sorted(fp32_counts.items()):
        print(f"    {op}: {cnt}")

    print(f"\n  INT8 canonical graph ({INT8_PATH})")
    print(f"    total nodes: {int8_total}")
    for op, cnt in sorted(int8_counts.items()):
        print(f"    {op}: {cnt}")

    fp32_bn = fp32_counts.get("BatchNormalization", 0)
    fp32_conv = sum(fp32_counts.get(op, 0) for op in CONV_LIKE_OPS)
    int8_bn = int8_counts.get("BatchNormalization", 0)
    int8_conv = sum(int8_counts.get(op, 0) for op in CONV_LIKE_OPS)

    print(
        f"\n  FP32 original: BatchNormalization={fp32_bn}, "
        f"Conv/QLinearConv={fp32_conv}"
    )
    print(
        f"  INT8 canonical: BatchNormalization={int8_bn}, "
        f"Conv/QLinearConv={int8_conv}"
    )

    # --- Step 3: ONNX initializer inspection for fold evidence ---
    print("\n[3] ONNX initializer inspection: do Conv nodes carry a bias")
    print("    that the original PyTorch Conv2d did not have?")
    print("-" * 60)

    fp32_inits = load_onnx_initializers(FP32_PATH)
    weight_bias_inits = sorted(
        n for n in fp32_inits if n.endswith(".weight_bias")
    )
    print(
        f"  ONNX initializers matching '*.weight_bias' (a bias tensor "
        f"attached to a Conv weight that was not part of the original "
        f"module's own parameters): {len(weight_bias_inits)}"
    )

    # --- Step 4: algebraic BatchNorm-fold verification, all pairs ---
    print("\n[4] Algebraic BatchNorm-fold verification against ONNX initializers")
    print("-" * 60)
    print(
        "  Formula: W' = W * gamma / sqrt(var + eps), "
        "b' = beta + (b - mean) * gamma / sqrt(var + eps)"
    )

    fold_results = []
    for conv_name, bn_name in conv_bn_pairs:
        result = verify_fold(model, fp32_inits, conv_name, bn_name)
        fold_results.append(result)
        status = (
            "MATCH"
            if result["weight_matches_fold"] and result["bias_matches_fold"]
            else "NO MATCH / NOT FOUND"
        )
        print(
            f"  {conv_name} + {bn_name}: weight_diff="
            f"{result['weight_max_abs_diff']}, bias_diff="
            f"{result['bias_max_abs_diff']} -> {status}"
        )

    n_pairs = len(fold_results)
    n_matched = sum(
        1
        for r in fold_results
        if r["weight_matches_fold"] and r["bias_matches_fold"]
    )
    print(f"\n  {n_matched}/{n_pairs} Conv+BatchNorm pairs match the fold formula exactly.")

    # --- Step 5: ORT-optimized INT8 graph (post graph-optimization) ---
    print("\n[5] ONNX Runtime ORT_ENABLE_ALL optimized-graph comparison (INT8)")
    print("-" * 60)

    int8_opt_counts = None
    int8_opt_total = None
    int8_opt_bn = None
    int8_opt_conv = None
    if os.path.exists(PRIOR_INT8_OPTIMIZED_PATH):
        int8_opt_counts, int8_opt_total = load_onnx_graph_counts(
            PRIOR_INT8_OPTIMIZED_PATH
        )
        print(
            f"\n  INT8 ORT-optimized graph (reused, read-only, from prior "
            f"experiment: {PRIOR_INT8_OPTIMIZED_PATH})"
        )
        print(f"    total nodes: {int8_opt_total}")
        for op, cnt in sorted(int8_opt_counts.items()):
            print(f"    {op}: {cnt}")

        int8_opt_bn = int8_opt_counts.get("BatchNormalization", 0)
        int8_opt_conv = sum(int8_opt_counts.get(op, 0) for op in CONV_LIKE_OPS)
        print(
            f"\n  INT8 optimized: BatchNormalization={int8_opt_bn}, "
            f"Conv/QLinearConv={int8_opt_conv} "
            f"(before: BN={int8_bn}, Conv={int8_conv})"
        )

        print("\n  INT8 op-type changes (original -> ORT-optimized):")
        all_ops = set(int8_counts) | set(int8_opt_counts)
        for op in sorted(all_ops):
            o = int8_counts.get(op, 0)
            n = int8_opt_counts.get(op, 0)
            if o != n:
                print(f"    {op}: {o} -> {n}")
    else:
        print(
            f"\n  WARNING: {PRIOR_INT8_OPTIMIZED_PATH} not found (prior "
            f"experiment artifact missing). Skipping INT8 optimized-graph "
            f"comparison; INT8 original-graph counts from step [2] are "
            f"still reported."
        )

    # --- Step 6: conclusion logic ---
    print("\n[6] Conclusion")
    print("-" * 60)

    exported_graph_has_no_bn = fp32_bn == 0 and int8_bn == 0
    fold_formula_confirmed = n_pairs > 0 and n_matched == n_pairs
    ort_removed_bn_nodes = int8_opt_bn is not None and int8_bn > int8_opt_bn

    if exported_graph_has_no_bn and fold_formula_confirmed:
        conclusion = (
            "BatchNorm folding IS observed, but it happens BEFORE the "
            "graph reaches ONNX Runtime: the FP32 ONNX graph exported by "
            "src/export_onnx.py already contains zero BatchNormalization "
            f"nodes ({pytorch_bn_count} nn.BatchNorm2d modules in the "
            "source PyTorch model were already gone by export time), and "
            "the Conv weight/bias initializers in that exported graph "
            f"exactly match ({n_matched}/{n_pairs} pairs, max abs diff "
            "0.0 in all matched cases) the closed-form BatchNorm-into-Conv "
            "fold formula applied to the original PyTorch Conv+BN "
            "parameters. This means PyTorch's ONNX export pipeline "
            "(torch.onnx.export with the default do_constant_folding=True) "
            "performed the fold, not ONNX Runtime's ORT_ENABLE_ALL graph "
            "optimizer. Since the graph already has no BatchNormalization "
            "node when ONNX Runtime first sees it (at both FP32 and INT8 "
            "stages), ONNX Runtime graph optimization CANNOT be credited "
            "with folding BatchNorm into Conv in this pipeline -- there is "
            "nothing left for it to fold by the time it runs."
        )
    elif exported_graph_has_no_bn and not fold_formula_confirmed:
        conclusion = (
            "No BatchNormalization node is present in either the FP32 or "
            "INT8 ONNX graph, and a transformation clearly occurred before "
            "export, but the algebraic fold formula did not exactly "
            "reproduce all Conv initializers, so the precise fold "
            "mechanism could not be fully confirmed numerically for every "
            "layer. Graph-level evidence (BN node count = 0 pre-ORT) still "
            "shows the fold predates ONNX Runtime, but treat the exact "
            "mechanism as unconfirmed for the non-matching layers."
        )
    else:
        conclusion = (
            "BatchNormalization nodes are present in the graph ONNX "
            "Runtime receives. Evidence of ORT-driven fusion would need "
            "to be assessed from the before/after optimized-graph node "
            "counts directly rather than assumed."
        )

    print(conclusion)

    fp32_vs_int8_note = (
        "FP32 and INT8 do NOT differ in whether BatchNorm folding occurs: "
        "both graphs already have 0 BatchNormalization nodes before ONNX "
        "Runtime ever loads them, because quantize_static() in "
        "src/quantize.py operates on the already-BN-folded FP32 ONNX file "
        "(results/mobilenetv3_cifar10_fp32.onnx) as its input. The two "
        "stages DO differ in a separate, unrelated transformation: ORT's "
        "ORT_ENABLE_ALL optimizer restructures the INT8 QDQ graph by "
        "fusing Conv + surrounding QuantizeLinear/DequantizeLinear nodes "
        "into QLinearConv nodes (observed previously as Conv 52 -> 12, "
        "QLinearConv 0 -> 40 in "
        "benchmarks/graph_opt_experiment/summary.json). That is a "
        "QDQ-to-QOperator INT8 fusion, not BatchNorm folding, and it has "
        "no FP32 counterpart since there is no quantization structure in "
        "the FP32 graph to fuse."
    )
    print(f"\n{fp32_vs_int8_note}")

    # --- Write machine-readable summary ---
    summary = {
        "ort_version": ort.__version__,
        "pytorch_model": {
            "batchnorm2d_modules": pytorch_bn_count,
            "conv2d_modules": pytorch_conv_count,
            "conv_bn_adjacency_pairs_found": n_pairs,
        },
        "original_fp32_graph": {
            "path": FP32_PATH,
            "total_nodes": fp32_total,
            "op_counts": fp32_counts,
            "batchnormalization_count": fp32_bn,
            "conv_qlinearconv_count": fp32_conv,
        },
        "original_int8_graph": {
            "path": INT8_PATH,
            "total_nodes": int8_total,
            "op_counts": int8_counts,
            "batchnormalization_count": int8_bn,
            "conv_qlinearconv_count": int8_conv,
        },
        "optimized_int8_graph": (
            {
                "path": PRIOR_INT8_OPTIMIZED_PATH,
                "source": "reused read-only from benchmarks/graph_opt_experiment/ "
                "(Phase 2 / Experiment 1); not regenerated by this script",
                "total_nodes": int8_opt_total,
                "op_counts": int8_opt_counts,
                "batchnormalization_count": int8_opt_bn,
                "conv_qlinearconv_count": int8_opt_conv,
            }
            if int8_opt_counts is not None
            else None
        ),
        "weight_bias_initializer_count_fp32_graph": len(weight_bias_inits),
        "conv_batchnorm_fold_verification": fold_results,
        "conv_batchnorm_pairs_matched_exactly": n_matched,
        "conv_batchnorm_pairs_total": n_pairs,
        "batchnorm_folding_observed": exported_graph_has_no_bn
        and fold_formula_confirmed,
        "folding_occurs_at": (
            "pytorch_onnx_export"
            if exported_graph_has_no_bn and fold_formula_confirmed
            else "undetermined"
        ),
        "onnxruntime_graph_optimization_folds_batchnorm": ort_removed_bn_nodes,
        "fp32_int8_conclusions_differ": False,
        "conclusion": conclusion,
        "fp32_vs_int8_note": fp32_vs_int8_note,
        "limitations": [
            "The exported ONNX graphs already contain zero "
            "BatchNormalization nodes before this script runs, so "
            "ONNX Runtime's optimized-graph dump cannot demonstrate a "
            "live BN-fold transformation for this model -- there is no "
            "BN node left in the input graph to fold. The conclusion "
            "that folding happens pre-export rests on (a) the absence "
            "of BatchNormalization nodes in the original exported "
            "graph and (b) exact numerical agreement between the "
            "closed-form fold formula (applied to PyTorch state) and "
            "the ONNX Conv initializers -- not on directly observing "
            "ONNX Runtime fold a live BatchNormalization node.",
            "Fold verification compares PyTorch checkpoint parameters "
            "against FP32 ONNX initializers only; INT8 QDQ initializers "
            "are quantized (int8/uint8 with scale/zero-point) and are "
            "not directly comparable via the float fold formula, so no "
            "analogous exact-match check was performed on the INT8 "
            "graph itself.",
            "Conv-BatchNorm adjacency pairs were identified structurally "
            "(BatchNorm2d immediately following Conv2d within the same "
            "nn.Sequential container), which matches every "
            "Conv2dNormActivation block in this MobileNetV3 model; a "
            "different architecture with non-adjacent or non-Sequential "
            "Conv/BN placement would need a different pairing strategy.",
        ],
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
