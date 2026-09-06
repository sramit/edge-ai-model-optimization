# edge-ai-model-optimization
Practical experiments in computer vision, model optimization, quantization, and edge AI deployment.

## INT8 Quantization Results

Model: MobileNetV3-Small, trained on CIFAR-10. Quantized post-training with ONNX Runtime 1.29.0
(`onnxruntime.quantization.quantize_static`), evaluated on the CIFAR-10 test set, benchmarked with
`onnxruntime.CPUExecutionProvider` on a fixed input.

All latency/throughput numbers below are **CPUExecutionProvider-specific and inference-only**
(single fixed input, no data loading/preprocessing/postprocessing, no other execution provider
tested). They do not necessarily generalize to GPU, NPU, or other CPU/thread configurations.

### Baselines (measured)

| | FP32 | INT8 QDQ, MinMax calibration (500 samples) |
|---|---|---|
| Accuracy | 73.56% | 72.97% (drop: 0.59 pp) |
| Model size | 6.12 MB | 1.95 MB |
| Median latency | ~1.02–1.05 ms | 1.199 ms |

The FP32 model and CIFAR-10 test set were held fixed across every experiment below. Each
experiment changed exactly one quantization parameter at a time from this starting configuration
(QDQ format, QInt8 activations, QInt8 weights, per-channel weights, MinMax calibration).

### Experiment 1 — QDQ vs QOperator format (measured)

| | QDQ | QOperator |
|---|---|---|
| Accuracy | 72.97% | 73.33% |
| Model size | 1.95 MB | 1.69 MB |
| Median latency | 1.199 ms | 2.239 ms |

**Interpretation:** QOperator produces a smaller, more fused graph (178 vs 432 nodes) but was
roughly 2x slower here; both formats were slower than FP32 (~1.04–1.05 ms) on this CPU. ONNX
Runtime itself warns that QDQ is recommended for QInt8 activations + QInt8 weights on x64. QDQ was
selected and kept for the remaining experiments.

### Experiment 2 — Calibration sample count (measured)

| Calibration samples | Accuracy | Accuracy drop vs FP32 |
|---|---|---|
| 500 | 72.97% | 0.59 pp |
| 1,000 | 73.31% | 0.25 pp |
| 5,000 | 72.80% | 0.76 pp |

**Interpretation:** accuracy did not improve monotonically with more calibration data; 1,000
samples gave the best result of the three and was selected. No causal explanation for the
non-monotonicity is claimed.

### Experiment 3 — Per-channel vs per-tensor weight quantization (measured, 1,000 samples, MinMax)

| | Per-channel | Per-tensor |
|---|---|---|
| Accuracy | 73.31% | 73.10% |
| Accuracy drop vs FP32 | 0.25 pp | 0.46 pp |
| Model size | 1.95 MB | 1.82 MB |
| Median latency | 1.129 ms | 1.114 ms |
| Throughput | 885.63 FPS | 897.88 FPS |

**Interpretation:** per-channel weight quantization gave a real accuracy benefit (about half the
accuracy drop of per-tensor) for a small size cost and no consistent latency penalty. Per-channel
was selected.

### Experiment 4 — Activation calibration method (measured, 1,000 samples, per-channel, QDQ)

| Calibration method | Accuracy | Accuracy drop vs FP32 |
|---|---|---|
| MinMax (baseline) | 73.31% | 0.25 pp |
| Entropy | 73.31% | 0.25 pp |
| Percentile | 73.52% | 0.04 pp |

**Measured, not interpretation:** the Entropy-calibrated model was found to be **byte-identical**
(same SHA-256) to the MinMax model — every one of the 107 scalar activation quantization scales in
the graph matched MinMax exactly. Entropy calibration made no measurable difference in this
configuration.

**Interpretation:** a plausible explanation is that MobileNetV3's ReLU6/hardswish activations are
already range-bounded, so the entropy (KL-divergence) threshold and the raw min/max threshold
coincide for this model — but this mechanism was not independently verified beyond the scale
comparison, so it is offered as a hypothesis, not a proven cause.

Percentile calibration (ONNX Runtime defaults: 2048 histogram bins, 99.999th percentile,
symmetric=True) changed all 107 activation scales (mean ~16% relative shift vs MinMax) and reduced the
accuracy drop from FP32 to 0.04 pp — an ~84% reduction in the gap versus MinMax. This result was
**independently re-evaluated in a second run against the same artifact and reproduced exactly at
73.52%** (identical accuracy, since the test set and model weights are both fixed/deterministic).
Percentile was selected as the final calibration method.

### Final selected configuration

- Format: QDQ
- Activations: QInt8
- Weights: QInt8, per-channel
- Calibration: Percentile (2048 histogram bins, 99.999th percentile, symmetric=True — ONNX Runtime defaults)
- Calibration samples: 1,000
- ONNX Runtime: 1.29.0

**Measured final results:**

| | FP32 | INT8 (final config) |
|---|---|---|
| Accuracy | 73.56% | 73.52% (drop: 0.04 pp) |
| Model size | 6.12 MB | 1.95 MB (−68.14%) |
| Median latency | ~1.01–1.05 ms | ~1.14–1.17 ms |
| Throughput | ~975–986 FPS | ~858–880 FPS |

### Important finding: INT8 did not improve CPU latency in this environment

**Measured:** across every experiment above, the INT8 QDQ models were consistently **slower**
than FP32 on `CPUExecutionProvider` (speedup < 1x in every measured configuration), despite the
~68% model-size reduction. INT8 provides **no latency or throughput improvement** in this
environment — model-size reduction and latency moved in opposite directions.

**Interpretation:** this is the opposite of the commonly assumed INT8-on-CPU speedup, and is
specific to this model, this quantization format, and this execution provider/hardware — it should
not be generalized to other models or environments without separate measurement.

**Measured:** operator-level profiling of the INT8 graph (see `benchmarks/profiling_results/`)
showed that `DequantizeLinear` and `QuantizeLinear` operators together account for roughly 45% of
the instrumented per-operator execution time, and the INT8 graph has substantially more nodes (432
vs FP32's node count) due to the QDQ pattern surrounding each quantized op.

**Interpretation:** this profiling is evidence of substantial Q/DQ- and graph-structure-related
overhead — it is **not proof** that Q/DQ ops are definitively the cause of the end-to-end latency
regression. The ONNX Runtime profiler adds its own instrumentation overhead, and the profiled run
is a separate measurement from the wall-clock benchmark above, so the two cannot be directly
equated.

**Takeaway:** for this model/hardware/execution-provider combination, INT8 quantization is a clear
win for model size and a near-zero-cost win for accuracy (0.04 pp drop with Percentile
calibration), but it is **not** a latency or throughput optimization — deploying INT8 here trades
size for a measured latency regression, not an improvement.

## Phase 2 — Deeper Optimization Analysis

Phase 2 investigates the behavior of the selected INT8 QDQ configuration beyond
the original accuracy, size, and latency sweep. Each experiment is implemented
as standalone investigative tooling under `benchmarks/`.

### Phase 2 Experiment 1 — ONNX Runtime graph optimization (measured)

Explicitly setting ONNX Runtime's `graph_optimization_level` to
`ORT_ENABLE_ALL` produced no meaningful latency difference compared with the
existing benchmark configuration.

**Interpretation:** both configurations already used ONNX Runtime's effective
default graph optimization level, so explicitly setting `ORT_ENABLE_ALL` did
not provide an additional optimization benefit.

### Phase 2 Experiment 2 — BatchNorm folding verification (measured)

BatchNorm folding was independently verified by comparing the algebraically
folded Conv parameters against the Conv initializers exported to ONNX.

**Result:** all **34/34 Conv + BatchNorm folds** matched the exported ONNX Conv
initializers with a maximum numerical difference of **0.0**.

**Interpretation:** the BatchNorm folding performed during model/export
optimization is numerically consistent with the exported ONNX graph for all
verified Conv + BatchNorm pairs.

### Phase 2 Experiment 3 — End-to-end quantization error analysis (measured)

The selected FP32 and INT8 models were evaluated on the same **10,000 CIFAR-10
test samples**. Final output tensors were compared directly to quantify the
numerical effect of INT8 quantization.

| Metric | Result |
|---|---:|
| Mean absolute error (MAE) | 0.18883 |
| RMSE | 0.24208 |
| P95 absolute error | 0.48289 |
| P99 absolute error | 0.67298 |
| Maximum absolute error | 1.42308 |
| Cosine similarity | 0.99671 |
| Prediction agreement | 95.69% |
| FP32 accuracy | 73.56% |
| INT8 accuracy | 73.52% |
| Accuracy drop | 0.04 pp |

**Interpretation:** INT8 introduces measurable numerical differences at the
model output, but the overall output vectors remain highly similar to FP32
(cosine similarity **0.99671**). The P95 and P99 absolute errors were **0.48289**
and **0.67298**, respectively.

Prediction agreement was **95.69%**, while aggregate accuracy changed by only
**0.04 percentage points** (73.56% → 73.52%). This demonstrates that the
observed quantization error has negligible impact on aggregate classification
accuracy for this model and calibration configuration.

Mean relative error is reported for completeness but is not treated as the
primary quality indicator because relative error can become unstable when the
reference FP32 logits are close to zero.

The experiment is implemented in
`benchmarks/quantization_error_experiment.py`, with machine-readable results in
`benchmarks/quantization_error_results/quantization_error_summary.json`.

