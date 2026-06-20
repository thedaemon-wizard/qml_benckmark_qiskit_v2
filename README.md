<table>
	<thead>
    	<tr>
      		<th style="text-align:center">English</th>
      		<th style="text-align:center"><a href="README_ja.md">日本語</a></th>
    	</tr>
  	</thead>
</table>

# Quantum Machine Learning Benchmark Suite (Qiskit v2.x)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.x-purple.svg)](https://www.ibm.com/quantum/qiskit)
[![Qiskit ML](https://img.shields.io/badge/qiskit--machine--learning-0.9-purple.svg)](https://qiskit-community.github.io/qiskit-machine-learning/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A benchmarking framework that compares six Quantum Machine Learning (QML) algorithms across
**CPU**, **GPU (cuStateVec)** and **real IBM Quantum hardware**, fully migrated to the latest
**Qiskit v2.x** stack and runnable on a **local Python 3.12 venv** (no Google Colab required).

## 📊 Overview

The `QML_benchmark.ipynb` notebook trains and evaluates six quantum models on the same `ad_hoc_data`
dataset and compares accuracy and runtime across backends. It uses Qiskit-2.x **V2 primitives**,
the **functional circuit-library builders**, structured **logging**, and writes every artifact to a
fresh `results/` directory on each run.

### Quantum algorithms
- **VQC** – Variational Quantum Classifier (per-batch SPSA training)
- **QSVM** – Quantum Support Vector Machine (fidelity quantum kernel)
- **QNN** – Quantum Neural Network (`EstimatorQNN` + SPSA)
- **QCNN** – Quantum Convolutional Neural Network
- **QRNN** – Quantum Recurrent Neural Network
- **QGAN** – Quantum Generative Adversarial Network (quantum generator + classical discriminator via `TorchConnector`); scored by a comparable `1 − TVD` generation-quality metric

### Backends
- **CPU** – Aer `statevector` simulator (multi-threaded)
- **GPU** – Aer `statevector` on NVIDIA GPU via **cuStateVec** (`qiskit-aer-gpu-cu11`)
- **IBM Quantum** – real hardware through the **`ibm_quantum_platform`** channel (optional)

## 🔄 What changed in the Qiskit v2.x migration

This notebook was originally written for **Qiskit 1.x on Google Colab**. The following APIs were
removed or deprecated in Qiskit 2.x and have been migrated:

| Old (Qiskit 1.x) | New (Qiskit 2.x, used here) |
|---|---|
| `from qiskit.primitives import Sampler, Estimator` (V1) | **removed** – use V2 primitives |
| `qiskit_aer.primitives.Sampler/Estimator` (V1) | `SamplerV2`, `EstimatorV2` (`options={'backend_options': {...}}`) |
| `ZZFeatureMap`, `ZFeatureMap`, `RealAmplitudes`, `EfficientSU2` *(classes)* | `zz_feature_map`, `z_feature_map`, `real_amplitudes`, `efficient_su2` *(functions)* |
| `FidelityQuantumKernel(..., pass_manager=...)` | `ComputeUncompute(sampler=..., pass_manager=...)` passed as `fidelity=` |
| `channel='ibm_quantum'` | `channel='ibm_quantum_platform'` (+ CRN `instance`) |
| Colab `!pip install` / `display()` | local `.venv`, `logging`, `results/` artifacts |

## 🖥️ Verified environment

Tested end-to-end on:

- **OS**: AlmaLinux 9.7 · **CPU**: Intel Core i5-13600K (14 cores) · **RAM**: 128 GB DDR5
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (96 GB, `sm_120`, driver 580.x) · **CUDA**: 13.0
- **Python**: 3.12 in a `.venv` virtual environment

| Package | Version |
|---|---|
| qiskit | 2.4.2 |
| qiskit-machine-learning | 0.9.0 |
| qiskit-aer / qiskit-aer-gpu-cu11 | 0.17.2 |
| qiskit-ibm-runtime | 0.47.0 |
| torch | 2.11.0+cu128 |
| scikit-learn / matplotlib / pandas / numpy | 1.9.0 / 3.11.0 / 3.0.3 / 2.4.6 |

> **Note on the GPU build.** The latest published `qiskit-aer-gpu-cu11==0.17.2` matches the CPU
> `qiskit-aer` 0.17.2 and is compatible with Qiskit 2.x. Although it bundles CUDA 11.8 / cuStateVec
> 1.6.0 (which predate Blackwell), it runs correctly on the `sm_120` card through the NVIDIA 580
> driver — `AerSimulator().available_devices()` reports `('CPU', 'GPU')` and GPU simulation returns
> correct results.

## 🛠️ Installation (local Python 3.12 venv)

```bash
git clone https://github.com/thedaemon-wizard/qml_benckmark_qiskit_v2.git
cd qml_benckmark_qiskit_v2

# Create and activate a Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Core stack (latest Qiskit v2.x)
pip install -U qiskit qiskit-machine-learning qiskit-ibm-runtime \
               scikit-learn matplotlib pylatexenc pandas ipykernel nbconvert jupyter

# GPU (NVIDIA) – Aer GPU build (same 0.17.x as the CPU qiskit-aer)
pip install "qiskit-aer-gpu-cu11==0.17.2"

# PyTorch with Blackwell / sm_120 support
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

For **IBM Quantum hardware**, place an `apikey.json` in the repository root (one level above the
notebook):

```json
{ "apikey": "<IBM_QUANTUM_API_KEY>", "crn": "<IBM_QUANTUM_PLATFORM_CRN>" }
```

The notebook loads `apikey` as the token and `crn` as the `instance` on the `ibm_quantum_platform`
channel. If `apikey.json` is absent, the IBM backend is simply skipped.

## 📝 Usage

Run interactively in Jupyter, or headless via `nbconvert`:

```bash
# Interactive
.venv/bin/jupyter notebook QML_benchmark.ipynb

# Headless (executes in place and saves outputs)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace QML_benchmark.ipynb \
  --ExecutePreprocessor.timeout=7200
```

### Selecting backends

By default the notebook auto-detects: `cpu` (always) + `gpu` (if an Aer GPU device is present) +
`ibm_quantum_platform` (if `apikey.json` is found). Override with the `QML_BACKENDS` environment
variable:

```bash
QML_BACKENDS=cpu                          # CPU only (fast)
QML_BACKENDS=cpu,gpu                      # CPU + GPU
QML_BACKENDS=cpu,gpu,ibm_quantum_platform # + real IBM hardware
```

### Outputs (`results/`) and logging

Every run **recreates** a `results/` directory and uses Python `logging` (console + file) instead of
bare `print`. Artifacts:

```
results/
├── benchmark.log                       # full structured run log
├── dataset_distribution.png            # dataset scatter
├── vqc_circuit_*.png, qcnn_circuit_*.png, qrnn_circuit_*.png, qgan_training_*.png
├── quantum_ml_comparison_detailed.png  # accuracy / time / loss / overview
├── quantum_ml_results.json             # machine-readable results
└── summary.csv, summary.txt            # results table
```

### Configuration

The `CONFIG` dictionary (cell *4. Configuration*) controls the run:

```python
CONFIG = {
    'feature_dim': 2,
    'num_qubits': 2,            # fixed qubit count for all models
    'training_size': 80,
    'test_size': 20,
    'batch_size': 20,
    'max_iterations': 6,        # small for a fast demo; raise for real training
    'ibm_token':   <loaded from apikey.json>,
    'ibm_instance':<loaded from apikey.json (CRN)>,
    'ibm_channel': 'ibm_quantum_platform',
    'use_gpu': torch.cuda.is_available(),
    'num_parallel_jobs': 4,
    # --- IBM hardware-only knobs (keep the real-hardware pass fast/affordable) ---
    'ibm_models': ['VQC', 'QSVM'],   # which models actually run on hardware
    'ibm_max_iterations': 1,         # fixed-step SPSA, no auto-calibration
    'ibm_train_subset': 8,           # training samples for variational HW models
    'ibm_qsvm_subset': 8,            # training samples for the QSVM kernel on HW
}
```

## 📈 Benchmark results (CPU vs GPU, this environment)

Representative run on the verified environment (`max_iterations=6`, 80 train / 20 test).
Accuracies are low for QNN/QCNN/QRNN by design — only 3–6 SPSA iterations are used so the demo runs
in seconds.

| Model | CPU acc | CPU time (s) | GPU acc | GPU time (s) |
|------|:------:|:----:|:------:|:----:|
| VQC  | 0.50 | 11.3 | 0.83 | 18.1 |
| QSVM | 0.85 | 9.4  | 0.88 | 23.7 |
| QNN  | 0.45 | 0.7  | 0.28 | 2.7  |
| QCNN | 0.23 | 0.4  | 0.28 | 2.0  |
| QRNN | 0.28 | 0.5  | 0.25 | 1.8  |
| QGAN | 0.79† | 19.2 | 0.74† | 27.5 |

† QGAN is generative, so it has no classification accuracy. The value is its **generation-quality
score** = `1 − TVD` (total-variation distance between the generated and a fixed target
distribution), in `[0,1]`, higher = better — directly comparable to the classifiers' accuracy.
`tvd` / `kl_divergence` / `final_loss` are also recorded in `quantum_ml_results.json`.
(Accuracies for QNN/QCNN/QRNN are low by design — only 3–6 SPSA iterations.)

> **GPU is slower than CPU at this scale.** All models use only **2–4 qubits**, where GPU
> kernel-launch overhead dominates (GPU speedups here are 0.2–0.7×). The GPU only wins for large
> state vectors: an independent 24-qubit `efficient_su2` circuit (statevector, 256 shots, 5-run
> median after warm-up, measured on this machine) runs in **~0.04 s on the GPU vs ~0.84 s on the
> CPU — ~21× faster**. (The very first GPU call is ~0.3 s because of one-time cuStateVec/CUDA
> initialization, which is why the tiny per-call circuits in the benchmark above don't benefit.)
> This benchmark is intentionally small; increase `num_qubits` to see the GPU advantage.

### IBM Quantum real hardware

Verified end-to-end on real IBM hardware via the `ibm_quantum_platform` channel (backends seen:
`ibm_fez`, `ibm_marrakesh`, `ibm_kingston`):

| Model | Backend | HW acc | HW time (s) |
|------|---------|:------:|:----:|
| VQC  | ibm_fez | 0.60 | 57.9 |
| QSVM | ibm_fez | 0.675 | 42.9 |

**Why only two models, and reduced settings.** On real hardware every SPSA evaluation is a
separate queued job. The credentials here are an **IBM Open plan**, which **does not allow
Sessions**, so jobs cannot run back-to-back and each one carries minutes of queue/turnaround.
Running the *full* SPSA benchmark unchanged took **~1 hour for VQC alone** and would consume the
**3-hour QPU quota**. To keep the hardware pass a fast, faithful end-to-end proof, `CONFIG`
exposes hardware-only knobs (see *Configuration*): it runs the two representative models in
`ibm_models` (`VQC` = variational, `QSVM` = kernel), fixes the SPSA step sizes to skip its
~25-evaluation auto-calibration (`ibm_max_iterations=1`), and trains on a small stratified subset
(`ibm_train_subset` / `ibm_qsvm_subset = 8`). Both models then complete on hardware in **~10
minutes total**. To run more models or larger problems on hardware, edit those `CONFIG` keys (and,
on a paid plan, the code automatically uses a `Session` for much faster turnaround).

## 🏗️ Notebook structure

1. Environment & system check (local workstation)
2. Installation reference + version check
3. Imports, `results/` setup, logging
4. Configuration (+ IBM credential loading)
5. Data preparation (`ad_hoc_data`) and visualization
6. Backend factories (CPU / GPU / IBM) + IBM account/connectivity
7. Training-progress callback
8. Model implementations (`QuantumModels` class; VQC, QSVM, QNN, QCNN, QRNN, QGAN)
9. Run all models across the selected backends
10. Results visualization and summary

## 📚 References

- [Qiskit Documentation](https://www.ibm.com/quantum/qiskit)
- [Qiskit Machine Learning](https://qiskit-community.github.io/qiskit-machine-learning/)
- [Qiskit Aer](https://qiskit.github.io/qiskit-aer/)
- [IBM Quantum Platform](https://quantum.cloud.ibm.com/)
- [NVIDIA cuQuantum / cuStateVec](https://developer.nvidia.com/cuquantum-sdk)

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## ✨ Acknowledgments

- IBM Quantum team for the Qiskit framework
- NVIDIA for cuQuantum / cuStateVec GPU acceleration
- The quantum computing community
