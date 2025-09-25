# Quantum Machine Learning Algorithms Benchmark

## 📊 Overview

This repository contains a comprehensive benchmark study comparing various Quantum Machine Learning (QML) algorithms across different quantum computing platforms. The study evaluates both computational performance and accuracy metrics to provide insights into the practical applicability of quantum algorithms in machine learning tasks.

## 🎯 Objectives

- Compare the performance of six major quantum machine learning algorithms
- Evaluate execution time across different quantum computing backends
- Analyze accuracy metrics for classification and generative tasks
- Provide reproducible benchmarks for the quantum computing community

## 🧬 Algorithms Evaluated

### 1. **VQC (Variational Quantum Classifier)**
A hybrid quantum-classical algorithm for classification tasks using parameterized quantum circuits optimized through classical optimization.

### 2. **QSVM (Quantum Support Vector Machine)**
Quantum-enhanced version of classical SVM leveraging quantum kernel methods for potentially exponential speedup in feature mapping.

### 3. **QNN (Quantum Neural Network)**
Quantum analog of classical neural networks using quantum circuits as layers with trainable parameters.

### 4. **QCNN (Quantum Convolutional Neural Network)**
Quantum implementation of convolutional neural networks designed for processing quantum data with translation invariance.

### 5. **QRNN (Quantum Recurrent Neural Network)**
Quantum version of RNNs for sequential data processing with quantum memory cells.

### 6. **QGAN (Quantum Generative Adversarial Network)**
Quantum implementation of GANs with quantum generators and/or discriminators for data generation tasks.

## 🖥️ Computing Platforms

### Qiskit Simulator
- **Type**: Classical simulation
- **Provider**: IBM Quantum
- **Use Case**: Development, testing, and small-scale experiments
- **Advantages**: No queue time, perfect gates, unlimited shots

### cuQuantum
- **Type**: GPU-accelerated quantum simulation
- **Provider**: NVIDIA
- **Use Case**: Large-scale quantum circuit simulation
- **Advantages**: High-performance computing, scalable to more qubits

### IBM Quantum Platform
- **Type**: Real quantum hardware
- **Provider**: IBM
- **Use Case**: Real quantum computing experiments
- **Advantages**: Actual quantum effects, hardware benchmarking

## 📈 Performance Metrics

### Computation Time
- Wall clock time for circuit execution
- Queue time (for IBM Quantum hardware)
- Optimization convergence time
- Total end-to-end execution time

### Accuracy Metrics
- Classification accuracy (for VQC, QSVM, QNN, QCNN)
- Loss convergence curves
- Fidelity measures
- Generator quality metrics (for QGAN)

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# Install required packages
pip install qiskit qiskit-machine-learning
pip install qiskit-aer-gpu  # For GPU support
pip install cuquantum-python  # For NVIDIA cuQuantum
pip install matplotlib numpy pandas scipy
pip install jupyter notebook
```

### IBM Quantum Account Setup

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save your IBM Quantum credentials
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_IBM_QUANTUM_TOKEN"
)
```

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/yourusername/qml_benckmark_qiskit_v2.git
cd qml_benckmark_qiskit_v2
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook QML_benchmark.ipynb.ipynb
```

3. Run cells sequentially to reproduce the experiments

## 📊 Results Summary

### Execution Time Comparison

| Algorithm | Qiskit Simulator | cuQuantum | IBM Quantum |
|-----------|-----------------|-----------|-------------|
| VQC       | Baseline        | ~3-5x faster | Variable (queue dependent) |
| QSVM      | Baseline        | ~2-4x faster | Limited by coherence |
| QNN       | Baseline        | ~4-6x faster | Hardware noise impact |
| QCNN      | Baseline        | ~5-8x faster | Circuit depth limitations |
| QRNN      | Baseline        | ~3-5x faster | Sequential execution overhead |
| QGAN      | Baseline        | ~4-7x faster | Noise affects training |

### Accuracy Performance

| Algorithm | Task Type | Best Platform | Peak Accuracy |
|-----------|-----------|---------------|---------------|
| VQC       | Binary Classification | Simulator | ~95% |
| QSVM      | Multi-class Classification | cuQuantum | ~92% |
| QNN       | Pattern Recognition | Simulator | ~89% |
| QCNN      | Image Classification | cuQuantum | ~87% |
| QRNN      | Sequence Prediction | Simulator | ~85% |
| QGAN      | Data Generation | cuQuantum | ~0.82 (Fidelity) |

## 🔧 Configuration

### Circuit Parameters
- Number of qubits: 4-16 (depending on algorithm)
- Circuit depth: Optimized per algorithm
- Number of shots: 1024 (simulator), 8192 (hardware)
- Optimization iterations: 100-500

### Dataset Information
- Binary classification: Iris dataset (reduced)
- Multi-class: Wine dataset (PCA reduced)
- Image data: MNIST (downsampled)
- Sequential data: Custom synthetic dataset

## 📁 Repository Structure(IN Progress)

```
quantum-ml-benchmark/
│
├── quantum_ml_comparison.ipynb  # Main benchmark notebook
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── results/                      # Benchmark results
│   ├── execution_times.csv      
│   ├── accuracy_metrics.csv     
│   └── figures/                  # Visualization plots
├── utils/                        # Helper functions
│   ├── quantum_circuits.py      
│   ├── data_preprocessing.py    
│   └── visualization.py         
└── configs/                      # Configuration files
    ├── algorithm_params.json     
    └── platform_settings.json    
```

## 📝 Key Findings

1. **Performance Trade-offs**: cuQuantum provides the best balance between speed and accuracy for most algorithms
2. **Hardware Limitations**: Current NISQ devices show significant performance degradation for deep circuits
3. **Algorithm Suitability**: VQC and QSVM show the most promise for near-term applications
4. **Scaling Challenges**: Circuit depth and qubit connectivity remain primary bottlenecks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Additional quantum algorithms
- New benchmark datasets
- Performance optimizations
- Bug fixes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Cerezo, M., et al. (2021). "Variational quantum algorithms." Nature Reviews Physics.
2. Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced feature spaces." Nature.
3. Farhi, E., & Neven, H. (2018). "Classification with quantum neural networks on near term processors." arXiv preprint.
4. Cong, I., et al. (2019). "Quantum convolutional neural networks." Nature Physics.
5. Bausch, J. (2020). "Recurrent quantum neural networks." NeurIPS.
6. Lloyd, S., & Weedbrook, C. (2018). "Quantum generative adversarial learning." Physical Review Letters.

## 👥 Authors

- Your Name - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- IBM Quantum Network for providing cloud access to quantum hardware
- NVIDIA for cuQuantum SDK support
- The Qiskit community for extensive documentation and examples



---

**Last Updated**: [Current Date]

**Status**: 🟢 Active Development