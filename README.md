# Quantum Machine Learning Benchmark Suite

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-purple.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive benchmarking framework for comparing various Quantum Machine Learning (QML) algorithms across different quantum computing platforms and backends.

## 📊 Overview

This notebook provides an extensive comparison of six quantum machine learning algorithms, evaluating their performance in terms of computation time and accuracy across multiple quantum computing backends. The implementation includes parallel execution support, GPU acceleration, and noise-resilient optimizers.

## 🚀 Features

### Quantum Algorithms Implemented
- **VQC (Variational Quantum Classifier)** - Hybrid quantum-classical classifier with parallel batch execution
- **QSVM (Quantum Support Vector Machine)** - Quantum kernel-based SVM with parallel kernel computation
- **QNN (Quantum Neural Network)** - Quantum neural network with SPSA optimizer for noise resilience
- **QCNN (Quantum Convolutional Neural Network)** - Fixed-qubit quantum CNN implementation
- **QRNN (Quantum Recurrent Neural Network)** - Fixed-qubit quantum RNN for sequential data
- **QGAN (Quantum Generative Adversarial Network)** - Quantum GAN for generative tasks

### Supported Backends
- **Qiskit Simulator** - CPU-based quantum circuit simulation
- **cuQuantum** - GPU-accelerated quantum simulation using NVIDIA cuQuantum
- **IBM Quantum Platform** - Real quantum hardware execution (optional, requires API token)

### Key Features
- ✅ **Parallel Execution** - Efficient batch processing with multiprocessing support
- ✅ **GPU Acceleration** - CUDA support via cuQuantum for faster simulations
- ✅ **Training Progress Monitoring** - Real-time training callbacks and progress tracking
- ✅ **SPSA Optimizer** - Noise-resilient optimization for quantum hardware
- ✅ **Fixed Qubit Implementation** - Optimized quantum circuits with controlled qubit usage
- ✅ **Comprehensive Visualization** - Performance metrics and comparison charts

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for cuQuantum backend)
- IBM Quantum account (optional, for hardware execution)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/qml_benckmark_qiskit_v2.git
cd qml_benckmark_qiskit_v2
```

2. **Enable GPU Runtime (for Google Colab)**
   - Go to Runtime → Change runtime type
   - Select GPU as Hardware accelerator (T4 recommended)
   - Click Save

3. **Install dependencies**
```bash
pip install -q qiskit qiskit-machine-learning qiskit-ibm-runtime qiskit-aer
pip install -q qiskit-aer-gpu  # For GPU support
pip install -q torch torchvision scikit-learn matplotlib pylatexenc
```

## 📝 Usage

### Basic Usage

Open and run the `QML_benchmark.ipynb` notebook in Jupyter or Google Colab:

```python
jupyter notebook QML_benchmark.ipynb
```

### Configuration

The notebook includes a configuration section where you can customize:

```python
CONFIG = {
    'n_qubits': 4,           # Number of qubits
    'n_samples': 100,        # Dataset size
    'max_iterations': 20,    # Training iterations
    'batch_size': 10,        # Batch size for parallel execution
    'use_gpu': True,         # Enable GPU acceleration
    'ibm_token': '',         # IBM Quantum API token (optional)
}
```

### Running Specific Models

You can run individual models or the complete benchmark suite:

```python
# Initialize the quantum models
qm = QuantumModels(backend_type='gpu', config=CONFIG)

# Run individual model
results_vqc = qm.train_vqc()

# Run all models
all_results = qm.run_all_models()
```

## 📈 Benchmark Results

The notebook generates comprehensive performance metrics including:

- **Accuracy Comparison** - Classification accuracy across all models and backends
- **Runtime Analysis** - Training and inference time measurements
- **Resource Utilization** - Memory usage and computational requirements
- **Convergence Plots** - Training loss and accuracy evolution


## 🏗️ Architecture

### Notebook Structure

1. **Environment Setup** - GPU detection and system information
2. **Dependencies Installation** - Required packages and libraries
3. **Library Imports** - Quantum and classical ML libraries
4. **Configuration** - Hyperparameters and settings
5. **Data Preparation** - Dataset loading and preprocessing
6. **Backend Setup** - Initialize quantum simulators and hardware
7. **Model Implementations** - Quantum ML algorithm classes
8. **Benchmark Execution** - Run models across all backends
9. **Results Visualization** - Performance plots and comparisons
10. **Summary & Conclusions** - Key findings and recommendations

### Key Components

- **Parallel Execution Engine** - Multiprocessing-based batch processing
- **Backend Manager** - Handles simulator/hardware initialization
- **Progress Callbacks** - Real-time training monitoring
- **Performance Metrics** - Accuracy, time, and resource tracking

## 🔬 Technical Details

### Quantum Circuit Design
- **Encoding**: Angle encoding for classical data representation
- **Ansatz**: Hardware-efficient ansatz with entangling layers
- **Measurements**: Pauli-Z basis measurements
- **Optimization**: Gradient-free SPSA optimizer

### Performance Optimizations
- Batch processing for parallel quantum circuit execution
- GPU acceleration via cuQuantum for large-scale simulations
- Circuit optimization and transpilation for hardware execution
- Memory-efficient data handling for large datasets

## 📊 Visualization Examples

The notebook generates various visualizations:
- Model accuracy comparison bar charts
- Runtime performance heatmaps
- Training convergence plots
- Backend performance comparisons

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- New quantum ML algorithms
- Additional backend support
- Performance optimizations
- Bug fixes and improvements

## 📚 References

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Qiskit Machine Learning](https://qiskit.org/ecosystem/machine-learning/)
- [cuQuantum SDK](https://developer.nvidia.com/cuquantum-sdk)
- [IBM Quantum Network](https://quantum-computing.ibm.com/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✨ Acknowledgments

- IBM Quantum team for Qiskit framework
- NVIDIA for cuQuantum GPU acceleration
- The quantum computing community for continuous support

