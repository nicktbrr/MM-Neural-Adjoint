# MM-Neural-Adjoint

A Python package implementing neural adjoint methods, specifically designed for predicting the geometries of metamaterials. This implementation is based on the work from [BDIMNNA (Benchmarking Deep Inverse Models over time, and the Neural-Adjoint method)](https://github.com/BensonRen/BDIMNNA), published in NeurIPS 2020 by Simiao Ren, Willie J. Padilla and Jordan Malof.

## About

This package focuses on the Neural Adjoint (NA) method for inverse design of metamaterials. It provides a streamlined implementation specifically optimized for metamaterial geometry prediction tasks, building upon the benchmarking work done in the original BDIMNNA repository.

## Installation

This package supports different hardware configurations including CPU, Apple Silicon (M1/M2), and NVIDIA GPUs. Choose the appropriate installation method based on your hardware:

### Basic Installation (CPU/Apple Silicon)
```bash
pip install -e .
```

### For NVIDIA GPU Users
If you have an NVIDIA GPU, you can install with CUDA support:

```bash
# For CUDA 11.8
pip install -e .[cuda]

# For CUDA 12.1
pip install -e .[cuda12]
```

### Development Installation
For contributors and developers:
```bash
# Clone the repository
git clone https://github.com/yourusername/MM-Neural-Adjoint.git
cd MM-Neural-Adjoint

# Install in editable mode with all dependencies
pip install -e .
```

## Hardware Support

- **CPU**: Supported on all platforms
- **Apple Silicon (M1/M2)**: GPU acceleration via MPS backend (included in default installation)
- **NVIDIA GPUs**: CUDA support available through specific installation options

## Usage

After installation, you can import and use the package in your Python code:

```python
import mm_neural_adjoint
```

## Examples

The package includes several example notebooks and scripts to help you get started with metamaterial design using the Neural Adjoint method. You can find these in the `examples/` directory:

- `examples/example1.ipynb`: Introduction to basic package functionality

To run the examples:
```bash
cd examples
jupyter notebook

## Requirements

- Python >= 3.7
- PyTorch >= 2.6.0
- NumPy >= 2.2.4
- Pandas >= 2.2.3
- tqdm >= 4.67.1
- MLflow >= 2.21.3
- scikit-learn >= 1.4.0

## Acknowledgments

This package is based on the Neural Adjoint implementation from the [BDIMNNA repository](https://github.com/BensonRen/BDIMNNA) by Benson Ren et al. We thank the original authors for their foundational work in developing and benchmarking the Neural Adjoint method.