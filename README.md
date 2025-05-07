# MM-Neural-Adjoint

A Python package implementing neural adjoint methods, specifically designed for predicting the geometries of metamaterials. This implementation is based on the work from [BDIMNNA (Benchmarking Deep Inverse Models over time, and the Neural-Adjoint method)](https://github.com/BensonRen/BDIMNNA), published in NeurIPS 2020 by Simiao Ren, Willie J. Padilla and Jordan Malof.

## About

This package focuses on the Neural Adjoint (NA) method for inverse design of metamaterials. It provides a streamlined implementation specifically optimized for metamaterial geometry prediction tasks, building upon the benchmarking work done in the original BDIMNNA repository.

## Installation

This package supports different hardware configurations including CPU, Apple Silicon (M1/M2), and NVIDIA GPUs. Choose the appropriate installation method based on your hardware:

### Basic Installation (CPU/Apple Silicon)
```bash
pip install MM-neural-adjoint
```

## Usage

### Basic Usage

Here's a simple example of how to use the package:

```python
from mm_neural_adjoint import NANetwork, ConvModel, LinModel
import torch
from torch.utils.data import DataLoader, TensorDataset

# Initialize the model
geometry_size = 8  # Input size (geometry parameters)
spectrum_size = 300  # Output size (spectrum parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# This can be any well defined torch model.
model = LinModel(geometry_size, spectrum_size)

model = NANetwork(
    model=model
    device=device
)

model.train(
    epochs=100,
    train_loader=train_loader,
    val_loader=val_loader
)
```

### MLflow Integration

The package uses MLflow for experiment tracking and model management. By default, MLflow data is stored in a local SQLite database (`mlflow.db`). Here's how to use MLflow features:

```python
import mlflow
from mm_neural_adjoint import NANetwork

# Set a custom experiment name
model = NANetwork(
    model=model,
    device=device,
    mlflow_exp_name="my_experiment"  # Custom experiment name
)

# Training will automatically log:
# - Training loss
# - Validation loss
# - Best validation loss
# - Total training time
# - Model parameters
model.train(
    epochs=100,
    train_loader=train_loader,
    val_loader=val_loader
)

# View MLflow UI
# Run this command in your terminal:
# mlflow ui
```

#### MLflow Features

1. **Automatic Logging**:
   - Training metrics (loss, validation loss)
   - Model parameters
   - Training duration
   - Model checkpoints

2. **Experiment Organization**:
   - Each training run is tracked as a separate experiment
   - Experiments are named using timestamps by default
   - Custom experiment names can be set during model initialization

3. **Model Checkpoints**:
   - Best models are automatically saved
   - Checkpoints are stored in the `checkpoints/` directory
   - MLflow tracks the relationship between metrics and checkpoints

4. **Viewing Results**:
   ```bash
   # Start MLflow UI
   mlflow ui
   
   # Access the UI at http://localhost:5000
   ```

5. **Custom Logging**:
   ```python
   # Add custom metrics during training
   with mlflow.start_run():
       mlflow.log_param("learning_rate", 0.001)
       mlflow.log_metric("custom_metric", value)
   ```

## Requirements

- Python >= 3.10
- PyTorch >= 2.6.0
- NumPy >= 2.2.4
- Pandas >= 2.2.3
- tqdm >= 4.67.1
- MLflow >= 2.21.3
- scikit-learn >= 1.4.0

## Examples

The package includes several example notebooks in the `examples/` directory:

- `examples/example1.ipynb`: Basic usage and training
- `examples/example2.ipynb`: Advanced features and customization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This package is based on the Neural Adjoint implementation from the [BDIMNNA repository](https://github.com/BensonRen/BDIMNNA) by Benson Ren et al. We thank the original authors for their foundational work in developing and benchmarking the Neural Adjoint method.


