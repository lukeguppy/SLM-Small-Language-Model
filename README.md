# Small Language Model (SLM)

A transformer-based language model with a graphical user interface for training, inference, and real-time exploration of attention mechanisms and embeddings. Features a clean, maintainable codebase with proper separation of concerns and comprehensive testing.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lukeguppy/SLM-Small-Language-Model.git
   cd SLM-Small-Language-Model
   ```

2. **Windows**:

   Create and activate a virtual environment (Python 3.12 recommended):

   ```bash
   py -3.12 -m venv venv
   venv\Scripts\activate
   ```

   Run the setup script (follow on-screen instructions):

   ```bash
   setup.bat
   ```

3. **Linux/Mac**:

   Create and activate a virtual environment (Python 3.12 recommended):

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

   Run the setup script (auto-detects CUDA and installs dependencies):

   ```bash
   ./setup.sh
   ```

> **Note:** Two sets of requirements are provided:
>
> * `requirements-cpu.txt` – CPU-only
> * `requirements-cuda.txt` – GPU acceleration (requires CUDA; optional)

## Usage

Activate your virtual environment and run:

```bash
python -m src
```

The GUI provides three main tabs:

* **Input Tab**: Text input with tokenisation and next-token prediction
* **Training Tab**: Model training with real-time loss curves and metrics
* **Visualisation Tab**: Attention mechanism exploration and embedding visualisation

### Controls

* **Tab**: Accept the predicted next word in the Input Tab

## Demo

[View demo assets](https://github.com/user-attachments/assets/31e13160-d339-4833-8f5b-5150a053dd16)

### Model Architecture

* **Transformer Encoder**: Multi-layer transformer with custom attention mechanism
* **Attention Mechanism**: Multi-head self-attention with Query-Key-Value (QKV) vectors

  * Queries, Keys, and Values are linearly projected from input embeddings
  * Scaled dot-product attention: `Attention(Q, K, V) = softmax((Q·K^T)/√d_k) · V`
  * Multi-head: 8 attention heads with head dimension derived from model dimension
* **Feed-Forward Networks**: Position-wise feed-forward layers with ReLU activation
* **Positional Encoding**: Learned positional embeddings added to input embeddings
* **Layer Normalisation**: Applied before attention and feed-forward layers

### Configuration

Model parameters can be viewed and adjusted in the GUI or modified in `src/model_config.py`.

## Features

* **Transformer Model**: Multi-head self-attention for next-word prediction
* **Interactive GUI**: Three-tab interface (Input, Training, Visualisation)
* **Real-time Visualisation**: Attention heatmaps and embedding plots
* **Smart Autocomplete**: Context-aware token suggestions
* **Training Interface**: Real-time loss curves and model monitoring
* **CUDA Support**: GPU acceleration when available
* **Comprehensive Testing**: Tests designed for full coverage

## Requirements

| Component    | Version/Details      |
| ------------ | -------------------- |
| Python       | 3.12+                |
| PyTorch      | 2.0+ (CUDA optional) |
| PyQt5        | Latest               |
| NumPy        | Latest               |
| Matplotlib   | Latest               |
| Scikit-learn | Latest               |

## Disclaimer
This project is for educational purposes only. The code may not be finished and polished.
