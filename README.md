# Small Language Model (SLM)

A transformer-based language model with a graphical user interface for training, inference, and real-time exploration of attention mechanisms and embeddings. Features a clean, maintainable codebase with proper separation of concerns and comprehensive testing.

## Installation

1. Clone or download the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
3. Run the setup script for your platform:
    - Windows: `setup.bat` (lists installation options - follow on-screen instructions)
    - Linux/Mac: `setup.sh` (automatically installs dependencies)

The Linux/Mac script automatically detects CUDA and installs the appropriate dependencies. The Windows script displays options that you need to follow manually. CUDA is optional for GPU acceleration; if available, it will be used for faster training.

## Usage

Run `python -m src` to start the GUI application. The interface provides three main tabs:

- **Input Tab**: Text input with tokenisation and next-token prediction
- **Training Tab**: Model training with real-time loss curves and metrics
- **Visualisation Tab**: Attention mechanism exploration and embedding visualisation

## Result

<video src="readme_graphics/demo.mp4" controls width="600"></video>

## Architecture

The codebase follows a layered architecture:

- **GUI Layer**: User interface components and managers
- **Service Layer**: Service logic (ModelService, DataService, VocabService)
- **Core Layer**: Fundamental ML components and utilities
- **Training Layer**: Training pipeline and optimisation

## Features

- **Transformer Model**: Multi-head self-attention for next-word prediction
- **Interactive GUI**: Three-tab interface (Input, Training, Visualisation) with dark theme
- **Real-time Visualisation**: Attention heatmaps and embedding plots
- **Smart Autocomplete**: Context-aware token suggestions
- **Training Interface**: Real-time loss curves and model monitoring
- **CUDA Support**: GPU acceleration when available
- **Comprehensive Testing**: Tests designed for full coverage

## Technical Overview

### Architecture
SLM uses a layered architecture:

- **Core Layer**: Models, utilities, and serialisation
- **Service Layer**: Business logic (model, data, vocab services)
- **Training Layer**: Training pipeline and optimisation
- **GUI Layer**: User interface with visualisation
- **Test Layer**: Test suite for development

### Model Architecture
- **Transformer Encoder**: Multi-layer transformer with custom attention mechanism
- **Attention Mechanism**: Multi-head self-attention with Query-Key-Value (QKV) vectors
  - Queries, Keys, and Values are linearly projected from input embeddings
  - Scaled dot-product attention: `Attention(Q, K, V) = softmax((Q·K^T)/√d_k) · V`
  - Multi-head: 8 attention heads with head dimension derived from model dimension
- **Feed-Forward Networks**: Position-wise feed-forward layers with ReLU activation
- **Positional Encoding**: Learned positional embeddings added to input embeddings
- **Layer Normalisation**: Applied before attention and feed-forward layers

### Configuration
Model parameters can be viewed and adjusted in the GUI or modified in `src/model_config.py`.

## Requirements

| Component          | Version/Details          |
|--------------------|--------------------------|
| Python             | 3.12+                    |
| PyTorch            | 2.0+ (CUDA optional)     |
| PyQt5              | Latest                   |
| NumPy              | Latest                   |
| Matplotlib         | Latest                   |
| Scikit-learn       | Latest                   |
