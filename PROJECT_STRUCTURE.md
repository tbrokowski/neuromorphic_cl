# Neuromorphic Continual Learning System - Project Structure

This document provides an overview of the project structure and key components.

## Directory Structure

```
neuromorphic_cl_system/
├── README.md                      # Main documentation
├── pyproject.toml                 # Project configuration and dependencies
├── requirements.txt               # Python dependencies
├── setup.sh                       # Quick setup script
├── configs/                       # Configuration files
│   ├── default_config.yaml        # Basic configuration template
│   └── medical_config.yaml        # Medical imaging optimized config
├── examples/                      # Example usage scripts
│   └── basic_usage.py             # Complete demo of system components
└── src/neuromorphic_cl/           # Main source code
    ├── __init__.py                # Package initialization
    ├── cli/                       # Command line interfaces
    │   ├── train.py               # Training CLI
    │   ├── evaluate.py            # Evaluation CLI
    │   └── inference.py           # Inference CLI
    ├── configs/                   # Configuration schemas
    │   └── schema.py              # Pydantic configuration models
    ├── core/                      # Core system components
    │   ├── concept_encoder.py     # Multimodal concept encoding
    │   ├── prototype_manager.py   # Dynamic prototype clustering
    │   ├── snn.py                 # Spiking neural network memory
    │   ├── answer_composer.py     # Response generation
    │   └── system.py              # Main system integration
    ├── data/                      # Data loading and processing
    │   └── dataloader.py          # Multimodal data loaders
    ├── training/                  # Training infrastructure
    │   └── distributed.py         # Distributed training framework
    └── utils/                     # Utility modules
        ├── losses.py              # Loss functions
        ├── metrics.py             # Evaluation metrics
        ├── preprocessing.py       # Data preprocessing
        ├── registry.py            # Component registry
        └── logging.py             # Logging utilities
```

## Key Components

### 1. Core System (`src/neuromorphic_cl/core/`)

#### Concept Encoder (`concept_encoder.py`)
- **Purpose**: Extract multimodal concept embeddings from images and text
- **Key Features**:
  - Multiple backbone encoders (ViT, DeepSeek-OCR, Donut, LayoutLMv3)
  - CLIP-style contrastive projection heads
  - Token-level saliency mapping
  - Medical image preprocessing
- **Main Classes**: `ConceptEncoder`, `ConceptProjectionHead`

#### Prototype Manager (`prototype_manager.py`)
- **Purpose**: Dynamic clustering and management of concept prototypes
- **Key Features**:
  - EMA-based prototype updates
  - Automatic merging and splitting
  - Efficient similarity search (FAISS/HNSWLIB)
  - Prototype statistics and analysis
- **Main Classes**: `PrototypeManager`, `Prototype`

#### Spiking Neural Network (`snn.py`)
- **Purpose**: Neuromorphic associative memory using SNNs
- **Key Features**:
  - Multiple neuron types (LIF, PLIF, ALIF)
  - STDP learning with reward modulation
  - Attractor dynamics for memory retrieval
  - Energy efficiency tracking
- **Main Classes**: `SpikingNeuralNetwork`, `NeuromorphicMemory`, `STDPSynapse`

#### Answer Composer (`answer_composer.py`)
- **Purpose**: Generate responses from memory activations
- **Key Features**:
  - Evidence extraction and ranking
  - Text generation with language models
  - Structured prediction heads
  - Abstention mechanisms
- **Main Classes**: `AnswerComposer`, `EvidenceExtractor`, `TextGenerator`

#### System Integration (`system.py`)
- **Purpose**: Main system class integrating all components
- **Key Features**:
  - End-to-end training pipeline
  - Continual learning task management
  - Comprehensive evaluation metrics
  - State saving and loading
- **Main Classes**: `NeuromorphicContinualLearningSystem`

### 2. Training Infrastructure (`src/neuromorphic_cl/training/`)

#### Distributed Training (`distributed.py`)
- **Purpose**: Multi-node, multi-GPU training support
- **Key Features**:
  - PyTorch Lightning integration
  - FSDP and DDP strategies
  - Prototype synchronization across workers
  - Memory estimation utilities
- **Main Classes**: `DistributedTrainer`, `DistributedNeuromorphicSystem`

### 3. Data Processing (`src/neuromorphic_cl/data/`)

#### Data Loaders (`dataloader.py`)
- **Purpose**: Multimodal dataset loading and preprocessing
- **Key Features**:
  - Support for PubMed, MIMIC-CXR, VQA-RAD, BioASQ
  - Continual learning data streaming
  - Medical image preprocessing
  - Text tokenization and encoding
- **Main Classes**: `NeuromorphicDataModule`, `BaseMultimodalDataset`

### 4. Command Line Interfaces (`src/neuromorphic_cl/cli/`)

#### Training CLI (`train.py`)
- **Commands**: `train`, `estimate-memory`, `generate-config`
- **Purpose**: Training management and configuration

#### Evaluation CLI (`evaluate.py`)
- **Commands**: `evaluate`, `analyze-results`
- **Purpose**: Model evaluation and analysis

#### Inference CLI (`inference.py`)
- **Commands**: `infer`, `batch-infer`
- **Purpose**: Running inference on new data

### 5. Configuration (`src/neuromorphic_cl/configs/`)

#### Schema (`schema.py`)
- **Purpose**: Type-safe configuration using Pydantic
- **Key Classes**: `SystemConfig`, `ConceptEncoderConfig`, `PrototypeManagerConfig`, etc.

### 6. Utilities (`src/neuromorphic_cl/utils/`)

#### Loss Functions (`losses.py`)
- **Available Losses**: InfoNCE, Supervised Contrastive, Prototype Alignment, etc.
- **Purpose**: Training objectives for continual learning

#### Metrics (`metrics.py`)
- **Available Metrics**: Accuracy, Forgetting, Transfer, Energy Efficiency, etc.
- **Purpose**: Comprehensive evaluation of continual learning

#### Preprocessing (`preprocessing.py`)
- **Features**: PDF rendering, medical image processing, text cleaning
- **Purpose**: Data preparation for multimodal learning

#### Registry (`registry.py`)
- **Purpose**: Dynamic component registration and instantiation
- **Features**: Type-safe registries, lazy loading

#### Logging (`logging.py`)
- **Features**: Structured logging, experiment tracking, distributed logging
- **Purpose**: Comprehensive monitoring and debugging

## Configuration Files

### Default Configuration (`configs/default_config.yaml`)
- Basic configuration suitable for getting started
- Moderate resource requirements
- General-purpose settings

### Medical Configuration (`configs/medical_config.yaml`)
- Optimized for medical imaging tasks
- Higher resolution processing
- Medical-specific model choices
- Enhanced prototype capacity

## Example Usage (`examples/basic_usage.py`)

Complete demonstration script showing:
- System component initialization
- Training loop implementation
- Inference procedures
- Continual learning simulation

## Quick Start

1. **Setup**: Run `bash setup.sh` for automatic installation
2. **Configure**: Edit `config.yaml` with your dataset paths
3. **Train**: `neuromorphic-train train --config config.yaml`
4. **Evaluate**: `neuromorphic-eval evaluate --config config.yaml --checkpoint model.ckpt`
5. **Infer**: `neuromorphic-infer infer --config config.yaml --checkpoint model.ckpt --input image.jpg`

## Development

### Adding New Components

1. **Encoders**: Register in `core/concept_encoder.py` using `@ENCODER_REGISTRY.register`
2. **Datasets**: Inherit from `BaseMultimodalDataset` in `data/dataloader.py`
3. **Loss Functions**: Add to `utils/losses.py` and register in training config
4. **Metrics**: Add to `utils/metrics.py` and evaluation config

### Testing

Run the basic usage example to test installation:
```bash
python examples/basic_usage.py
```

### Code Style

- Use Black for code formatting: `black src/`
- Use isort for import sorting: `isort src/`
- Use mypy for type checking: `mypy src/`
- Follow PEP 8 guidelines

## Dependencies

### Core Dependencies
- PyTorch 2.0+ (deep learning framework)
- PyTorch Lightning (training infrastructure)
- Transformers (pre-trained models)
- SpikingJelly/Norse (neuromorphic computing)
- FAISS/HNSWLIB (vector search)

### Optional Dependencies
- Weights & Biases (experiment tracking)
- Medical imaging libraries (pydicom, nibabel)
- Visualization libraries (matplotlib, seaborn)

## Hardware Requirements

### Minimum Requirements
- Python 3.9+
- 8GB RAM
- CPU-only training supported

### Recommended Requirements
- NVIDIA GPU with 16GB+ VRAM
- 32GB+ system RAM
- NVMe SSD for fast data loading

### Large-Scale Training
- Multiple NVIDIA A100/H100 GPUs
- 100GB+ RAM per node
- High-speed interconnect (InfiniBand)

## Support and Documentation

- **Main Documentation**: README.md
- **API Documentation**: Generated from docstrings
- **Examples**: examples/ directory
- **Configuration**: configs/ directory with commented examples
- **Issues**: Use GitHub issues for bug reports and feature requests
