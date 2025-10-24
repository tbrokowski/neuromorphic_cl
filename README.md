# Neuromorphic Continual Learning System

A biologically-inspired continual learning architecture that integrates multimodal document understanding, concept-level representation learning, and neuromorphic memory using spiking neural networks (SNNs).

## Overview

This system addresses the challenge of continual learning in AI by implementing a neuromorphic approach that learns continuously from diverse data streams without catastrophic forgetting. Instead of relying on symbolic knowledge graphs, it encodes knowledge as concept embeddings dynamically clustered into prototypes that form a concept manifold.

### Key Features

- **Concept-Level Encoding**: Multimodal document understanding using various vision encoders (ViT, DeepSeek-OCR, Donut, LayoutLMv3)
- **Dynamic Prototype Management**: Self-organizing concept clustering with efficient similarity search
- **Neuromorphic Memory**: Spiking neural network-based attractor memory with STDP learning
- **Distributed Training**: Multi-node, multi-GPU training with PyTorch Lightning
- **Medical AI Focus**: Specialized for medical documents, images, and question-answering tasks

## System Architecture

The system consists of four primary modules:

### 1. Concept Encoder (CE)
- Extracts high-level multimodal embeddings from visual and textual inputs
- Supports multiple backbone encoders with CLIP-style contrastive projection
- Handles document rendering, medical images, and text processing

### 2. Prototype Manager (PM)
- Dynamically maintains evolving concept prototypes using EMA updates
- Efficient similarity search with FAISS/HNSWLIB backends
- Automatic prototype merging, splitting, and maintenance

### 3. Spiking Neural Network (SNN)
- Implements attractor-based associative memory over prototypes
- Uses biologically plausible STDP learning rules with reward modulation
- Supports LIF, PLIF, and ALIF neuron models

### 4. Answer Composer (AC)
- Integrates SNN activations with reasoning modules
- Supports classification, text generation, and visual question answering
- Evidence-based response generation with abstention capabilities

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.7+ (for GPU training)
- 16GB+ RAM (32GB+ recommended for large datasets)

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd neuromorphic_cl_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install optional dependencies
pip install -e ".[gpu]"  # For GPU support
pip install -e ".[dev]"  # For development tools
```

### Install Dependencies

```bash
# Core dependencies
pip install torch torchvision pytorch-lightning
pip install transformers datasets
pip install spikingjelly norse  # Neuromorphic computing
pip install faiss-cpu hnswlib  # Vector search
pip install opencv-python pillow pdf2image

# Optional: For medical datasets
pip install pydicom nibabel
pip install scikit-image

# Optional: For visualization
pip install matplotlib seaborn wandb
```

## Quick Start

### 1. Generate Configuration

```bash
# Generate a basic configuration template
neuromorphic-train generate-config --template basic --output config.yaml

# Or use the provided default configuration
cp configs/default_config.yaml config.yaml
```

### 2. Update Data Paths

Edit `config.yaml` to point to your datasets:

```yaml
data:
  pubmed_path: "/path/to/pubmed/data"
  mimic_cxr_path: "/path/to/mimic-cxr"
  vqa_rad_path: "/path/to/vqa-rad"
  bioasq_path: "/path/to/bioasq"
```

### 3. Train the Model

```bash
# Single GPU training
neuromorphic-train train --config config.yaml

# Multi-GPU training
neuromorphic-train train --config config.yaml --num-gpus 4

# Distributed training across nodes
neuromorphic-train train --config config.yaml --num-nodes 2 --num-gpus 4
```

### 4. Evaluate the Model

```bash
# Evaluate on test sets
neuromorphic-eval evaluate \
  --config config.yaml \
  --checkpoint checkpoints/best_model.ckpt \
  --output-dir evaluation_results

# Analyze results
neuromorphic-eval analyze-results \
  --results-file evaluation_results/evaluation_results.json \
  --output-dir analysis_plots
```

### 5. Run Inference

```bash
# Single image inference
neuromorphic-infer infer \
  --config config.yaml \
  --checkpoint checkpoints/best_model.ckpt \
  --input image.jpg \
  --task-type classification

# Batch inference
neuromorphic-infer batch-infer \
  --config config.yaml \
  --checkpoint checkpoints/best_model.ckpt \
  --input-dir images/ \
  --output-dir results/ \
  --task-type classification
```

## Configuration

The system uses YAML configuration files with comprehensive settings for all components. Key configuration sections:

### Concept Encoder
```yaml
concept_encoder:
  encoder_type: "vit"  # vit, deepseek_ocr, donut, layoutlmv3
  projection_dim: 256
  freeze_backbone: true
  contrastive_temperature: 0.07
```

### Prototype Manager
```yaml
prototype_manager:
  max_prototypes: 5000
  similarity_threshold: 0.85
  indexing_backend: "faiss"
  maintenance_interval: 1000
```

### Spiking Neural Network
```yaml
snn:
  neuron_type: "lif"  # lif, plif, alif
  num_timesteps: 20
  stdp_lr: 0.01
  reward_modulation: true
```

### Distributed Training
```yaml
distributed:
  num_nodes: 1
  num_gpus_per_node: 4
  backend: "nccl"
  precision: "16-mixed"
  use_fsdp: false
```

## Datasets

The system supports several medical and scientific datasets:

### PubMed Central (PMC)
- Open-access biomedical papers
- PDF rendering to images for visual processing
- Text extraction for multimodal learning

### MIMIC-CXR
- Chest X-ray images with radiology reports
- Multi-label classification tasks
- 14 CheXpert pathology labels

### VQA-RAD
- Visual Question Answering for radiology
- Medical image understanding and reasoning
- Natural language question-answer pairs

### BioASQ
- Biomedical question answering
- Multiple question types (factoid, list, yes/no, summary)
- Large-scale biomedical literature corpus

## Training

### Single GPU Training

```bash
neuromorphic-train train \
  --config config.yaml \
  --experiment-name my_experiment \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --max-epochs 50
```

### Distributed Training

```bash
# Multi-GPU on single node
neuromorphic-train train \
  --config config.yaml \
  --num-gpus 4 \
  --batch-size 16  # Per-GPU batch size

# Multi-node training
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
  --master_addr=<master_ip> --master_port=12355 \
  -m neuromorphic_cl.cli.train \
  --config config.yaml
```

### SLURM Training

```bash
# Submit SLURM job
sbatch --nodes=2 --ntasks-per-node=4 --gres=gpu:4 \
  --wrap="neuromorphic-train train --config config.yaml --slurm"
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics for continual learning:

### Standard Metrics
- **Accuracy**: Classification accuracy on each task
- **Precision/Recall/F1**: Detailed classification metrics
- **Top-k Accuracy**: Multi-class ranking performance

### Continual Learning Metrics
- **Catastrophic Forgetting**: Performance degradation on previous tasks
- **Forward Transfer**: Improvement on new tasks due to previous learning
- **Backward Transfer**: Improvement on old tasks due to new learning

### System Metrics
- **Memory Efficiency**: Prototype storage vs. raw data storage
- **Energy Efficiency**: SNN spike counts and energy consumption
- **Prototype Statistics**: Clustering quality and usage patterns

## Memory System Analysis

### Prototype Analysis
```bash
# Analyze prototype clustering
python -c "
from neuromorphic_cl.utils.metrics import PrototypeAnalyzer
from neuromorphic_cl.core.system import NeuromorphicContinualLearningSystem

model = NeuromorphicContinualLearningSystem.load_from_checkpoint('model.ckpt')
analyzer = PrototypeAnalyzer(model.prototype_manager)
stats = analyzer.compute_prototype_statistics()
print(stats)
"
```

### SNN Dynamics
```python
# Visualize spike patterns
import matplotlib.pyplot as plt

# Run inference with full dynamics
outputs = model(images, return_intermediate=True)
spike_output = outputs['snn_outputs']['spike_output']

# Plot spike raster
plt.figure(figsize=(12, 8))
for neuron_id in range(min(50, spike_output.size(-1))):
    spike_times = torch.where(spike_output[:, 0, neuron_id] > 0)[0]
    plt.scatter(spike_times, [neuron_id] * len(spike_times), s=1)

plt.xlabel('Time Step')
plt.ylabel('Neuron ID')
plt.title('Spike Raster Plot')
plt.show()
```

## Extending the System

### Adding New Encoders

```python
from neuromorphic_cl.utils.registry import register_encoder
from neuromorphic_cl.core.concept_encoder import ConceptEncoder

@register_encoder("my_encoder")
class MyCustomEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize your encoder
        
    def forward(self, pixel_values, **kwargs):
        # Implement encoding logic
        return concept_embedding, saliency_weights
```

### Adding New Datasets

```python
from neuromorphic_cl.data.dataloader import BaseMultimodalDataset

class MyDataset(BaseMultimodalDataset):
    def _load_data(self):
        # Load your dataset
        pass
        
    def __getitem__(self, idx):
        # Return formatted sample
        return {
            "pixel_values": ...,
            "text_tokens": ...,
            "labels": ...,
            "task_type": "my_task"
        }
```

### Custom Loss Functions

```python
from neuromorphic_cl.utils.losses import register_loss

@register_loss("my_loss")
class MyCustomLoss(nn.Module):
    def forward(self, predictions, targets):
        # Implement loss computation
        return loss_value
```

## Performance Optimization

### Memory Optimization
- Use mixed precision training (`precision: "16-mixed"`)
- Enable gradient checkpointing for large models
- Adjust batch sizes based on GPU memory
- Use FSDP for very large models

### Speed Optimization
- Increase `num_workers` for data loading
- Use `pin_memory: true` for GPU training
- Enable `benchmark: true` for consistent input sizes
- Use compiled models with `torch.compile()` (PyTorch 2.0+)

### Distributed Optimization
- Use NCCL backend for multi-GPU training
- Enable `gradient_as_bucket_view` for DDP
- Tune `find_unused_parameters` based on model architecture
- Use appropriate sharding strategies for FSDP

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --batch-size 16
   
   # Use gradient accumulation
   --accumulate-grad-batches 2
   
   # Enable mixed precision
   --precision 16-mixed
   ```

2. **Slow Data Loading**
   ```yaml
   data:
     num_workers: 8  # Increase workers
     pin_memory: true
     prefetch_factor: 4
   ```

3. **Prototype Memory Issues**
   ```yaml
   prototype_manager:
     max_prototypes: 2000  # Reduce max prototypes
     maintenance_interval: 500  # More frequent cleanup
   ```

4. **SNN Convergence Issues**
   ```yaml
   snn:
     stdp_lr: 0.005  # Reduce learning rate
     num_timesteps: 10  # Reduce simulation time
   ```

### Debugging

Enable debug mode for detailed logging:
```bash
neuromorphic-train train --config config.yaml --debug
```

Monitor system resources:
```bash
# GPU memory usage
nvidia-smi -l 1

# System memory
htop

# Training progress
tensorboard --logdir logs/
```

## Citation

If you use this system in your research, please cite:

```bibtex
@article{neuromorphic_cl_2024,
  title={Neuromorphic Continual Learning with Concept-Level Memory},
  author={Research Team},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/
isort src/

# Type checking
mypy src/
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Contact the research team at research@company.com

## Roadmap

- [ ] Hardware acceleration on neuromorphic chips (Loihi 2, SpiNNaker)
- [ ] Additional encoder architectures (CLIP, BLIP)
- [ ] Hierarchical prototype organization
- [ ] Real-time deployment capabilities
- [ ] Interactive visualization tools
- [ ] Mobile/edge deployment optimization
