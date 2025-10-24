"""
Neuromorphic Continual Learning System.

A biologically-inspired continual learning architecture that integrates
multimodal document understanding, concept-level representation learning,
and neuromorphic memory using spiking neural networks.

This package provides:
- Concept-level encoding of multimodal inputs
- Dynamic prototype clustering for memory organization  
- Spiking neural network based associative memory
- Answer composition for various downstream tasks
- Distributed training and evaluation capabilities
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@company.com"

from .configs.schema import SystemConfig
from .core.system import NeuromorphicContinualLearningSystem
from .data.dataloader import NeuromorphicDataModule

__all__ = [
    "SystemConfig",
    "NeuromorphicContinualLearningSystem", 
    "NeuromorphicDataModule",
]
