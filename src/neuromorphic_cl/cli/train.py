"""
Training CLI for Neuromorphic Continual Learning System.

This module provides the command-line interface for training the
neuromorphic continual learning system with distributed support.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from ..configs.schema import DistributedConfig, SystemConfig
from ..data.dataloader import NeuromorphicDataModule
from ..training.distributed import (
    DistributedTrainer,
    estimate_memory_requirements,
    launch_distributed_training,
    setup_slurm_distributed,
)
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config", 
    "-c", 
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to configuration file (YAML)"
)
@click.option(
    "--experiment-name",
    "-n",
    type=str,
    help="Override experiment name"
)
@click.option(
    "--resume-from",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    help="Resume training from checkpoint"
)
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    help="Override data directory"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Override output directory"
)
@click.option(
    "--num-gpus",
    type=int,
    help="Override number of GPUs per node"
)
@click.option(
    "--num-nodes",
    type=int,
    help="Override number of nodes"
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    help="Override batch size"
)
@click.option(
    "--learning-rate",
    "-lr",
    type=float,
    help="Override learning rate"
)
@click.option(
    "--max-epochs",
    "-e",
    type=int,
    help="Override maximum epochs"
)
@click.option(
    "--precision",
    type=click.Choice(["16", "32", "64", "16-mixed", "bf16-mixed"]),
    help="Training precision"
)
@click.option(
    "--seed",
    type=int,
    help="Random seed for reproducibility"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Perform dry run without actual training"
)
@click.option(
    "--profile",
    is_flag=True,
    help="Enable profiling"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode"
)
@click.option(
    "--slurm",
    is_flag=True,
    help="Running on SLURM cluster"
)
@click.option(
    "--wandb-project",
    type=str,
    help="Weights & Biases project name"
)
@click.option(
    "--tags",
    type=str,
    help="Comma-separated tags for experiment"
)
def train(
    config: Path,
    experiment_name: Optional[str] = None,
    resume_from: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    num_gpus: Optional[int] = None,
    num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    max_epochs: Optional[int] = None,
    precision: Optional[str] = None,
    seed: Optional[int] = None,
    dry_run: bool = False,
    profile: bool = False,
    debug: bool = False,
    slurm: bool = False,
    wandb_project: Optional[str] = None,
    tags: Optional[str] = None,
) -> None:
    """
    Train the neuromorphic continual learning system.
    
    This command trains the system using the specified configuration,
    with support for distributed training across multiple GPUs and nodes.
    """
    try:
        # Setup SLURM environment if needed
        if slurm:
            slurm_env = setup_slurm_distributed()
            click.echo(f"SLURM environment: {slurm_env}")
        
        # Load and validate configuration
        click.echo(f"Loading configuration from {config}")
        system_config, distributed_config = load_and_validate_config(
            config,
            experiment_name=experiment_name,
            data_dir=data_dir,
            output_dir=output_dir,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            precision=precision,
            seed=seed,
            wandb_project=wandb_project,
            tags=tags,
        )
        
        # Setup logging
        log_level = "DEBUG" if debug else system_config.log_level
        setup_logging(
            level=log_level,
            log_dir=system_config.log_dir,
            experiment_name=system_config.experiment_name,
        )
        
        # Log system information
        log_system_info(system_config, distributed_config)
        
        # Estimate memory requirements
        memory_info = estimate_memory_requirements(
            system_config,
            system_config.data.batch_size,
        )
        click.echo(f"Estimated memory requirements: {memory_info}")
        
        # Check GPU availability
        if not torch.cuda.is_available() and distributed_config.num_gpus_per_node > 0:
            click.echo("Warning: CUDA not available but GPUs requested", err=True)
        
        if dry_run:
            click.echo("Dry run completed successfully")
            return
        
        # Create data module
        click.echo("Setting up data module...")
        data_module = NeuromorphicDataModule(system_config)
        
        # Validate data availability
        validate_data_paths(system_config)
        
        # Launch training
        click.echo("Starting training...")
        launch_distributed_training(
            config=system_config,
            distributed_config=distributed_config,
            data_module=data_module,
            resume_from_checkpoint=str(resume_from) if resume_from else None,
        )
        
        click.echo("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if debug:
            raise
        sys.exit(1)


def load_and_validate_config(
    config_path: Path,
    **overrides,
) -> tuple[SystemConfig, DistributedConfig]:
    """
    Load configuration file and apply overrides.
    
    Args:
        config_path: Path to configuration file
        **overrides: Configuration overrides
        
    Returns:
        Tuple of (SystemConfig, DistributedConfig)
    """
    # Load base configuration
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to OmegaConf for easier manipulation
    cfg = OmegaConf.create(config_dict)
    
    # Apply overrides
    for key, value in overrides.items():
        if value is not None:
            if key == "experiment_name":
                cfg.experiment_name = value
            elif key == "data_dir":
                # Update all data paths
                if "data" in cfg:
                    for data_key in ["pubmed_path", "mimic_cxr_path", "vqa_rad_path", "bioasq_path"]:
                        if data_key in cfg.data and cfg.data[data_key]:
                            cfg.data[data_key] = str(Path(value) / Path(cfg.data[data_key]).name)
            elif key == "output_dir":
                cfg.output_dir = str(value)
                cfg.log_dir = str(Path(value) / "logs")
                cfg.checkpoint_dir = str(Path(value) / "checkpoints")
            elif key == "num_gpus":
                cfg.distributed.num_gpus_per_node = value
            elif key == "num_nodes":
                cfg.distributed.num_nodes = value
            elif key == "batch_size":
                cfg.data.batch_size = value
            elif key == "learning_rate":
                cfg.training.learning_rate = value
            elif key == "max_epochs":
                cfg.training.max_epochs = value
            elif key == "precision":
                cfg.distributed.precision = value
            elif key == "seed":
                cfg.seed = value
            elif key == "wandb_project":
                cfg.project_name = value
            elif key == "tags":
                if value:
                    cfg.tags = [tag.strip() for tag in value.split(",")]
    
    # Create and validate configurations
    try:
        system_config = SystemConfig(**OmegaConf.to_object(cfg))
        distributed_config = system_config.distributed
        
        return system_config, distributed_config
        
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")


def validate_data_paths(config: SystemConfig) -> None:
    """Validate that required data paths exist."""
    data_config = config.data
    
    paths_to_check = [
        ("PubMed", data_config.pubmed_path),
        ("MIMIC-CXR", data_config.mimic_cxr_path),
        ("VQA-RAD", data_config.vqa_rad_path),
        ("BioASQ", data_config.bioasq_path),
    ]
    
    available_datasets = []
    for name, path in paths_to_check:
        if path and Path(path).exists():
            available_datasets.append(name)
        elif path:
            logger.warning(f"{name} dataset path does not exist: {path}")
    
    if not available_datasets:
        raise ValueError("No valid dataset paths found")
    
    logger.info(f"Available datasets: {', '.join(available_datasets)}")


def log_system_info(system_config: SystemConfig, distributed_config: DistributedConfig) -> None:
    """Log system and configuration information."""
    logger.info("="*50)
    logger.info("NEUROMORPHIC CONTINUAL LEARNING TRAINING")
    logger.info("="*50)
    
    # System info
    logger.info(f"Experiment: {system_config.experiment_name}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Distributed setup
    logger.info(f"Distributed training: {distributed_config.num_nodes}x{distributed_config.num_gpus_per_node}")
    logger.info(f"Backend: {distributed_config.backend}")
    logger.info(f"Precision: {distributed_config.precision}")
    
    # Model config
    logger.info(f"Encoder: {system_config.concept_encoder.encoder_type}")
    logger.info(f"Embedding dim: {system_config.concept_encoder.projection_dim}")
    logger.info(f"Max prototypes: {system_config.prototype_manager.max_prototypes}")
    logger.info(f"SNN neuron type: {system_config.snn.neuron_type}")
    
    # Training config
    logger.info(f"Batch size: {system_config.data.batch_size}")
    logger.info(f"Learning rate: {system_config.training.learning_rate}")
    logger.info(f"Max epochs: {system_config.training.max_epochs}")
    
    logger.info("="*50)


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to configuration file"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for memory estimates"
)
def estimate_memory(config: Path, output: Optional[Path] = None) -> None:
    """
    Estimate memory requirements for training configuration.
    
    This helps in planning resource allocation for distributed training.
    """
    try:
        # Load configuration
        with open(config) as f:
            config_dict = yaml.safe_load(f)
        
        system_config = SystemConfig(**config_dict)
        
        # Estimate memory for different batch sizes
        batch_sizes = [1, 8, 16, 32, 64, 128]
        estimates = {}
        
        for batch_size in batch_sizes:
            memory_info = estimate_memory_requirements(
                system_config,
                batch_size,
            )
            estimates[batch_size] = memory_info
        
        # Display results
        click.echo("Memory Requirements Estimation")
        click.echo("="*50)
        
        for batch_size, info in estimates.items():
            click.echo(f"\nBatch size: {batch_size}")
            click.echo(f"  Total memory: {info['total_memory_gb']:.2f} GB")
            click.echo(f"  Recommended GPU: {info['recommended_gpu_memory_gb']:.2f} GB")
            click.echo(f"  Model memory: {info['encoder_memory_gb'] + info['llm_memory_gb']:.2f} GB")
            click.echo(f"  Prototype memory: {info['prototype_memory_gb']:.2f} GB")
            click.echo(f"  SNN memory: {info['snn_memory_gb']:.2f} GB")
        
        # Save to file if requested
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(estimates, f, indent=2)
            click.echo(f"\nEstimates saved to {output}")
            
    except Exception as e:
        click.echo(f"Failed to estimate memory: {e}", err=True)
        sys.exit(1)


@click.command()
@click.option(
    "--template",
    "-t",
    type=click.Choice(["basic", "distributed", "medical"]),
    default="basic",
    help="Configuration template to generate"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="config.yaml",
    help="Output configuration file"
)
def generate_config(template: str, output: Path) -> None:
    """
    Generate a configuration template.
    
    This creates a template configuration file that can be customized
    for specific training scenarios.
    """
    try:
        # Create template configuration
        if template == "basic":
            config = create_basic_config()
        elif template == "distributed":
            config = create_distributed_config()
        elif template == "medical":
            config = create_medical_config()
        else:
            raise ValueError(f"Unknown template: {template}")
        
        # Save to file
        with open(output, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        click.echo(f"Generated {template} configuration template: {output}")
        click.echo(f"Edit the file to customize for your use case")
        
    except Exception as e:
        click.echo(f"Failed to generate config: {e}", err=True)
        sys.exit(1)


def create_basic_config() -> dict:
    """Create basic configuration template."""
    return {
        "experiment_name": "neuromorphic_cl_basic",
        "project_name": "neuromorphic-continual-learning",
        "seed": 42,
        
        "concept_encoder": {
            "encoder_type": "vit",
            "embedding_dim": 768,
            "projection_dim": 256,
            "freeze_backbone": True,
            "unfreeze_last_n_blocks": 2,
        },
        
        "prototype_manager": {
            "similarity_threshold": 0.85,
            "max_prototypes": 5000,
            "indexing_backend": "faiss",
        },
        
        "snn": {
            "neuron_type": "lif",
            "num_timesteps": 20,
            "spike_threshold": 1.0,
        },
        
        "answer_composer": {
            "top_k_prototypes": 10,
            "llm_model_name": "microsoft/DialoGPT-medium",
        },
        
        "data": {
            "batch_size": 32,
            "num_workers": 4,
            "image_size": [224, 224],
            "text_max_length": 512,
        },
        
        "training": {
            "learning_rate": 1e-4,
            "max_epochs": 50,
            "optimizer": "adamw",
            "scheduler": "cosine",
        },
        
        "distributed": {
            "num_nodes": 1,
            "num_gpus_per_node": 1,
            "precision": "16-mixed",
        },
    }


def create_distributed_config() -> dict:
    """Create distributed training configuration template."""
    config = create_basic_config()
    
    # Update for distributed training
    config.update({
        "experiment_name": "neuromorphic_cl_distributed",
        
        "data": {
            **config["data"],
            "batch_size": 16,  # Smaller per-GPU batch size
            "num_workers": 8,
        },
        
        "distributed": {
            "num_nodes": 2,
            "num_gpus_per_node": 4,
            "backend": "nccl",
            "precision": "16-mixed",
            "use_fsdp": True,
            "find_unused_parameters": False,
        },
        
        "training": {
            **config["training"],
            "learning_rate": 2e-4,  # Higher LR for larger effective batch size
            "accumulate_grad_batches": 2,
        },
    })
    
    return config


def create_medical_config() -> dict:
    """Create medical imaging focused configuration template."""
    config = create_basic_config()
    
    # Update for medical imaging
    config.update({
        "experiment_name": "neuromorphic_cl_medical",
        
        "concept_encoder": {
            **config["concept_encoder"],
            "encoder_type": "vit",  # Good for medical images
            "projection_dim": 512,  # Larger for complex medical concepts
        },
        
        "prototype_manager": {
            **config["prototype_manager"],
            "max_prototypes": 10000,  # More prototypes for medical diversity
            "similarity_threshold": 0.8,  # Slightly lower for medical nuances
        },
        
        "data": {
            **config["data"],
            "batch_size": 16,  # Medical images can be memory-intensive
            "image_size": [384, 384],  # Higher resolution for medical details
        },
        
        # Add medical-specific data paths
        "data_paths": {
            "mimic_cxr_path": "/path/to/mimic-cxr",
            "vqa_rad_path": "/path/to/vqa-rad",
            "pubmed_path": "/path/to/pubmed",
        },
    })
    
    return config


@click.group()
def main() -> None:
    """Neuromorphic Continual Learning Training CLI."""
    pass


# Add commands to the group
main.add_command(train)
main.add_command(estimate_memory)
main.add_command(generate_config)


if __name__ == "__main__":
    main()
