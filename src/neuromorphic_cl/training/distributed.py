"""
Distributed Training Framework for Neuromorphic Continual Learning.

This module provides distributed training capabilities across multiple
nodes and GPUs using PyTorch Lightning, with special handling for the
prototype manager and SNN components that require synchronization.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.strategies import (
    DDPStrategy,
    FSDPStrategy,
    DeepSpeedStrategy,
)

from ..configs.schema import DistributedConfig, SystemConfig
from ..core.system import NeuromorphicContinualLearningSystem
from ..data.dataloader import NeuromorphicDataModule

logger = logging.getLogger(__name__)


class DistributedPrototypeSync:
    """
    Handles synchronization of prototype manager state across distributed processes.
    
    Since the prototype manager maintains dynamic state that needs to be
    consistent across processes, we need custom synchronization logic.
    """
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.is_master = rank == 0
    
    def sync_prototypes(self, prototype_manager: "PrototypeManager") -> None:
        """Synchronize prototype state across all processes."""
        if not dist.is_initialized() or self.world_size == 1:
            return
        
        if self.is_master:
            # Master process broadcasts its prototype state
            prototype_data = self._serialize_prototypes(prototype_manager)
            
            # Broadcast the size first
            data_size = torch.tensor(len(prototype_data), dtype=torch.long)
            dist.broadcast(data_size, src=0)
            
            # Broadcast the actual data
            if data_size > 0:
                data_tensor = torch.tensor(
                    list(prototype_data), 
                    dtype=torch.uint8
                )
                dist.broadcast(data_tensor, src=0)
                
        else:
            # Worker processes receive and apply prototype state
            data_size = torch.tensor(0, dtype=torch.long)
            dist.broadcast(data_size, src=0)
            
            if data_size > 0:
                data_tensor = torch.zeros(data_size, dtype=torch.uint8)
                dist.broadcast(data_tensor, src=0)
                
                prototype_data = bytes(data_tensor.numpy())
                self._deserialize_prototypes(prototype_manager, prototype_data)
    
    def _serialize_prototypes(self, prototype_manager: "PrototypeManager") -> bytes:
        """Serialize prototype manager state to bytes."""
        import pickle
        
        state = {
            "prototypes": {k: v.to_dict() for k, v in prototype_manager.prototypes.items()},
            "next_prototype_id": prototype_manager.next_prototype_id,
            "current_step": prototype_manager.current_step,
            "assignment_counts": prototype_manager.assignment_counts,
        }
        
        return pickle.dumps(state)
    
    def _deserialize_prototypes(
        self, 
        prototype_manager: "PrototypeManager", 
        data: bytes
    ) -> None:
        """Deserialize and apply prototype state."""
        import pickle
        from ..core.prototype_manager import Prototype
        
        state = pickle.loads(data)
        
        # Clear existing prototypes
        prototype_manager.prototypes.clear()
        
        # Restore prototypes
        for k, v in state["prototypes"].items():
            prototype_manager.prototypes[k] = Prototype.from_dict(v)
        
        prototype_manager.next_prototype_id = state["next_prototype_id"]
        prototype_manager.current_step = state["current_step"]
        prototype_manager.assignment_counts = state["assignment_counts"]
        
        # Rebuild index
        prototype_manager._rebuild_index()
    
    def aggregate_prototype_updates(
        self, 
        prototype_manager: "PrototypeManager",
        sync_interval: int = 100,
    ) -> None:
        """
        Aggregate prototype updates from all processes.
        
        This is called periodically to merge prototype updates from
        different processes, ensuring consistency.
        """
        if not dist.is_initialized() or self.world_size == 1:
            return
        
        # Collect assignment counts from all processes
        local_counts = prototype_manager.assignment_counts.copy()
        
        # Convert to tensor for all_reduce
        max_prototype_id = max(local_counts.keys()) if local_counts else 0
        
        # Get global max prototype ID
        max_id_tensor = torch.tensor(max_prototype_id, dtype=torch.long)
        dist.all_reduce(max_id_tensor, op=dist.ReduceOp.MAX)
        global_max_id = max_id_tensor.item()
        
        if global_max_id >= 0:
            # Create count tensors
            count_tensor = torch.zeros(global_max_id + 1, dtype=torch.long)
            for proto_id, count in local_counts.items():
                if proto_id <= global_max_id:
                    count_tensor[proto_id] = count
            
            # All-reduce to sum counts across processes
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            
            # Update local assignment counts
            for proto_id in range(global_max_id + 1):
                if count_tensor[proto_id] > 0:
                    prototype_manager.assignment_counts[proto_id] = count_tensor[proto_id].item()


class DistributedNeuromorphicSystem(NeuromorphicContinualLearningSystem):
    """
    Distributed version of the neuromorphic system with synchronization support.
    """
    
    def __init__(self, config: SystemConfig, distributed_config: DistributedConfig):
        super().__init__(config)
        
        self.distributed_config = distributed_config
        self.prototype_sync = None
        
        # Initialize distributed synchronization if running distributed
        if dist.is_initialized():
            self.prototype_sync = DistributedPrototypeSync(
                world_size=dist.get_world_size(),
                rank=dist.get_rank(),
            )
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Override training step to include distributed synchronization."""
        # Standard training step
        loss = super().training_step(batch, batch_idx)
        
        # Periodic prototype synchronization
        if (self.prototype_sync and 
            self.training_step_count % 50 == 0):  # Sync every 50 steps
            
            self.prototype_sync.aggregate_prototype_updates(
                self.prototype_manager
            )
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        """Synchronize prototypes at end of each epoch."""
        if self.prototype_sync:
            self.prototype_sync.sync_prototypes(self.prototype_manager)
    
    def configure_sharded_model(self) -> None:
        """Configure model sharding for FSDP if enabled."""
        if self.distributed_config.use_fsdp:
            # Wrap specific modules for FSDP
            # SNN can be sharded as it's large and stateless during forward pass
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            
            # Concept encoder can be sharded
            self.concept_encoder = FSDP(self.concept_encoder)
            
            # Answer composer can be sharded
            self.answer_composer = FSDP(self.answer_composer)
            
            # Keep prototype manager and SNN on master node for simplicity
            # (they require special synchronization logic)


class DistributedTrainer:
    """
    Main distributed training coordinator for the neuromorphic system.
    
    Handles setup of distributed training, logging, checkpointing,
    and coordination across multiple nodes and GPUs.
    """
    
    def __init__(
        self,
        config: SystemConfig,
        distributed_config: DistributedConfig,
        data_module: NeuromorphicDataModule,
        experiment_name: Optional[str] = None,
    ):
        self.config = config
        self.distributed_config = distributed_config
        self.data_module = data_module
        self.experiment_name = experiment_name or config.experiment_name
        
        # Initialize distributed backend if not already done
        self._setup_distributed()
        
        # Create system
        self.system = DistributedNeuromorphicSystem(config, distributed_config)
        
        # Setup trainer
        self.trainer = self._create_trainer()
        
        logger.info(
            f"Initialized distributed trainer with {distributed_config.num_nodes} nodes, "
            f"{distributed_config.num_gpus_per_node} GPUs per node"
        )
    
    def _setup_distributed(self) -> None:
        """Setup distributed training backend."""
        if self.distributed_config.num_nodes == 1 and self.distributed_config.num_gpus_per_node == 1:
            # Single GPU training - no distributed setup needed
            return
        
        # Set environment variables for distributed training
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"
        
        # Set CUDA device
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
    
    def _create_trainer(self) -> Trainer:
        """Create PyTorch Lightning trainer with distributed strategy."""
        
        # Select distributed strategy
        strategy = self._get_distributed_strategy()
        
        # Create callbacks
        callbacks = self._create_callbacks()
        
        # Create logger
        logger_instance = self._create_logger()
        
        # Trainer configuration
        trainer_config = {
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": self.distributed_config.num_gpus_per_node,
            "num_nodes": self.distributed_config.num_nodes,
            "strategy": strategy,
            "precision": self.distributed_config.precision,
            "max_epochs": self.config.training.max_epochs,
            "max_steps": self.config.training.max_steps,
            "val_check_interval": self.config.training.val_check_interval,
            "num_sanity_val_steps": self.config.training.num_sanity_val_steps,
            "gradient_clip_val": self.config.training.gradient_clip_norm,
            "accumulate_grad_batches": self.config.training.accumulate_grad_batches,
            "callbacks": callbacks,
            "logger": logger_instance,
            "enable_checkpointing": True,
            "deterministic": self.config.deterministic,
            "benchmark": self.config.benchmark,
        }
        
        # Add resource limits if specified
        if self.config.max_memory_gb:
            # This would require custom memory management
            pass
        
        return Trainer(**trainer_config)
    
    def _get_distributed_strategy(self) -> Union[DDPStrategy, FSDPStrategy, DeepSpeedStrategy, str]:
        """Get the appropriate distributed strategy."""
        total_gpus = self.distributed_config.num_nodes * self.distributed_config.num_gpus_per_node
        
        if total_gpus == 1:
            return "auto"
        
        if self.distributed_config.use_fsdp:
            # Fully Sharded Data Parallel
            return FSDPStrategy(
                sharding_strategy=self.distributed_config.fsdp_sharding_strategy,
                backward_prefetch=self.distributed_config.fsdp_backward_prefetch,
                cpu_offload=False,  # Keep on GPU for performance
            )
        
        else:
            # Distributed Data Parallel
            return DDPStrategy(
                find_unused_parameters=self.distributed_config.find_unused_parameters,
                gradient_as_bucket_view=self.distributed_config.gradient_as_bucket_view,
                static_graph=self.distributed_config.static_graph,
                timeout=self.distributed_config.timeout_seconds,
            )
    
    def _create_callbacks(self) -> List:
        """Create training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.checkpoint_dir / self.experiment_name,
            filename="{epoch}-{step}-{val/accuracy:.3f}",
            monitor=self.config.training.monitor_metric,
            mode=self.config.training.mode,
            save_top_k=self.config.training.save_top_k,
            save_last=True,
            save_on_train_epoch_end=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.config.training.monitor_metric,
            mode=self.config.training.mode,
            patience=10,
            verbose=True,
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        
        # Rich progress bar
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)
        
        return callbacks
    
    def _create_logger(self) -> Union[TensorBoardLogger, WandbLogger]:
        """Create experiment logger."""
        log_dir = self.config.log_dir
        
        # Try to use Weights & Biases if available
        try:
            import wandb
            
            logger_instance = WandbLogger(
                project=self.config.project_name,
                name=self.experiment_name,
                save_dir=log_dir,
                tags=self.config.tags,
                notes=self.config.notes,
            )
            
            # Log system configuration
            logger_instance.experiment.config.update(self.config.dict())
            
        except ImportError:
            # Fallback to TensorBoard
            logger.warning("Weights & Biases not available, using TensorBoard")
            
            logger_instance = TensorBoardLogger(
                save_dir=log_dir,
                name=self.experiment_name,
            )
        
        return logger_instance
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Run distributed training."""
        logger.info(f"Starting distributed training for {self.experiment_name}")
        
        # Log system summary
        if self.trainer.is_global_zero:
            system_summary = self.system.get_system_summary()
            logger.info(f"System configuration: {system_summary}")
        
        # Start training
        self.trainer.fit(
            model=self.system,
            datamodule=self.data_module,
            ckpt_path=resume_from_checkpoint,
        )
        
        logger.info("Training completed")
    
    def test(self, test_dataloaders: Optional[List] = None) -> Dict[str, float]:
        """Run distributed testing."""
        logger.info("Starting distributed testing")
        
        # Use test dataloader from data module if not provided
        if test_dataloaders is None:
            test_results = self.trainer.test(
                model=self.system,
                datamodule=self.data_module,
            )
        else:
            test_results = self.trainer.test(
                model=self.system,
                dataloaders=test_dataloaders,
            )
        
        return test_results[0] if test_results else {}
    
    def validate(self, val_dataloaders: Optional[List] = None) -> Dict[str, float]:
        """Run distributed validation."""
        logger.info("Starting distributed validation")
        
        if val_dataloaders is None:
            val_results = self.trainer.validate(
                model=self.system,
                datamodule=self.data_module,
            )
        else:
            val_results = self.trainer.validate(
                model=self.system,
                dataloaders=val_dataloaders,
            )
        
        return val_results[0] if val_results else {}
    
    def evaluate_continual_learning(
        self, 
        task_dataloaders: List[torch.utils.data.DataLoader],
    ) -> Dict[str, float]:
        """Evaluate continual learning performance across tasks."""
        logger.info("Evaluating continual learning performance")
        
        # This should only run on the master process to avoid conflicts
        if not self.trainer.is_global_zero:
            return {}
        
        return self.system.evaluate_continual_learning(task_dataloaders)
    
    def save_system_checkpoint(self, checkpoint_path: Optional[str] = None) -> str:
        """Save complete system state including prototypes."""
        if checkpoint_path is None:
            checkpoint_path = (
                self.config.checkpoint_dir / 
                self.experiment_name / 
                f"system_checkpoint_step_{self.system.training_step_count}"
            )
        
        checkpoint_path = Path(checkpoint_path)
        
        # Only save on master process
        if self.trainer.is_global_zero:
            self.system.save_system_state(checkpoint_path)
            logger.info(f"Saved system checkpoint to {checkpoint_path}")
        
        # Synchronize across processes
        if dist.is_initialized():
            dist.barrier()
        
        return str(checkpoint_path)
    
    def load_system_checkpoint(self, checkpoint_path: str) -> None:
        """Load complete system state including prototypes."""
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            self.system.load_system_state(checkpoint_path)
            logger.info(f"Loaded system checkpoint from {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint path {checkpoint_path} does not exist")
    
    def cleanup(self) -> None:
        """Cleanup distributed training resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
        
        logger.info("Cleaned up distributed training resources")


def launch_distributed_training(
    config: SystemConfig,
    distributed_config: DistributedConfig,
    data_module: NeuromorphicDataModule,
    resume_from_checkpoint: Optional[str] = None,
) -> None:
    """
    Launch distributed training across multiple nodes/GPUs.
    
    This function handles the setup and execution of distributed training,
    including proper initialization of process groups and cleanup.
    """
    
    # Set random seeds for reproducibility
    if config.seed is not None:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    
    try:
        # Create distributed trainer
        trainer = DistributedTrainer(
            config=config,
            distributed_config=distributed_config,
            data_module=data_module,
            experiment_name=config.experiment_name,
        )
        
        # Run training
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final checkpoint
        final_checkpoint = trainer.save_system_checkpoint()
        
        # Run final evaluation if requested
        if config.evaluation.run_baselines:
            # This would involve creating baseline models and comparing
            pass
        
        logger.info(f"Distributed training completed. Final checkpoint: {final_checkpoint}")
        
    except Exception as e:
        logger.error(f"Distributed training failed: {e}")
        raise
    
    finally:
        # Cleanup
        if 'trainer' in locals():
            trainer.cleanup()


def setup_slurm_distributed() -> Dict[str, str]:
    """
    Setup environment variables for SLURM distributed training.
    
    Returns:
        Dictionary with SLURM environment variables
    """
    slurm_env = {}
    
    if "SLURM_PROCID" in os.environ:
        # Running under SLURM
        slurm_env["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "localhost")
        slurm_env["MASTER_PORT"] = "12355"
        slurm_env["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        slurm_env["RANK"] = os.environ["SLURM_PROCID"]
        slurm_env["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
        
        # Set environment variables
        for key, value in slurm_env.items():
            os.environ[key] = value
        
        logger.info(f"SLURM distributed setup: {slurm_env}")
    
    return slurm_env


def estimate_memory_requirements(
    config: SystemConfig,
    batch_size: int,
    sequence_length: int = 512,
) -> Dict[str, float]:
    """
    Estimate GPU memory requirements for the distributed system.
    
    Args:
        config: System configuration
        batch_size: Training batch size
        sequence_length: Average sequence length
        
    Returns:
        Dictionary with memory estimates in GB
    """
    
    # Model parameters
    embedding_dim = config.concept_encoder.projection_dim
    max_prototypes = config.prototype_manager.max_prototypes
    population_size = 10  # Default SNN population size
    
    # Concept encoder memory (rough estimate based on backbone)
    if config.concept_encoder.encoder_type.value == "vit":
        encoder_memory = 0.5  # GB for ViT-base
    elif config.concept_encoder.encoder_type.value == "deepseek_ocr":
        encoder_memory = 1.5  # GB (larger model)
    else:
        encoder_memory = 0.8  # GB (average)
    
    # Prototype manager memory
    prototype_memory = (
        max_prototypes * embedding_dim * 4 / (1024**3)  # Float32 centroids
        + max_prototypes * embedding_dim * embedding_dim * 4 / (1024**3)  # Covariance matrices
    )
    
    # SNN memory
    total_neurons = max_prototypes * population_size
    snn_memory = (
        total_neurons * total_neurons * 4 / (1024**3) * 2  # Recurrent + lateral weights
        + total_neurons * config.snn.num_timesteps * 4 / (1024**3)  # Spike history
    )
    
    # Answer composer memory (LLM)
    llm_memory = 1.0  # GB (rough estimate for medium model)
    
    # Batch memory
    batch_memory = (
        batch_size * 3 * 224 * 224 * 4 / (1024**3)  # Input images
        + batch_size * sequence_length * 4 / (1024**3)  # Text tokens
        + batch_size * embedding_dim * 4 / (1024**3)  # Embeddings
    )
    
    # Optimizer states (2x model parameters for Adam)
    model_params = encoder_memory + llm_memory
    optimizer_memory = model_params * 2
    
    # Total memory
    total_memory = (
        encoder_memory +
        prototype_memory +
        snn_memory +
        llm_memory +
        batch_memory +
        optimizer_memory +
        1.0  # Buffer for other operations
    )
    
    return {
        "encoder_memory_gb": encoder_memory,
        "prototype_memory_gb": prototype_memory,
        "snn_memory_gb": snn_memory,
        "llm_memory_gb": llm_memory,
        "batch_memory_gb": batch_memory,
        "optimizer_memory_gb": optimizer_memory,
        "total_memory_gb": total_memory,
        "recommended_gpu_memory_gb": total_memory * 1.2,  # 20% buffer
    }
