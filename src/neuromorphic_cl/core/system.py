"""
Main Neuromorphic Continual Learning System.

This module integrates all components (Concept Encoder, Prototype Manager,
Spiking Neural Network, and Answer Composer) into a unified system for
continual learning with neuromorphic memory.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from ..configs.schema import SystemConfig
from ..core.answer_composer import AnswerComposer
from ..core.concept_encoder import ConceptEncoder
from ..core.prototype_manager import PrototypeManager
from ..core.snn import SpikingNeuralNetwork
from ..utils.losses import (
    ClipSupConLoss,
    InfoNCELoss,
    PrototypeAlignmentLoss,
    PullPushLoss,
    SupervisedContrastiveLoss,
)
from ..utils.metrics import (
    accuracy,
    compute_forgetting,
    compute_transfer,
    energy_efficiency,
    memory_efficiency,
)

logger = logging.getLogger(__name__)


class NeuromorphicContinualLearningSystem(LightningModule):
    """
    Main system integrating all neuromorphic continual learning components.
    
    This system performs end-to-end continual learning by:
    1. Encoding multimodal inputs into concept embeddings
    2. Dynamically clustering concepts into prototypes  
    3. Storing and retrieving memories via spiking neural networks
    4. Generating responses through evidence-based composition
    """
    
    def __init__(self, config: SystemConfig):
        super().__init__()
        
        self.config = config
        self.save_hyperparameters(config.dict())
        
        # Initialize components
        logger.info("Initializing Neuromorphic Continual Learning System...")
        
        # 1. Concept Encoder
        self.concept_encoder = ConceptEncoder(config.concept_encoder)
        embedding_dim = self.concept_encoder.get_embedding_dim()
        
        # 2. Prototype Manager
        self.prototype_manager = PrototypeManager(
            config.prototype_manager, 
            embedding_dim
        )
        
        # 3. Spiking Neural Network
        self.snn = SpikingNeuralNetwork(
            config.snn,
            max_prototypes=config.prototype_manager.max_prototypes,
        )
        
        # 4. Answer Composer
        self.answer_composer = AnswerComposer(
            config.answer_composer,
            prototype_dim=embedding_dim,
        )
        
        # Loss functions
        self.losses = self._initialize_losses()
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        self.task_memories = {}  # Store task-specific information
        
        # Training state
        self.current_task = 0
        self.training_step_count = 0
        self.last_maintenance_step = 0
        
        logger.info(
            f"Initialized system with {embedding_dim}D embeddings, "
            f"max {config.prototype_manager.max_prototypes} prototypes"
        )
    
    def _initialize_losses(self) -> Dict[str, nn.Module]:
        """Initialize loss functions based on configuration."""
        losses = {}
        
        if "infonce" in self.config.training.loss_weights:
            losses["infonce"] = InfoNCELoss(
                temperature=self.config.concept_encoder.contrastive_temperature
            )
        
        if "supervised_contrastive" in self.config.training.loss_weights:
            losses["supervised_contrastive"] = SupervisedContrastiveLoss(
                temperature=self.config.concept_encoder.contrastive_temperature
            )
        
        if "prototype_alignment" in self.config.training.loss_weights:
            losses["prototype_alignment"] = PrototypeAlignmentLoss()
        
        if "clip_supcon" in self.config.training.loss_weights:
            losses["clip_supcon"] = ClipSupConLoss(
                temperature=self.config.concept_encoder.contrastive_temperature
            )
        
        if "pull_push" in self.config.training.loss_weights:
            losses["pull_push"] = PullPushLoss()
        
        return nn.ModuleDict(losses)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        task_type: str = "text_generation",
        query_text: Optional[str] = None,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete system.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            text_tokens: Optional text tokens [B, T]
            attention_mask: Optional attention mask [B, T]
            labels: Optional labels for supervised tasks
            task_type: Type of task for answer composition
            query_text: Optional query text for generation
            return_intermediate: Whether to return intermediate representations
            
        Returns:
            Dictionary with system outputs and intermediate representations
        """
        batch_size = pixel_values.size(0)
        
        # 1. Concept Encoding
        encoder_outputs = self.concept_encoder(
            pixel_values=pixel_values,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            return_saliency=True,
            return_text_embedding=text_tokens is not None,
        )
        
        concept_embeddings = encoder_outputs["concept_embedding"]  # [B, D]
        
        # 2. Prototype Assignment and Management
        prototype_ids = self.prototype_manager(concept_embeddings)  # [B]
        
        # 3. SNN Memory Dynamics
        snn_outputs = self.snn(
            prototype_ids=prototype_ids,
            return_full_output=return_intermediate,
        )
        
        active_basin = snn_outputs["active_basin"]  # [B, num_prototypes]
        
        # 4. Answer Composition
        if task_type and len(torch.unique(prototype_ids)) > 0:
            # Only compose answers if we have valid prototypes
            answer_outputs = self.answer_composer(
                query_embedding=concept_embeddings,
                active_basin=active_basin,
                prototype_manager=self.prototype_manager,
                task_type=task_type,
                query_text=query_text,
                num_classes=labels.max().item() + 1 if labels is not None else None,
            )
        else:
            # Return empty outputs if no valid prototypes
            answer_outputs = {"should_abstain": torch.ones(batch_size, dtype=torch.bool)}
        
        # Combine outputs
        outputs = {
            "concept_embeddings": concept_embeddings,
            "prototype_ids": prototype_ids,
            "active_basin": active_basin,
            **answer_outputs,
        }
        
        if return_intermediate:
            outputs.update({
                "encoder_outputs": encoder_outputs,
                "snn_outputs": snn_outputs,
            })
        
        return outputs
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with multi-component loss computation."""
        self.training_step_count += 1
        
        # Extract batch data
        pixel_values = batch["pixel_values"]
        text_tokens = batch.get("text_tokens")
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        task_type = batch.get("task_type", "text_generation")
        
        # Forward pass
        outputs = self.forward(
            pixel_values=pixel_values,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            labels=labels,
            task_type=task_type,
            return_intermediate=True,
        )
        
        # Compute losses
        total_loss = 0.0
        loss_dict = {}
        
        concept_embeddings = outputs["concept_embeddings"]
        encoder_outputs = outputs["encoder_outputs"]
        
        # Contrastive losses
        if "infonce" in self.losses and text_tokens is not None:
            text_embeddings = encoder_outputs.get("text_embedding")
            if text_embeddings is not None:
                infonce_loss = self.losses["infonce"](
                    concept_embeddings, text_embeddings
                )
                loss_dict["infonce"] = infonce_loss
                total_loss += self.config.training.loss_weights["infonce"] * infonce_loss
        
        if "supervised_contrastive" in self.losses and labels is not None:
            supcon_loss = self.losses["supervised_contrastive"](
                concept_embeddings, labels
            )
            loss_dict["supervised_contrastive"] = supcon_loss
            total_loss += (
                self.config.training.loss_weights["supervised_contrastive"] * supcon_loss
            )
        
        # Prototype alignment loss
        if "prototype_alignment" in self.losses:
            prototype_ids = outputs["prototype_ids"]
            alignment_loss = self.losses["prototype_alignment"](
                concept_embeddings, prototype_ids, self.prototype_manager
            )
            loss_dict["prototype_alignment"] = alignment_loss
            total_loss += (
                self.config.training.loss_weights["prototype_alignment"] * alignment_loss
            )
        
        # Task-specific losses
        if task_type == "classification" and labels is not None:
            if "logits" in outputs:
                classification_loss = F.cross_entropy(outputs["logits"], labels)
                loss_dict["classification"] = classification_loss
                total_loss += classification_loss
        
        # SNN learning (STDP is handled internally)
        prototype_ids = outputs["prototype_ids"]
        for i, proto_id in enumerate(prototype_ids):
            if proto_id.item() >= 0:  # Valid prototype
                neighbors = self.prototype_manager.get_neighbors(proto_id.item())
                reward = 1.0 if labels is None else (labels[i] == outputs.get("predictions", labels)[i]).float().item()
                
                self.snn.learn(
                    prototype_id=proto_id.item(),
                    neighbors=neighbors,
                    reward=reward,
                )
        
        # Logging
        self.log("train/total_loss", total_loss, prog_bar=True)
        for loss_name, loss_value in loss_dict.items():
            self.log(f"train/{loss_name}", loss_value)
        
        # Periodic maintenance
        if (self.training_step_count - self.last_maintenance_step >= 
            self.config.prototype_manager.maintenance_interval):
            
            maintenance_stats = self.prototype_manager.maintain()
            self.snn.consolidate()
            
            self.log_dict({
                f"maintenance/{k}": v for k, v in maintenance_stats.items()
            })
            self.last_maintenance_step = self.training_step_count
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step with comprehensive metrics."""
        # Extract batch data
        pixel_values = batch["pixel_values"]
        text_tokens = batch.get("text_tokens")
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        task_type = batch.get("task_type", "classification")
        
        # Forward pass
        outputs = self.forward(
            pixel_values=pixel_values,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            labels=labels,
            task_type=task_type,
        )
        
        # Compute metrics
        metrics = {}
        
        if task_type == "classification" and labels is not None:
            if "predictions" in outputs and not outputs["should_abstain"].all():
                # Only compute accuracy on non-abstained examples
                valid_mask = ~outputs["should_abstain"]
                if valid_mask.any():
                    valid_predictions = outputs["predictions"][valid_mask]
                    valid_labels = labels[valid_mask]
                    
                    acc = accuracy(valid_predictions, valid_labels)
                    metrics["accuracy"] = acc
                    
                    # Coverage (fraction of examples answered)
                    coverage = valid_mask.float().mean()
                    metrics["coverage"] = coverage
        
        # Memory efficiency
        prototype_stats = self.prototype_manager.get_statistics()
        metrics["num_prototypes"] = prototype_stats["total_prototypes"]
        metrics["avg_prototype_size"] = prototype_stats.get("avg_prototype_size", 0)
        
        # Energy efficiency (from SNN)
        snn_stats = self.snn.get_memory_statistics()
        metrics["energy_efficiency"] = snn_stats.get("total_energy_pj", 0)
        
        self.log_dict({f"val/{k}": v for k, v in metrics.items()})
        
        return metrics
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step for final evaluation."""
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizers for different components."""
        # Separate optimizers for different components
        optimizers = []
        schedulers = []
        
        # Main optimizer for encoders and answer composer
        main_params = list(self.concept_encoder.parameters()) + list(self.answer_composer.parameters())
        
        if self.config.training.optimizer == "adamw":
            main_optimizer = torch.optim.AdamW(
                main_params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "adam":
            main_optimizer = torch.optim.Adam(
                main_params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        optimizers.append(main_optimizer)
        
        # SNN optimizer (typically lower learning rate)
        snn_optimizer = torch.optim.Adam(
            self.snn.parameters(),
            lr=self.config.training.learning_rate * 0.1,  # Lower LR for SNN
            weight_decay=self.config.training.weight_decay,
        )
        optimizers.append(snn_optimizer)
        
        # Schedulers
        if self.config.training.scheduler == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                main_optimizer,
                T_max=self.config.training.max_epochs,
            )
            snn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                snn_optimizer,
                T_max=self.config.training.max_epochs,
            )
            schedulers.extend([main_scheduler, snn_scheduler])
        
        if schedulers:
            return optimizers, schedulers
        else:
            return optimizers
    
    def on_task_switch(self, new_task_id: int, task_metadata: Optional[Dict] = None) -> None:
        """Handle switching to a new continual learning task."""
        logger.info(f"Switching to task {new_task_id}")
        
        # Store previous task statistics
        if self.current_task in self.task_memories:
            prev_stats = self.prototype_manager.get_statistics()
            self.task_memories[self.current_task].update({
                "end_prototypes": prev_stats["total_prototypes"],
                "final_step": self.training_step_count,
            })
        
        # Initialize new task memory
        self.task_memories[new_task_id] = {
            "start_step": self.training_step_count,
            "start_prototypes": self.prototype_manager.get_statistics()["total_prototypes"],
            "metadata": task_metadata or {},
        }
        
        self.current_task = new_task_id
        
        # Trigger memory consolidation for important prototypes
        important_prototypes = self._select_important_prototypes()
        self.snn.consolidate(important_prototypes)
    
    def _select_important_prototypes(self, top_k: int = 50) -> List[int]:
        """Select important prototypes for consolidation."""
        prototype_stats = self.prototype_manager.get_statistics()
        if prototype_stats["total_prototypes"] == 0:
            return []
        
        # Get prototypes sorted by usage count
        prototypes = self.prototype_manager.get_all_prototypes()
        prototypes.sort(key=lambda p: p.count, reverse=True)
        
        # Return top-k most used prototypes
        important_ids = [p.id for p in prototypes[:min(top_k, len(prototypes))]]
        
        return important_ids
    
    def evaluate_continual_learning(
        self, 
        test_dataloaders: List[torch.utils.data.DataLoader],
    ) -> Dict[str, float]:
        """
        Evaluate continual learning performance across all tasks.
        
        Args:
            test_dataloaders: List of test dataloaders for each task
            
        Returns:
            Dictionary with continual learning metrics
        """
        self.eval()
        
        task_accuracies = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for task_id, dataloader in enumerate(test_dataloaders):
                task_predictions = []
                task_labels = []
                
                for batch in dataloader:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    outputs = self.forward(
                        pixel_values=batch["pixel_values"],
                        text_tokens=batch.get("text_tokens"),
                        attention_mask=batch.get("attention_mask"),
                        labels=batch.get("labels"),
                        task_type="classification",
                    )
                    
                    if "predictions" in outputs and batch.get("labels") is not None:
                        # Only consider non-abstained predictions
                        valid_mask = ~outputs["should_abstain"]
                        if valid_mask.any():
                            task_predictions.append(outputs["predictions"][valid_mask])
                            task_labels.append(batch["labels"][valid_mask])
                
                if task_predictions:
                    task_preds = torch.cat(task_predictions)
                    task_lbls = torch.cat(task_labels)
                    
                    task_acc = accuracy(task_preds, task_lbls)
                    task_accuracies.append(task_acc.item())
                    
                    all_predictions.append(task_preds)
                    all_labels.append(task_lbls)
                else:
                    task_accuracies.append(0.0)
        
        # Compute continual learning metrics
        metrics = {
            "average_accuracy": np.mean(task_accuracies) if task_accuracies else 0.0,
            "final_accuracy": task_accuracies[-1] if task_accuracies else 0.0,
        }
        
        if len(task_accuracies) > 1:
            # Forgetting: how much accuracy dropped on previous tasks
            forgetting = compute_forgetting(task_accuracies[:-1])
            metrics["forgetting"] = forgetting
            
            # Forward transfer: improvement on new tasks due to previous learning
            # (This would require baseline single-task performance for comparison)
            
            # Backward transfer: improvement on old tasks due to new learning
            # (Also requires tracking accuracy changes over time)
        
        # Memory efficiency
        prototype_stats = self.prototype_manager.get_statistics()
        metrics["memory_efficiency"] = memory_efficiency(
            num_prototypes=prototype_stats["total_prototypes"],
            embedding_dim=self.concept_encoder.get_embedding_dim(),
        )
        
        # Energy efficiency
        snn_stats = self.snn.get_memory_statistics()
        metrics["energy_efficiency"] = energy_efficiency(
            total_energy=snn_stats.get("total_energy_pj", 0),
            num_inferences=sum(len(dl) for dl in test_dataloaders),
        )
        
        return metrics
    
    def save_system_state(self, checkpoint_path: Union[str, Path]) -> None:
        """Save complete system state including prototypes and SNN connections."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch Lightning checkpoint
        self.trainer.save_checkpoint(checkpoint_path / "model.ckpt")
        
        # Save prototype manager state
        self.prototype_manager.save(checkpoint_path / "prototypes.pkl")
        
        # Save SNN connections
        self.snn.save_connections(checkpoint_path / "snn_connections.pt")
        
        # Save task memories
        torch.save(self.task_memories, checkpoint_path / "task_memories.pt")
        
        logger.info(f"Saved system state to {checkpoint_path}")
    
    def load_system_state(self, checkpoint_path: Union[str, Path]) -> None:
        """Load complete system state including prototypes and SNN connections."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load prototype manager state
        if (checkpoint_path / "prototypes.pkl").exists():
            self.prototype_manager.load(checkpoint_path / "prototypes.pkl")
        
        # Load SNN connections
        if (checkpoint_path / "snn_connections.pt").exists():
            self.snn.load_connections(checkpoint_path / "snn_connections.pt")
        
        # Load task memories
        if (checkpoint_path / "task_memories.pt").exists():
            self.task_memories = torch.load(
                checkpoint_path / "task_memories.pt",
                map_location=self.device,
            )
        
        logger.info(f"Loaded system state from {checkpoint_path}")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the system state."""
        prototype_stats = self.prototype_manager.get_statistics()
        snn_stats = self.snn.get_memory_statistics()
        
        return {
            "config": self.config.dict(),
            "current_task": self.current_task,
            "training_steps": self.training_step_count,
            "prototype_stats": prototype_stats,
            "snn_stats": snn_stats,
            "task_memories": self.task_memories,
            "component_info": {
                "concept_encoder": {
                    "type": self.config.concept_encoder.encoder_type,
                    "embedding_dim": self.concept_encoder.get_embedding_dim(),
                },
                "prototype_manager": {
                    "backend": self.config.prototype_manager.indexing_backend,
                    "max_prototypes": self.config.prototype_manager.max_prototypes,
                },
                "snn": {
                    "neuron_type": self.config.snn.neuron_type,
                    "timesteps": self.config.snn.num_timesteps,
                },
                "answer_composer": {
                    "llm_model": self.config.answer_composer.llm_model_name,
                    "evidence_slots": self.config.answer_composer.evidence_slate_size,
                },
            },
        }
