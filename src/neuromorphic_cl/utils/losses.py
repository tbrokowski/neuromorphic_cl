"""
Loss Functions for Neuromorphic Continual Learning.

This module implements various loss functions used in the system,
including contrastive losses, prototype alignment losses, and
specialized losses for continual learning scenarios.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Information Noise Contrastive Estimation) loss.
    
    Used for contrastive learning between visual and textual representations.
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self, 
        embeddings1: torch.Tensor, 
        embeddings2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings [B, D]
            embeddings2: Second set of embeddings [B, D]
            labels: Optional labels for positive pairs [B]
            
        Returns:
            InfoNCE loss
        """
        batch_size = embeddings1.size(0)
        
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=-1)
        embeddings2 = F.normalize(embeddings2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings1, embeddings2.t()) / self.temperature
        
        # Create labels for contrastive learning
        if labels is None:
            # Assume positive pairs are along the diagonal
            labels = torch.arange(batch_size, device=embeddings1.device)
        
        # Compute loss
        loss_i2t = F.cross_entropy(similarity_matrix, labels, reduction=self.reduction)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels, reduction=self.reduction)
        
        return (loss_i2t + loss_t2i) / 2


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for learning representations with class labels.
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Feature embeddings [B, D]
            labels: Class labels [B]
            mask: Optional mask for valid samples [B]
            
        Returns:
            Supervised contrastive loss
        """
        batch_size = features.size(0)
        
        if mask is None:
            mask = torch.ones(batch_size, device=features.device).bool()
        
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.t()),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.t()).float()
        
        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask_pos),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask_pos = mask_pos * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / mask_pos.sum(1)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size)[mask].mean()
        
        return loss


class PrototypeAlignmentLoss(nn.Module):
    """
    Loss for aligning embeddings with their assigned prototypes.
    """
    
    def __init__(self, margin: float = 0.5, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        prototype_ids: torch.Tensor,
        prototype_manager: "PrototypeManager",
    ) -> torch.Tensor:
        """
        Compute prototype alignment loss.
        
        Args:
            embeddings: Input embeddings [B, D]
            prototype_ids: Assigned prototype IDs [B]
            prototype_manager: Prototype manager instance
            
        Returns:
            Prototype alignment loss
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # Get prototype centroids
        prototype_centroids = []
        valid_mask = []
        
        for i, proto_id in enumerate(prototype_ids):
            prototype = prototype_manager.get_prototype(proto_id.item())
            if prototype is not None:
                centroid = torch.from_numpy(prototype.centroid).to(device)
                prototype_centroids.append(centroid)
                valid_mask.append(True)
            else:
                # Create dummy centroid for invalid prototypes
                prototype_centroids.append(torch.zeros_like(embeddings[0]))
                valid_mask.append(False)
        
        if not any(valid_mask):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        prototype_centroids = torch.stack(prototype_centroids)  # [B, D]
        valid_mask = torch.tensor(valid_mask, device=device)
        
        # Compute distances to assigned prototypes
        assigned_distances = F.mse_loss(
            embeddings[valid_mask], 
            prototype_centroids[valid_mask],
            reduction='none'
        ).mean(dim=-1)
        
        # Compute distances to other prototypes (negative samples)
        all_prototypes = prototype_manager.get_all_prototypes()
        if len(all_prototypes) > 1:
            # Sample some negative prototypes
            neg_centroids = []
            for prototype in all_prototypes[:min(10, len(all_prototypes))]:
                neg_centroids.append(torch.from_numpy(prototype.centroid).to(device))
            
            neg_centroids = torch.stack(neg_centroids)  # [N, D]
            
            # Compute distances to negative prototypes
            embeddings_expanded = embeddings[valid_mask].unsqueeze(1)  # [B_valid, 1, D]
            neg_centroids_expanded = neg_centroids.unsqueeze(0)  # [1, N, D]
            
            neg_distances = F.mse_loss(
                embeddings_expanded, 
                neg_centroids_expanded,
                reduction='none'
            ).mean(dim=-1)  # [B_valid, N]
            
            min_neg_distances = neg_distances.min(dim=-1)[0]  # [B_valid]
            
            # Margin loss: assigned distance should be smaller than negative distance
            margin_loss = F.relu(assigned_distances - min_neg_distances + self.margin)
            
            return margin_loss.mean()
        else:
            # Only alignment loss if no negative samples
            return assigned_distances.mean()


class PullPushLoss(nn.Module):
    """
    Pull-push loss for prototype learning.
    
    Pulls embeddings towards their assigned prototype centroid and
    pushes them away from other prototype centroids.
    """
    
    def __init__(self, pull_weight: float = 1.0, push_weight: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.margin = margin
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        prototype_centroids: torch.Tensor,
        assignment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pull-push loss.
        
        Args:
            embeddings: Input embeddings [B, D]
            prototype_centroids: Prototype centroids [P, D]
            assignment_ids: Prototype assignment for each embedding [B]
            
        Returns:
            Pull-push loss
        """
        batch_size, embed_dim = embeddings.shape
        num_prototypes = prototype_centroids.size(0)
        device = embeddings.device
        
        # Pull loss: attract embeddings to assigned prototypes
        assigned_centroids = prototype_centroids[assignment_ids]  # [B, D]
        pull_loss = F.mse_loss(embeddings, assigned_centroids, reduction='mean')
        
        # Push loss: repel from non-assigned prototypes
        push_loss = 0.0
        
        if num_prototypes > 1:
            # Compute distances to all prototypes
            embeddings_expanded = embeddings.unsqueeze(1)  # [B, 1, D]
            centroids_expanded = prototype_centroids.unsqueeze(0)  # [1, P, D]
            
            distances = torch.norm(
                embeddings_expanded - centroids_expanded, 
                dim=-1
            )  # [B, P]
            
            # Create mask for non-assigned prototypes
            assignment_mask = torch.zeros(batch_size, num_prototypes, device=device)
            assignment_mask[torch.arange(batch_size), assignment_ids] = 1
            non_assigned_mask = 1 - assignment_mask
            
            # Push loss: maximize distance to non-assigned prototypes
            # Use margin-based loss
            assigned_distances = distances[assignment_mask.bool()]
            non_assigned_distances = distances * non_assigned_mask
            
            # Only consider non-assigned prototypes that are too close
            min_non_assigned_dist = non_assigned_distances.max(dim=-1)[0]  # Max to ignore zeros
            push_violations = F.relu(self.margin - min_non_assigned_dist)
            push_loss = push_violations.mean()
        
        total_loss = self.pull_weight * pull_loss + self.push_weight * push_loss
        
        return total_loss


class ClipSupConLoss(nn.Module):
    """
    CLIP-style supervised contrastive loss for multimodal learning.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute CLIP-style contrastive loss.
        
        Args:
            image_features: Image embeddings [B, D]
            text_features: Text embeddings [B, D]
            labels: Optional labels for supervision [B]
            
        Returns:
            CLIP contrastive loss
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()
        
        batch_size = image_features.size(0)
        
        if labels is None:
            # Standard CLIP loss (diagonal as positive pairs)
            targets = torch.arange(batch_size, device=image_features.device)
        else:
            # Supervised version: positive pairs have same label
            targets = labels
        
        # Symmetric loss
        loss_i2t = F.cross_entropy(logits_per_image, targets)
        loss_t2i = F.cross_entropy(logits_per_text, targets)
        
        return (loss_i2t + loss_t2i) / 2


class ForgettingRegularizationLoss(nn.Module):
    """
    Regularization loss to prevent catastrophic forgetting.
    
    Based on Elastic Weight Consolidation (EWC) principles.
    """
    
    def __init__(self, importance_weight: float = 1000.0):
        super().__init__()
        self.importance_weight = importance_weight
        self.fisher_information = {}
        self.optimal_params = {}
    
    def update_fisher_information(self, model: nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Compute and store Fisher Information Matrix for important parameters.
        
        Args:
            model: Model to compute Fisher information for
            dataloader: Dataloader to estimate Fisher information
        """
        fisher_dict = {}
        
        model.eval()
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param)
        
        num_samples = 0
        for batch in dataloader:
            model.zero_grad()
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.get('loss', 0)
            
            # Backward pass
            loss.backward()
            
            # Accumulate squared gradients (Fisher Information approximation)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
            
            num_samples += 1
            
            if num_samples >= 100:  # Limit computation
                break
        
        # Normalize by number of samples
        for name in fisher_dict:
            fisher_dict[name] /= num_samples
        
        self.fisher_information = fisher_dict
        
        # Store current optimal parameters
        self.optimal_params = {
            name: param.data.clone() 
            for name, param in model.named_parameters() 
            if param.requires_grad
        }
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC regularization loss.
        
        Args:
            model: Current model
            
        Returns:
            EWC regularization loss
        """
        if not self.fisher_information:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        ewc_loss = 0.0
        
        for name, param in model.named_parameters():
            if name in self.fisher_information and param.requires_grad:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                
                # EWC loss: Fisher-weighted parameter deviation
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
        
        return self.importance_weight * ewc_loss


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss for transferring knowledge from teacher to student.
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor,
        true_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits: Student model predictions [B, C]
            teacher_logits: Teacher model predictions [B, C]
            true_labels: Optional true labels [B]
            
        Returns:
            Distillation loss
        """
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence loss
        distillation_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        if true_labels is not None:
            # Combine with standard cross-entropy
            ce_loss = F.cross_entropy(student_logits, true_labels)
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * ce_loss
            return total_loss
        else:
            return distillation_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with automatic weight balancing.
    """
    
    def __init__(self, task_names: list, initial_weights: Optional[dict] = None):
        super().__init__()
        self.task_names = task_names
        self.num_tasks = len(task_names)
        
        # Initialize task weights
        if initial_weights:
            weights = [initial_weights.get(name, 1.0) for name in task_names]
        else:
            weights = [1.0] * self.num_tasks
        
        self.log_vars = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, task_losses: dict) -> Tuple[torch.Tensor, dict]:
        """
        Compute weighted multi-task loss.
        
        Args:
            task_losses: Dictionary of task losses
            
        Returns:
            Tuple of (total_loss, weight_dict)
        """
        total_loss = 0.0
        weights = {}
        
        for i, task_name in enumerate(self.task_names):
            if task_name in task_losses:
                # Automatic weight balancing using uncertainty
                weight = torch.exp(-self.log_vars[i])
                weighted_loss = weight * task_losses[task_name] + self.log_vars[i]
                
                total_loss += weighted_loss
                weights[task_name] = weight.item()
        
        return total_loss, weights


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple loss functions with configurable weights.
    """
    
    def __init__(self, loss_functions: dict, loss_weights: dict):
        super().__init__()
        self.loss_functions = nn.ModuleDict(loss_functions)
        self.loss_weights = loss_weights
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, dict]:
        """
        Compute composite loss.
        
        Returns:
            Tuple of (total_loss, individual_losses)
        """
        total_loss = 0.0
        individual_losses = {}
        
        for loss_name, loss_fn in self.loss_functions.items():
            if loss_name in self.loss_weights:
                try:
                    loss_value = loss_fn(*args, **kwargs)
                    weighted_loss = self.loss_weights[loss_name] * loss_value
                    
                    total_loss += weighted_loss
                    individual_losses[loss_name] = loss_value.item()
                    
                except Exception as e:
                    logger.warning(f"Failed to compute {loss_name} loss: {e}")
                    individual_losses[loss_name] = 0.0
        
        return total_loss, individual_losses
