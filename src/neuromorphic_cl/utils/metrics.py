"""
Metrics and Evaluation Utilities for Neuromorphic Continual Learning.

This module provides comprehensive metrics for evaluating the system's
performance across different tasks and continual learning scenarios.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def accuracy(predictions: torch.Tensor, targets: torch.Tensor, top_k: int = 1) -> torch.Tensor:
    """
    Compute top-k accuracy.
    
    Args:
        predictions: Model predictions [B, C] or [B]
        targets: Ground truth labels [B]
        top_k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy as tensor
    """
    if predictions.dim() == 1:
        # Binary or single predictions
        correct = (predictions == targets).float()
        return correct.mean()
    
    # Multi-class predictions
    batch_size = targets.size(0)
    
    if top_k == 1:
        pred_labels = predictions.argmax(dim=-1)
        correct = (pred_labels == targets).float()
        return correct.mean()
    else:
        # Top-k accuracy
        _, top_k_preds = predictions.topk(top_k, dim=-1)
        targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=-1).float()
        return correct.mean()


def precision_recall_f1(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    average: str = "macro",
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1-score.
    
    Args:
        predictions: Model predictions [B, C] or [B]
        targets: Ground truth labels [B]
        average: Averaging strategy ('macro', 'micro', 'weighted')
        num_classes: Number of classes (for binary case)
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        if predictions.dim() > 1:
            pred_labels = predictions.argmax(dim=-1).cpu().numpy()
        else:
            pred_labels = predictions.cpu().numpy()
    else:
        pred_labels = predictions
    
    if isinstance(targets, torch.Tensor):
        true_labels = targets.cpu().numpy()
    else:
        true_labels = targets
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average=average, zero_division=0
    )
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def multilabel_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute metrics for multi-label classification.
    
    Args:
        predictions: Model predictions [B, C] (logits or probabilities)
        targets: Ground truth binary labels [B, C]
        threshold: Decision threshold
        
    Returns:
        Dictionary with multi-label metrics
    """
    # Convert logits to probabilities if needed
    if predictions.max() > 1.0 or predictions.min() < 0.0:
        predictions = torch.sigmoid(predictions)
    
    # Apply threshold
    pred_binary = (predictions > threshold).float()
    
    # Convert to numpy
    pred_np = pred_binary.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Compute metrics
    metrics = {}
    
    # Per-sample accuracy (exact match)
    exact_match = (pred_np == targets_np).all(axis=1).mean()
    metrics["exact_match_ratio"] = float(exact_match)
    
    # Hamming loss (fraction of wrong labels)
    hamming = (pred_np != targets_np).mean()
    metrics["hamming_loss"] = float(hamming)
    
    # Macro-averaged F1
    f1_scores = []
    for i in range(targets_np.shape[1]):
        f1 = f1_score(targets_np[:, i], pred_np[:, i], zero_division=0)
        f1_scores.append(f1)
    
    metrics["macro_f1"] = float(np.mean(f1_scores))
    
    # Micro-averaged F1
    micro_f1 = f1_score(targets_np.ravel(), pred_np.ravel(), zero_division=0)
    metrics["micro_f1"] = float(micro_f1)
    
    # AUC for each class
    try:
        auc_scores = []
        for i in range(targets_np.shape[1]):
            if len(np.unique(targets_np[:, i])) > 1:  # Check if both classes present
                auc = roc_auc_score(targets_np[:, i], predictions[:, i].cpu().numpy())
                auc_scores.append(auc)
        
        if auc_scores:
            metrics["macro_auc"] = float(np.mean(auc_scores))
    except Exception as e:
        logger.warning(f"Failed to compute AUC: {e}")
        metrics["macro_auc"] = 0.0
    
    return metrics


def cosine_similarity(x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        x: First vector
        y: Second vector
        
    Returns:
        Cosine similarity score
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    # Flatten if needed
    x = x.flatten()
    y = y.flatten()
    
    # Compute cosine similarity
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    
    if norm_x == 0 or norm_y == 0:
        return 0.0
    
    return dot_product / (norm_x * norm_y)


def euclidean_distance(x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Args:
        x: First vector
        y: Second vector
        
    Returns:
        Euclidean distance
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    return float(np.linalg.norm(x.flatten() - y.flatten()))


def compute_forgetting(task_accuracies: List[float]) -> float:
    """
    Compute catastrophic forgetting metric.
    
    Args:
        task_accuracies: List of accuracies for previous tasks
        
    Returns:
        Average forgetting (higher means more forgetting)
    """
    if len(task_accuracies) <= 1:
        return 0.0
    
    # Compute forgetting as average accuracy drop
    forgetting_scores = []
    
    for i in range(len(task_accuracies) - 1):
        initial_acc = task_accuracies[i]
        final_acc = task_accuracies[-1]  # Current performance on old task
        
        if initial_acc > 0:
            forgetting = max(0, initial_acc - final_acc)
            forgetting_scores.append(forgetting)
    
    return np.mean(forgetting_scores) if forgetting_scores else 0.0


def compute_transfer(
    single_task_accuracies: List[float], 
    continual_accuracies: List[float],
) -> Dict[str, float]:
    """
    Compute forward and backward transfer metrics.
    
    Args:
        single_task_accuracies: Accuracies when training each task independently
        continual_accuracies: Accuracies when training continually
        
    Returns:
        Dictionary with transfer metrics
    """
    if len(single_task_accuracies) != len(continual_accuracies):
        logger.warning("Mismatched lengths for transfer computation")
        return {"forward_transfer": 0.0, "backward_transfer": 0.0}
    
    num_tasks = len(single_task_accuracies)
    
    # Forward transfer: improvement on new tasks due to previous learning
    forward_transfers = []
    for i in range(1, num_tasks):
        single_acc = single_task_accuracies[i]
        continual_acc = continual_accuracies[i]
        
        if single_acc > 0:
            transfer = (continual_acc - single_acc) / single_acc
            forward_transfers.append(transfer)
    
    forward_transfer = np.mean(forward_transfers) if forward_transfers else 0.0
    
    # Backward transfer: improvement on old tasks due to new learning
    # (This would require tracking accuracy changes over time)
    backward_transfer = 0.0  # Simplified for now
    
    return {
        "forward_transfer": float(forward_transfer),
        "backward_transfer": float(backward_transfer),
    }


def memory_efficiency(num_prototypes: int, embedding_dim: int, baseline_memory: Optional[float] = None) -> float:
    """
    Compute memory efficiency metric.
    
    Args:
        num_prototypes: Number of prototypes stored
        embedding_dim: Embedding dimension
        baseline_memory: Baseline memory usage for comparison
        
    Returns:
        Memory efficiency score
    """
    # Compute prototype memory usage (in MB)
    prototype_memory = (
        num_prototypes * embedding_dim * 4 / (1024 * 1024)  # Float32 centroids
        + num_prototypes * embedding_dim * embedding_dim * 4 / (1024 * 1024)  # Covariance
    )
    
    if baseline_memory is not None:
        # Efficiency relative to baseline
        return baseline_memory / max(prototype_memory, 1e-6)
    else:
        # Inverse of memory usage (higher is better)
        return 1.0 / max(prototype_memory, 1e-6)


def energy_efficiency(total_energy: float, num_inferences: int) -> float:
    """
    Compute energy efficiency metric.
    
    Args:
        total_energy: Total energy consumption (in picojoules)
        num_inferences: Number of inferences performed
        
    Returns:
        Energy per inference (lower is better)
    """
    if num_inferences == 0:
        return float('inf')
    
    return total_energy / num_inferences


def spike_count(spike_tensor: torch.Tensor) -> int:
    """
    Count total number of spikes in a tensor.
    
    Args:
        spike_tensor: Tensor containing spike data
        
    Returns:
        Total spike count
    """
    return int(spike_tensor.sum().item())


def spike_rate(spike_tensor: torch.Tensor, time_window: float = 1.0) -> float:
    """
    Compute average spike rate.
    
    Args:
        spike_tensor: Tensor containing spike data [T, B, N]
        time_window: Time window for rate computation
        
    Returns:
        Average spike rate (spikes per unit time per neuron)
    """
    if spike_tensor.numel() == 0:
        return 0.0
    
    total_spikes = spike_tensor.sum().item()
    num_neurons = spike_tensor.size(-1) if spike_tensor.dim() > 0 else 1
    num_timesteps = spike_tensor.size(0) if spike_tensor.dim() > 2 else 1
    
    # Rate = spikes / (time * neurons)
    rate = total_spikes / (time_window * num_neurons * num_timesteps)
    
    return float(rate)


class ContinualLearningMetrics:
    """
    Comprehensive metrics tracker for continual learning evaluation.
    """
    
    def __init__(self):
        self.task_results = {}  # Store results for each task
        self.current_task = 0
        
    def add_task_result(
        self, 
        task_id: int, 
        accuracy: float, 
        additional_metrics: Optional[Dict] = None,
    ) -> None:
        """Add results for a specific task."""
        self.task_results[task_id] = {
            "accuracy": accuracy,
            "step": self.current_task,
            **(additional_metrics or {}),
        }
        
    def compute_summary_metrics(self) -> Dict[str, float]:
        """Compute summary metrics across all tasks."""
        if not self.task_results:
            return {}
        
        accuracies = [result["accuracy"] for result in self.task_results.values()]
        
        metrics = {
            "average_accuracy": np.mean(accuracies),
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "num_tasks": len(self.task_results),
        }
        
        # Compute forgetting if multiple tasks
        if len(accuracies) > 1:
            metrics["forgetting"] = compute_forgetting(accuracies)
        
        return metrics
    
    def get_task_progression(self) -> List[Tuple[int, float]]:
        """Get task progression as list of (task_id, accuracy) pairs."""
        return [(task_id, result["accuracy"]) for task_id, result in self.task_results.items()]


class PrototypeAnalyzer:
    """
    Analyzer for prototype-based memory systems.
    """
    
    def __init__(self, prototype_manager: "PrototypeManager"):
        self.prototype_manager = prototype_manager
        
    def compute_prototype_statistics(self) -> Dict[str, float]:
        """Compute statistics about the prototype system."""
        stats = self.prototype_manager.get_statistics()
        prototypes = self.prototype_manager.get_all_prototypes()
        
        if not prototypes:
            return stats
        
        # Compute prototype diversity
        centroids = np.array([p.centroid for p in prototypes])
        
        # Average pairwise distance
        if len(centroids) > 1:
            distances = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    dist = euclidean_distance(centroids[i], centroids[j])
                    distances.append(dist)
            
            stats["avg_prototype_distance"] = np.mean(distances)
            stats["prototype_diversity"] = np.std(distances)
        else:
            stats["avg_prototype_distance"] = 0.0
            stats["prototype_diversity"] = 0.0
        
        # Prototype usage distribution
        counts = [p.count for p in prototypes]
        stats["prototype_usage_entropy"] = self._compute_entropy(counts)
        stats["prototype_usage_gini"] = self._compute_gini_coefficient(counts)
        
        return stats
    
    def _compute_entropy(self, counts: List[int]) -> float:
        """Compute entropy of prototype usage."""
        if not counts or sum(counts) == 0:
            return 0.0
        
        total = sum(counts)
        probs = [c / total for c in counts if c > 0]
        
        entropy = -sum(p * np.log2(p) for p in probs)
        return float(entropy)
    
    def _compute_gini_coefficient(self, counts: List[int]) -> float:
        """Compute Gini coefficient of prototype usage inequality."""
        if not counts:
            return 0.0
        
        # Sort counts
        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        
        # Compute Gini coefficient
        cumsum = np.cumsum(sorted_counts)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        return float(gini)


class TextGenerationMetrics:
    """
    Metrics for text generation tasks.
    """
    
    @staticmethod
    def bleu_score(generated: str, reference: str) -> float:
        """
        Compute BLEU score between generated and reference text.
        
        Args:
            generated: Generated text
            reference: Reference text
            
        Returns:
            BLEU score
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.tokenize import word_tokenize
            
            # Tokenize
            ref_tokens = word_tokenize(reference.lower())
            gen_tokens = word_tokenize(generated.lower())
            
            # Compute BLEU
            bleu = sentence_bleu([ref_tokens], gen_tokens)
            return float(bleu)
            
        except ImportError:
            logger.warning("NLTK not available for BLEU computation")
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to compute BLEU: {e}")
            return 0.0
    
    @staticmethod
    def rouge_score(generated: str, reference: str) -> Dict[str, float]:
        """
        Compute ROUGE scores between generated and reference text.
        
        Args:
            generated: Generated text
            reference: Reference text
            
        Returns:
            Dictionary with ROUGE scores
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, generated)
            
            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure,
            }
            
        except ImportError:
            logger.warning("rouge-score not available")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        except Exception as e:
            logger.warning(f"Failed to compute ROUGE: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def evaluate_model_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_type: str = "classification",
    **kwargs,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of model predictions.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        task_type: Type of task ("classification", "multilabel", "regression")
        **kwargs: Additional arguments for specific metrics
        
    Returns:
        Dictionary with comprehensive metrics
    """
    metrics = {}
    
    try:
        if task_type == "classification":
            # Single-label classification
            metrics["accuracy"] = accuracy(predictions, targets).item()
            
            prf_metrics = precision_recall_f1(predictions, targets)
            metrics.update(prf_metrics)
            
            # Top-5 accuracy if applicable
            if predictions.dim() > 1 and predictions.size(-1) >= 5:
                metrics["top5_accuracy"] = accuracy(predictions, targets, top_k=5).item()
        
        elif task_type == "multilabel":
            # Multi-label classification
            multilabel_results = multilabel_metrics(predictions, targets)
            metrics.update(multilabel_results)
        
        elif task_type == "regression":
            # Regression metrics
            if isinstance(predictions, torch.Tensor):
                pred_np = predictions.cpu().numpy()
            else:
                pred_np = predictions
                
            if isinstance(targets, torch.Tensor):
                target_np = targets.cpu().numpy()
            else:
                target_np = targets
            
            # MSE
            mse = np.mean((pred_np - target_np) ** 2)
            metrics["mse"] = float(mse)
            metrics["rmse"] = float(np.sqrt(mse))
            
            # MAE
            mae = np.mean(np.abs(pred_np - target_np))
            metrics["mae"] = float(mae)
            
            # R²
            ss_res = np.sum((target_np - pred_np) ** 2)
            ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            metrics["r2"] = float(r2)
            
    except Exception as e:
        logger.error(f"Failed to compute metrics for {task_type}: {e}")
        # Return default metrics
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    return metrics
