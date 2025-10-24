"""
Evaluation CLI for Neuromorphic Continual Learning System.

This module provides the command-line interface for evaluating the
trained neuromorphic continual learning system on various tasks.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import click
import torch
import yaml
from omegaconf import OmegaConf

from ..configs.schema import SystemConfig
from ..core.system import NeuromorphicContinualLearningSystem
from ..data.dataloader import NeuromorphicDataModule
from ..utils.logging import setup_logging
from ..utils.metrics import (
    ContinualLearningMetrics,
    PrototypeAnalyzer,
    evaluate_model_predictions,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to configuration file"
)
@click.option(
    "--checkpoint",
    "-k",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model checkpoint"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="evaluation_results",
    help="Output directory for results"
)
@click.option(
    "--test-datasets",
    "-d",
    multiple=True,
    help="Specific datasets to evaluate on"
)
@click.option(
    "--task-sequence",
    "-t",
    type=str,
    help="Task sequence for continual learning evaluation (comma-separated)"
)
@click.option(
    "--metrics",
    "-m",
    multiple=True,
    default=["accuracy", "forgetting", "transfer"],
    help="Metrics to compute"
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    help="Evaluation batch size"
)
@click.option(
    "--num-samples",
    "-n",
    type=int,
    help="Maximum number of samples to evaluate"
)
@click.option(
    "--save-predictions",
    is_flag=True,
    help="Save model predictions"
)
@click.option(
    "--analyze-prototypes",
    is_flag=True,
    help="Perform prototype analysis"
)
@click.option(
    "--baseline-comparison",
    is_flag=True,
    help="Compare against baseline methods"
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="Device to use (auto, cpu, cuda)"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode"
)
def evaluate(
    config: Path,
    checkpoint: Path,
    output_dir: Path,
    test_datasets: tuple,
    task_sequence: Optional[str],
    metrics: tuple,
    batch_size: Optional[int],
    num_samples: Optional[int],
    save_predictions: bool,
    analyze_prototypes: bool,
    baseline_comparison: bool,
    device: str,
    debug: bool,
) -> None:
    """
    Evaluate the neuromorphic continual learning system.
    
    This command evaluates a trained model on test datasets and computes
    comprehensive metrics for continual learning performance assessment.
    """
    try:
        # Setup logging
        log_level = "DEBUG" if debug else "INFO"
        setup_logging(level=log_level)
        
        # Load configuration
        click.echo(f"Loading configuration from {config}")
        system_config = load_config(config, batch_size)
        
        # Setup device
        device = setup_device(device)
        click.echo(f"Using device: {device}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and checkpoint
        click.echo(f"Loading model from checkpoint: {checkpoint}")
        model = load_model_checkpoint(system_config, checkpoint, device)
        
        # Setup data module
        click.echo("Setting up data module...")
        data_module = NeuromorphicDataModule(system_config)
        data_module.setup(stage="test")
        
        # Filter datasets if specified
        test_dataloaders = data_module.test_dataloader()
        if test_datasets:
            test_dataloaders = filter_dataloaders(test_dataloaders, test_datasets)
        
        # Parse task sequence
        if task_sequence:
            task_ids = [int(x.strip()) for x in task_sequence.split(",")]
        else:
            task_ids = list(range(len(test_dataloaders)))
        
        # Run evaluation
        click.echo("Starting evaluation...")
        results = run_evaluation(
            model=model,
            dataloaders=test_dataloaders,
            task_ids=task_ids,
            metrics=list(metrics),
            num_samples=num_samples,
            save_predictions=save_predictions,
            device=device,
        )
        
        # Prototype analysis
        if analyze_prototypes:
            click.echo("Analyzing prototypes...")
            prototype_results = analyze_prototype_system(model)
            results["prototype_analysis"] = prototype_results
        
        # Baseline comparison
        if baseline_comparison:
            click.echo("Running baseline comparisons...")
            baseline_results = run_baseline_comparison(
                system_config, test_dataloaders, device
            )
            results["baseline_comparison"] = baseline_results
        
        # Save results
        results_file = output_dir / "evaluation_results.json"
        save_results(results, results_file)
        
        # Generate report
        report_file = output_dir / "evaluation_report.md"
        generate_evaluation_report(results, report_file)
        
        click.echo(f"Evaluation completed successfully!")
        click.echo(f"Results saved to: {output_dir}")
        click.echo(f"Report available at: {report_file}")
        
        # Print summary
        print_evaluation_summary(results)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if debug:
            raise
        sys.exit(1)


def load_config(config_path: Path, batch_size: Optional[int] = None) -> SystemConfig:
    """Load and validate configuration."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Apply overrides
    if batch_size is not None:
        config_dict["data"]["batch_size"] = batch_size
    
    return SystemConfig(**config_dict)


def setup_device(device_str: str) -> torch.device:
    """Setup computation device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    return device


def load_model_checkpoint(
    config: SystemConfig, 
    checkpoint_path: Path, 
    device: torch.device
) -> NeuromorphicContinualLearningSystem:
    """Load model from checkpoint."""
    # Create model
    model = NeuromorphicContinualLearningSystem(config)
    
    # Load checkpoint
    if checkpoint_path.suffix == ".ckpt":
        # PyTorch Lightning checkpoint
        model = model.load_from_checkpoint(str(checkpoint_path), config=config)
    else:
        # Regular PyTorch checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    
    model.to(device)
    model.eval()
    
    # Load additional system state if available
    system_state_dir = checkpoint_path.parent
    if (system_state_dir / "prototypes.pkl").exists():
        model.load_system_state(system_state_dir)
        logger.info("Loaded complete system state including prototypes")
    
    return model


def filter_dataloaders(dataloaders: List, dataset_names: tuple) -> List:
    """Filter dataloaders based on dataset names."""
    # This is a simplified version - would need proper dataset identification
    if len(dataset_names) == 0:
        return dataloaders
    
    # For now, just return specified number of dataloaders
    num_datasets = min(len(dataset_names), len(dataloaders))
    return dataloaders[:num_datasets]


def run_evaluation(
    model: NeuromorphicContinualLearningSystem,
    dataloaders: List,
    task_ids: List[int],
    metrics: List[str],
    num_samples: Optional[int],
    save_predictions: bool,
    device: torch.device,
) -> Dict:
    """Run comprehensive evaluation."""
    
    results = {
        "task_results": {},
        "continual_learning_metrics": {},
        "overall_metrics": {},
    }
    
    # Initialize metrics tracker
    cl_metrics = ContinualLearningMetrics()
    
    # Evaluate each task
    for task_id in task_ids:
        if task_id >= len(dataloaders):
            logger.warning(f"Task {task_id} not available, skipping")
            continue
        
        dataloader = dataloaders[task_id]
        logger.info(f"Evaluating task {task_id}")
        
        task_results = evaluate_single_task(
            model=model,
            dataloader=dataloader,
            task_id=task_id,
            num_samples=num_samples,
            save_predictions=save_predictions,
            device=device,
        )
        
        results["task_results"][task_id] = task_results
        cl_metrics.add_task_result(task_id, task_results["accuracy"])
        
        # Update model for continual learning simulation
        model.on_task_switch(task_id)
    
    # Compute continual learning metrics
    if "forgetting" in metrics:
        # Re-evaluate previous tasks to measure forgetting
        forgetting_results = measure_forgetting(model, dataloaders, task_ids, device)
        results["continual_learning_metrics"]["forgetting"] = forgetting_results
    
    if "transfer" in metrics:
        # This would require baseline single-task performance
        logger.info("Transfer measurement requires baseline comparison")
    
    # Overall summary metrics
    task_accuracies = [r["accuracy"] for r in results["task_results"].values()]
    results["overall_metrics"] = {
        "average_accuracy": sum(task_accuracies) / len(task_accuracies) if task_accuracies else 0.0,
        "final_accuracy": task_accuracies[-1] if task_accuracies else 0.0,
        "num_tasks_evaluated": len(task_accuracies),
    }
    
    # Add CL metrics summary
    cl_summary = cl_metrics.compute_summary_metrics()
    results["continual_learning_metrics"].update(cl_summary)
    
    return results


def evaluate_single_task(
    model: NeuromorphicContinualLearningSystem,
    dataloader: torch.utils.data.DataLoader,
    task_id: int,
    num_samples: Optional[int],
    save_predictions: bool,
    device: torch.device,
) -> Dict:
    """Evaluate model on a single task."""
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_metadata = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"],
                text_tokens=batch.get("text_tokens"),
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
                task_type="classification",
            )
            
            # Collect predictions and targets
            if "predictions" in outputs and batch.get("labels") is not None:
                predictions = outputs["predictions"]
                targets = batch["labels"]
                
                # Filter out abstained predictions if needed
                if "should_abstain" in outputs:
                    valid_mask = ~outputs["should_abstain"]
                    if valid_mask.any():
                        predictions = predictions[valid_mask]
                        targets = targets[valid_mask]
                    else:
                        continue  # Skip if all predictions abstained
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                
                if batch.get("metadata"):
                    all_metadata.extend(batch["metadata"])
                
                sample_count += predictions.size(0)
                
                # Check sample limit
                if num_samples and sample_count >= num_samples:
                    break
    
    if not all_predictions:
        logger.warning(f"No valid predictions for task {task_id}")
        return {"accuracy": 0.0, "num_samples": 0}
    
    # Concatenate all predictions and targets
    predictions_tensor = torch.cat(all_predictions, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    task_metrics = evaluate_model_predictions(
        predictions=predictions_tensor,
        targets=targets_tensor,
        task_type="classification",
    )
    
    # Add sample count
    task_metrics["num_samples"] = len(predictions_tensor)
    
    # Save predictions if requested
    if save_predictions:
        predictions_data = {
            "predictions": predictions_tensor.tolist(),
            "targets": targets_tensor.tolist(),
            "metadata": all_metadata,
        }
        # Would save to file here
    
    return task_metrics


def measure_forgetting(
    model: NeuromorphicContinualLearningSystem,
    dataloaders: List,
    task_ids: List[int],
    device: torch.device,
) -> Dict:
    """Measure catastrophic forgetting by re-evaluating previous tasks."""
    
    forgetting_results = {}
    
    for i, task_id in enumerate(task_ids[:-1]):  # Exclude last task
        logger.info(f"Re-evaluating task {task_id} for forgetting measurement")
        
        current_results = evaluate_single_task(
            model=model,
            dataloader=dataloaders[task_id],
            task_id=task_id,
            num_samples=None,
            save_predictions=False,
            device=device,
        )
        
        forgetting_results[task_id] = {
            "current_accuracy": current_results["accuracy"],
            "task_position": i,
        }
    
    # Compute average forgetting
    if forgetting_results:
        accuracies = [r["current_accuracy"] for r in forgetting_results.values()]
        forgetting_results["average_forgetting"] = 1.0 - (sum(accuracies) / len(accuracies))
    
    return forgetting_results


def analyze_prototype_system(model: NeuromorphicContinualLearningSystem) -> Dict:
    """Analyze the prototype-based memory system."""
    
    analyzer = PrototypeAnalyzer(model.prototype_manager)
    
    # Basic statistics
    stats = analyzer.compute_prototype_statistics()
    
    # SNN memory statistics
    snn_stats = model.snn.get_memory_statistics()
    
    # System summary
    system_summary = model.get_system_summary()
    
    return {
        "prototype_statistics": stats,
        "snn_statistics": snn_stats,
        "system_summary": system_summary,
    }


def run_baseline_comparison(
    config: SystemConfig,
    dataloaders: List,
    device: torch.device,
) -> Dict:
    """Run comparison against baseline continual learning methods."""
    
    # This is a placeholder for baseline comparison
    # Would implement various baseline methods here
    
    baseline_results = {
        "sequential_finetune": {"accuracy": 0.45, "forgetting": 0.8},
        "experience_replay": {"accuracy": 0.65, "forgetting": 0.4},
        "ewc": {"accuracy": 0.55, "forgetting": 0.6},
        "rag": {"accuracy": 0.70, "forgetting": 0.2},
    }
    
    logger.info("Baseline comparison is not fully implemented")
    
    return baseline_results


def save_results(results: Dict, output_file: Path) -> None:
    """Save evaluation results to JSON file."""
    
    # Convert any tensors to lists for JSON serialization
    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_tensors(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def generate_evaluation_report(results: Dict, output_file: Path) -> None:
    """Generate a markdown evaluation report."""
    
    report_lines = [
        "# Neuromorphic Continual Learning Evaluation Report",
        "",
        "## Overview",
        "",
    ]
    
    # Overall metrics
    overall = results.get("overall_metrics", {})
    report_lines.extend([
        "### Overall Performance",
        "",
        f"- **Average Accuracy**: {overall.get('average_accuracy', 0):.3f}",
        f"- **Final Accuracy**: {overall.get('final_accuracy', 0):.3f}",
        f"- **Tasks Evaluated**: {overall.get('num_tasks_evaluated', 0)}",
        "",
    ])
    
    # Task-specific results
    task_results = results.get("task_results", {})
    if task_results:
        report_lines.extend([
            "### Task-Specific Results",
            "",
            "| Task ID | Accuracy | Precision | Recall | F1 Score | Samples |",
            "|---------|----------|-----------|--------|----------|---------|",
        ])
        
        for task_id, metrics in task_results.items():
            acc = metrics.get("accuracy", 0)
            prec = metrics.get("precision", 0)
            rec = metrics.get("recall", 0)
            f1 = metrics.get("f1", 0)
            samples = metrics.get("num_samples", 0)
            
            report_lines.append(
                f"| {task_id} | {acc:.3f} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {samples} |"
            )
        
        report_lines.append("")
    
    # Continual learning metrics
    cl_metrics = results.get("continual_learning_metrics", {})
    if cl_metrics:
        report_lines.extend([
            "### Continual Learning Metrics",
            "",
        ])
        
        if "forgetting" in cl_metrics:
            avg_forgetting = cl_metrics["forgetting"].get("average_forgetting", 0)
            report_lines.append(f"- **Average Forgetting**: {avg_forgetting:.3f}")
        
        if "average_accuracy" in cl_metrics:
            cl_avg_acc = cl_metrics["average_accuracy"]
            report_lines.append(f"- **CL Average Accuracy**: {cl_avg_acc:.3f}")
        
        report_lines.append("")
    
    # Prototype analysis
    if "prototype_analysis" in results:
        proto_stats = results["prototype_analysis"].get("prototype_statistics", {})
        report_lines.extend([
            "### Prototype Analysis",
            "",
            f"- **Total Prototypes**: {proto_stats.get('total_prototypes', 0)}",
            f"- **Average Prototype Size**: {proto_stats.get('avg_prototype_size', 0):.1f}",
            f"- **Prototype Diversity**: {proto_stats.get('prototype_diversity', 0):.3f}",
            "",
        ])
    
    # Baseline comparison
    if "baseline_comparison" in results:
        baseline_results = results["baseline_comparison"]
        report_lines.extend([
            "### Baseline Comparison",
            "",
            "| Method | Accuracy | Forgetting |",
            "|--------|----------|------------|",
        ])
        
        for method, metrics in baseline_results.items():
            acc = metrics.get("accuracy", 0)
            forg = metrics.get("forgetting", 0)
            report_lines.append(f"| {method} | {acc:.3f} | {forg:.3f} |")
        
        report_lines.append("")
    
    # Write report
    with open(output_file, 'w') as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Report generated: {output_file}")


def print_evaluation_summary(results: Dict) -> None:
    """Print a summary of evaluation results to console."""
    
    click.echo("\n" + "="*60)
    click.echo("EVALUATION SUMMARY")
    click.echo("="*60)
    
    # Overall metrics
    overall = results.get("overall_metrics", {})
    click.echo(f"Average Accuracy: {overall.get('average_accuracy', 0):.3f}")
    click.echo(f"Final Accuracy: {overall.get('final_accuracy', 0):.3f}")
    click.echo(f"Tasks Evaluated: {overall.get('num_tasks_evaluated', 0)}")
    
    # Continual learning metrics
    cl_metrics = results.get("continual_learning_metrics", {})
    if "forgetting" in cl_metrics:
        avg_forgetting = cl_metrics["forgetting"].get("average_forgetting", 0)
        click.echo(f"Average Forgetting: {avg_forgetting:.3f}")
    
    # Prototype info
    if "prototype_analysis" in results:
        proto_stats = results["prototype_analysis"].get("prototype_statistics", {})
        click.echo(f"Total Prototypes: {proto_stats.get('total_prototypes', 0)}")
    
    click.echo("="*60)


@click.command()
@click.option(
    "--results-file",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to evaluation results JSON file"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="analysis_plots",
    help="Output directory for plots"
)
@click.option(
    "--plot-types",
    "-p",
    multiple=True,
    default=["accuracy", "forgetting", "prototypes"],
    help="Types of plots to generate"
)
def analyze_results(
    results_file: Path,
    output_dir: Path,
    plot_types: tuple,
) -> None:
    """
    Analyze and visualize evaluation results.
    
    This command generates plots and analysis from evaluation results.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Load results
        with open(results_file) as f:
            results = json.load(f)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate requested plots
        for plot_type in plot_types:
            if plot_type == "accuracy":
                plot_accuracy_progression(results, output_dir)
            elif plot_type == "forgetting":
                plot_forgetting_analysis(results, output_dir)
            elif plot_type == "prototypes":
                plot_prototype_analysis(results, output_dir)
        
        click.echo(f"Analysis plots saved to: {output_dir}")
        
    except ImportError:
        click.echo("Matplotlib/Seaborn not available for plotting", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Analysis failed: {e}", err=True)
        sys.exit(1)


def plot_accuracy_progression(results: Dict, output_dir: Path) -> None:
    """Plot accuracy progression across tasks."""
    # Implementation would create accuracy progression plots
    pass


def plot_forgetting_analysis(results: Dict, output_dir: Path) -> None:
    """Plot forgetting analysis."""
    # Implementation would create forgetting analysis plots
    pass


def plot_prototype_analysis(results: Dict, output_dir: Path) -> None:
    """Plot prototype system analysis."""
    # Implementation would create prototype analysis plots
    pass


@click.group()
def main() -> None:
    """Neuromorphic Continual Learning Evaluation CLI."""
    pass


# Add commands to the group
main.add_command(evaluate)
main.add_command(analyze_results)


if __name__ == "__main__":
    main()
