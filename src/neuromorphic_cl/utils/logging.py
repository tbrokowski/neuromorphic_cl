"""
Logging Utilities for Neuromorphic Continual Learning.

This module provides comprehensive logging setup with support for
file logging, console output, and integration with experiment tracking.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import colorlog


def setup_logging(
    level: Union[str, int] = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    experiment_name: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    format_type: str = "detailed",
) -> None:
    """
    Setup comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        experiment_name: Name of experiment for log file naming
        console_output: Whether to output to console
        file_output: Whether to output to file
        format_type: Format type ("simple", "detailed", "json")
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    
    # Create formatters
    formatters = create_formatters(format_type)
    
    # Console handler with colors
    if console_output:
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatters["console"])
        root_logger.addHandler(console_handler)
    
    # File handlers
    if file_output and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        log_filename = f"{experiment_name}.log" if experiment_name else "neuromorphic_cl.log"
        file_handler = logging.FileHandler(log_dir / log_filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatters["file"])
        root_logger.addHandler(file_handler)
        
        # Error log file
        error_filename = f"{experiment_name}_error.log" if experiment_name else "neuromorphic_cl_error.log"
        error_handler = logging.FileHandler(log_dir / error_filename)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatters["file"])
        root_logger.addHandler(error_handler)
    
    # Set levels for specific loggers to reduce noise
    configure_external_loggers()
    
    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")
    if log_dir:
        logger.info(f"Log files will be saved to: {log_dir}")


def create_formatters(format_type: str = "detailed") -> dict:
    """
    Create logging formatters for different output types.
    
    Args:
        format_type: Type of formatting ("simple", "detailed", "json")
        
    Returns:
        Dictionary of formatters
    """
    if format_type == "simple":
        console_format = "%(log_color)s%(levelname)-8s%(reset)s %(message)s"
        file_format = "%(asctime)s - %(levelname)-8s - %(message)s"
        
    elif format_type == "detailed":
        console_format = (
            "%(log_color)s%(asctime)s%(reset)s | "
            "%(log_color)s%(levelname)-8s%(reset)s | "
            "%(cyan)s%(name)-20s%(reset)s | "
            "%(message)s"
        )
        file_format = (
            "%(asctime)s | %(levelname)-8s | %(name)-20s | "
            "%(filename)s:%(lineno)d | %(message)s"
        )
        
    elif format_type == "json":
        # For structured logging (would need json formatter)
        console_format = file_format = (
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
        )
        
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
    # Color formatter for console
    console_formatter = colorlog.ColoredFormatter(
        console_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Regular formatter for file
    file_formatter = logging.Formatter(
        file_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    return {
        "console": console_formatter,
        "file": file_formatter,
    }


def configure_external_loggers() -> None:
    """Configure logging levels for external libraries to reduce noise."""
    
    # Reduce noise from common libraries
    external_loggers = {
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        "matplotlib": logging.WARNING,
        "PIL": logging.WARNING,
        "transformers": logging.WARNING,
        "torch": logging.WARNING,
        "pytorch_lightning": logging.INFO,
        "lightning": logging.INFO,
        "wandb": logging.WARNING,
        "tensorboard": logging.WARNING,
        "spikingjelly": logging.INFO,
        "faiss": logging.WARNING,
        "hnswlib": logging.WARNING,
    }
    
    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


class ExperimentLogger:
    """
    Enhanced logger for experiment tracking and metrics.
    
    This class provides additional functionality for logging
    experiment metrics, model checkpoints, and system information.
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: Optional[Path] = None,
        enable_wandb: bool = False,
        wandb_project: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.enable_wandb = enable_wandb
        
        # Create experiment-specific logger
        self.logger = logging.getLogger(f"experiment.{experiment_name}")
        
        # Setup file handler for experiment logs
        self.log_dir.mkdir(parents=True, exist_ok=True)
        exp_log_file = self.log_dir / f"{experiment_name}_experiment.log"
        
        exp_handler = logging.FileHandler(exp_log_file)
        exp_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        exp_handler.setFormatter(exp_formatter)
        self.logger.addHandler(exp_handler)
        self.logger.setLevel(logging.INFO)
        
        # Initialize wandb if enabled
        if enable_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project or "neuromorphic-continual-learning",
                    name=experiment_name,
                    dir=str(self.log_dir),
                )
                self.wandb = wandb
            except ImportError:
                self.logger.warning("Wandb not available, skipping wandb logging")
                self.enable_wandb = False
                self.wandb = None
        else:
            self.wandb = None
    
    def log_experiment_start(self, config: dict) -> None:
        """Log experiment start with configuration."""
        self.logger.info("="*60)
        self.logger.info(f"EXPERIMENT START: {self.experiment_name}")
        self.logger.info("="*60)
        
        # Log configuration
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        # Log to wandb
        if self.wandb:
            self.wandb.config.update(config)
    
    def log_epoch_metrics(self, epoch: int, metrics: dict) -> None:
        """Log metrics for an epoch."""
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch:3d} | {metric_str}")
        
        # Log to wandb
        if self.wandb:
            wandb_metrics = {f"epoch/{k}": v for k, v in metrics.items()}
            wandb_metrics["epoch"] = epoch
            self.wandb.log(wandb_metrics)
    
    def log_step_metrics(self, step: int, metrics: dict) -> None:
        """Log metrics for a training step."""
        if step % 100 == 0:  # Log every 100 steps
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Step {step:6d} | {metric_str}")
        
        # Log to wandb
        if self.wandb:
            wandb_metrics = {f"step/{k}": v for k, v in metrics.items()}
            wandb_metrics["step"] = step
            self.wandb.log(wandb_metrics)
    
    def log_checkpoint(self, checkpoint_path: Path, metrics: dict) -> None:
        """Log checkpoint save with associated metrics."""
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.logger.info(f"Checkpoint metrics: {metrics}")
        
        # Log to wandb
        if self.wandb:
            self.wandb.log({"checkpoint_metrics": metrics})
    
    def log_task_switch(self, old_task: int, new_task: int, metrics: dict) -> None:
        """Log continual learning task switch."""
        self.logger.info("="*40)
        self.logger.info(f"TASK SWITCH: {old_task} -> {new_task}")
        self.logger.info("="*40)
        
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
        
        # Log to wandb
        if self.wandb:
            wandb_metrics = {f"task_switch/{k}": v for k, v in metrics.items()}
            wandb_metrics.update({
                "old_task": old_task,
                "new_task": new_task,
            })
            self.wandb.log(wandb_metrics)
    
    def log_memory_usage(self, memory_info: dict) -> None:
        """Log memory usage information."""
        self.logger.info("Memory Usage:")
        for key, value in memory_info.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.2f}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        # Log to wandb
        if self.wandb:
            wandb_metrics = {f"memory/{k}": v for k, v in memory_info.items()}
            self.wandb.log(wandb_metrics)
    
    def log_system_info(self, system_info: dict) -> None:
        """Log system information."""
        self.logger.info("System Information:")
        for key, value in system_info.items():
            self.logger.info(f"  {key}: {value}")
        
        # Log to wandb
        if self.wandb:
            self.wandb.config.update({"system": system_info})
    
    def log_experiment_end(self, final_metrics: dict) -> None:
        """Log experiment completion."""
        self.logger.info("="*60)
        self.logger.info(f"EXPERIMENT END: {self.experiment_name}")
        self.logger.info("="*60)
        
        self.logger.info("Final Results:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        # Log to wandb
        if self.wandb:
            wandb_metrics = {f"final/{k}": v for k, v in final_metrics.items()}
            self.wandb.log(wandb_metrics)
            self.wandb.finish()
    
    def close(self) -> None:
        """Close the experiment logger."""
        if self.wandb:
            self.wandb.finish()


class MetricsLogger:
    """
    Specialized logger for tracking metrics during training.
    """
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.metrics_history = []
        
        # Create CSV header
        with open(log_file, 'w') as f:
            f.write("timestamp,epoch,step,metric_name,metric_value\n")
    
    def log_metric(self, epoch: int, step: int, metric_name: str, metric_value: float) -> None:
        """Log a single metric value."""
        import datetime
        
        timestamp = datetime.datetime.now().isoformat()
        
        # Store in memory
        self.metrics_history.append({
            "timestamp": timestamp,
            "epoch": epoch,
            "step": step,
            "metric_name": metric_name,
            "metric_value": metric_value,
        })
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{epoch},{step},{metric_name},{metric_value}\n")
    
    def log_metrics_dict(self, epoch: int, step: int, metrics: dict) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_metric(epoch, step, name, value)
    
    def get_metric_history(self, metric_name: str) -> list:
        """Get history of a specific metric."""
        return [
            entry for entry in self.metrics_history
            if entry["metric_name"] == metric_name
        ]


def create_experiment_logger(
    experiment_name: str,
    config: dict,
    log_dir: Optional[Path] = None,
    enable_wandb: bool = False,
    wandb_project: Optional[str] = None,
) -> ExperimentLogger:
    """
    Create and configure an experiment logger.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        log_dir: Directory for log files
        enable_wandb: Whether to enable wandb logging
        wandb_project: Wandb project name
        
    Returns:
        Configured ExperimentLogger instance
    """
    exp_logger = ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=log_dir,
        enable_wandb=enable_wandb,
        wandb_project=wandb_project,
    )
    
    exp_logger.log_experiment_start(config)
    
    return exp_logger


def log_gpu_memory_usage() -> dict:
    """
    Log current GPU memory usage.
    
    Returns:
        Dictionary with memory usage information
    """
    import torch
    
    memory_info = {}
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            
            memory_info[f"gpu_{i}_allocated_gb"] = memory_allocated
            memory_info[f"gpu_{i}_reserved_gb"] = memory_reserved
            memory_info[f"gpu_{i}_total_gb"] = memory_total
            memory_info[f"gpu_{i}_utilization"] = memory_allocated / memory_total
    
    return memory_info


def setup_distributed_logging(rank: int, world_size: int) -> None:
    """
    Setup logging for distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    # Only log from master process for most loggers
    if rank == 0:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    # Setup basic logging
    setup_logging(
        level=level,
        console_output=rank == 0,  # Only master process logs to console
        file_output=True,  # All processes log to files
    )
    
    # Add rank to logger names
    logger = logging.getLogger(__name__)
    logger.info(f"Distributed logging setup: rank {rank}/{world_size}")
