#!/usr/bin/env python3
"""
Example usage script for the Neuromorphic Continual Learning System.

This script demonstrates how to:
1. Create a basic configuration
2. Initialize the system components
3. Train on sample data
4. Run evaluation and inference

Run with: python examples/basic_usage.py
"""

import logging
from pathlib import Path

import torch
import yaml
from PIL import Image

# Import the neuromorphic CL system
from neuromorphic_cl import NeuromorphicContinualLearningSystem, SystemConfig
from neuromorphic_cl.data.dataloader import NeuromorphicDataModule
from neuromorphic_cl.training.distributed import DistributedTrainer
from neuromorphic_cl.utils.logging import setup_logging

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def create_minimal_config():
    """Create a minimal configuration for demonstration."""
    
    config = {
        "experiment_name": "neuromorphic_cl_demo",
        "seed": 42,
        
        # Minimal encoder config
        "concept_encoder": {
            "encoder_type": "vit",
            "projection_dim": 128,  # Smaller for demo
            "freeze_backbone": True,
        },
        
        # Small prototype system
        "prototype_manager": {
            "max_prototypes": 100,  # Small for demo
            "similarity_threshold": 0.8,
            "indexing_backend": "faiss",
        },
        
        # Simple SNN
        "snn": {
            "neuron_type": "lif",
            "num_timesteps": 10,  # Short simulation
        },
        
        # Basic answer composer
        "answer_composer": {
            "top_k_prototypes": 5,
            "llm_model_name": "gpt2",  # Small model
        },
        
        # Demo data config
        "data": {
            "batch_size": 4,  # Small batches
            "num_workers": 2,
            "image_size": [224, 224],
        },
        
        # Training config
        "training": {
            "learning_rate": 1e-4,
            "max_epochs": 2,  # Very short training
            "optimizer": "adamw",
        },
        
        # Single GPU
        "distributed": {
            "num_nodes": 1,
            "num_gpus_per_node": 1,
            "precision": "32",  # Full precision for stability
        },
        
        # Basic evaluation
        "evaluation": {
            "compute_forgetting": True,
            "run_baselines": False,  # Skip for demo
        }
    }
    
    return SystemConfig(**config)


def demonstrate_system_components():
    """Demonstrate the individual system components."""
    
    logger.info("=== Demonstrating System Components ===")
    
    # Create configuration
    config = create_minimal_config()
    logger.info(f"Created configuration for experiment: {config.experiment_name}")
    
    # Initialize the main system
    logger.info("Initializing neuromorphic continual learning system...")
    system = NeuromorphicContinualLearningSystem(config)
    
    # Create some dummy data
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_labels = torch.randint(0, 10, (batch_size,))
    
    logger.info(f"Created dummy data: images {dummy_images.shape}, labels {dummy_labels.shape}")
    
    # Demonstrate concept encoding
    logger.info("Running concept encoding...")
    system.eval()
    with torch.no_grad():
        outputs = system(
            pixel_values=dummy_images,
            labels=dummy_labels,
            task_type="classification",
            return_intermediate=True,
        )
    
    # Show results
    concept_embeddings = outputs["concept_embeddings"]
    prototype_ids = outputs["prototype_ids"]
    active_basin = outputs["active_basin"]
    
    logger.info(f"Concept embeddings shape: {concept_embeddings.shape}")
    logger.info(f"Prototype IDs: {prototype_ids}")
    logger.info(f"Active basin shape: {active_basin.shape}")
    
    # Show prototype manager stats
    proto_stats = system.prototype_manager.get_statistics()
    logger.info(f"Prototype statistics: {proto_stats}")
    
    # Show SNN memory stats
    snn_stats = system.snn.get_memory_statistics()
    logger.info(f"SNN statistics: {snn_stats}")
    
    return system


def demonstrate_training():
    """Demonstrate a minimal training loop."""
    
    logger.info("=== Demonstrating Training ===")
    
    # Create config and system
    config = create_minimal_config()
    
    # For training demo, we'll create a simple synthetic dataset
    # In practice, you would use the NeuromorphicDataModule with real data
    
    logger.info("Creating synthetic training data...")
    
    # Generate some synthetic samples
    num_samples = 20
    synthetic_data = []
    
    for i in range(num_samples):
        sample = {
            "pixel_values": torch.randn(3, 224, 224),
            "labels": torch.tensor(i % 5),  # 5 classes
            "task_type": "classification",
        }
        synthetic_data.append(sample)
    
    # Create a simple DataLoader
    from torch.utils.data import DataLoader, Dataset
    
    class SyntheticDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = SyntheticDataset(synthetic_data)
    dataloader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True)
    
    # Initialize system
    system = NeuromorphicContinualLearningSystem(config)
    system.train()
    
    # Simple training loop
    optimizer = torch.optim.AdamW(system.parameters(), lr=config.training.learning_rate)
    
    logger.info("Starting training loop...")
    
    for epoch in range(config.training.max_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = system(
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                task_type=batch["task_type"],
            )
            
            # Simple classification loss
            if "logits" in outputs:
                loss = torch.nn.functional.cross_entropy(outputs["logits"], batch["labels"])
            else:
                # Fallback loss
                loss = torch.tensor(0.1, requires_grad=True)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx == 0:  # Log first batch of each epoch
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Demonstrate prototype evolution
        proto_stats = system.prototype_manager.get_statistics()
        logger.info(f"Prototypes after epoch {epoch}: {proto_stats['total_prototypes']}")
    
    logger.info("Training demonstration completed!")
    return system


def demonstrate_inference():
    """Demonstrate inference on new data."""
    
    logger.info("=== Demonstrating Inference ===")
    
    # Load a trained system (in practice, you'd load from checkpoint)
    config = create_minimal_config()
    system = NeuromorphicContinualLearningSystem(config)
    system.eval()
    
    # Create some test data
    test_image = torch.randn(1, 3, 224, 224)  # Single image
    
    logger.info("Running inference on test image...")
    
    with torch.no_grad():
        # Classification
        results = system(
            pixel_values=test_image,
            task_type="classification",
            return_intermediate=True,
        )
        
        logger.info("Classification Results:")
        if "predictions" in results:
            predictions = results["predictions"]
            top_class = predictions.argmax(dim=-1)
            confidence = predictions.max(dim=-1)[0]
            logger.info(f"  Predicted class: {top_class.item()}")
            logger.info(f"  Confidence: {confidence.item():.3f}")
        
        # Show active prototypes
        if "active_basin" in results:
            active_basin = results["active_basin"][0]  # Remove batch dim
            top_prototypes = torch.topk(active_basin, k=3)
            logger.info("  Top active prototypes:")
            for i, (proto_id, activation) in enumerate(zip(top_prototypes.indices, top_prototypes.values)):
                logger.info(f"    {i+1}. Prototype {proto_id.item()}: {activation.item():.3f}")
        
        # Text generation demo
        results_gen = system(
            pixel_values=test_image,
            task_type="text_generation",
            query_text="Describe this image",
        )
        
        logger.info("Text Generation Results:")
        if "generated_text" in results_gen:
            logger.info(f"  Generated: {results_gen['generated_text']}")
        else:
            logger.info("  No text generated (model may need more training)")
    
    logger.info("Inference demonstration completed!")


def demonstrate_continual_learning():
    """Demonstrate continual learning across multiple tasks."""
    
    logger.info("=== Demonstrating Continual Learning ===")
    
    config = create_minimal_config()
    system = NeuromorphicContinualLearningSystem(config)
    
    # Simulate learning different tasks sequentially
    tasks = [
        {"name": "Task A", "num_classes": 3, "samples": 10},
        {"name": "Task B", "num_classes": 4, "samples": 10},
        {"name": "Task C", "num_classes": 2, "samples": 10},
    ]
    
    logger.info(f"Learning {len(tasks)} tasks sequentially...")
    
    for task_id, task_info in enumerate(tasks):
        logger.info(f"Learning {task_info['name']}...")
        
        # Notify system of task switch
        system.on_task_switch(task_id, task_info)
        
        # Generate synthetic data for this task
        for sample_id in range(task_info["samples"]):
            # Create sample
            image = torch.randn(1, 3, 224, 224)
            label = torch.randint(0, task_info["num_classes"], (1,))
            
            # Forward pass (learning happens automatically)
            with torch.no_grad():
                outputs = system(
                    pixel_values=image,
                    labels=label,
                    task_type="classification",
                )
        
        # Show system state after task
        proto_stats = system.prototype_manager.get_statistics()
        logger.info(f"  Prototypes after {task_info['name']}: {proto_stats['total_prototypes']}")
        
        snn_stats = system.snn.get_memory_statistics()
        logger.info(f"  SNN neurons: {snn_stats['total_neurons']}")
    
    # Final system summary
    final_summary = system.get_system_summary()
    logger.info("Final System Summary:")
    logger.info(f"  Total tasks learned: {system.current_task + 1}")
    logger.info(f"  Total prototypes: {final_summary['prototype_stats']['total_prototypes']}")
    logger.info(f"  Training steps: {final_summary['training_steps']}")
    
    logger.info("Continual learning demonstration completed!")


def main():
    """Main demonstration function."""
    
    logger.info("Starting Neuromorphic Continual Learning System Demo")
    logger.info("=" * 60)
    
    try:
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available, using CPU")
        
        # Run demonstrations
        demonstrate_system_components()
        print("\n")
        
        demonstrate_training()
        print("\n")
        
        demonstrate_inference()
        print("\n")
        
        demonstrate_continual_learning()
        print("\n")
        
        logger.info("All demonstrations completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Configure real datasets in config.yaml")
        logger.info("2. Train on actual data: neuromorphic-train train --config config.yaml")
        logger.info("3. Evaluate: neuromorphic-eval evaluate --config config.yaml --checkpoint model.ckpt")
        logger.info("4. Run inference: neuromorphic-infer infer --config config.yaml --checkpoint model.ckpt --input image.jpg")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
