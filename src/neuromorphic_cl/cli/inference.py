"""
Inference CLI for Neuromorphic Continual Learning System.

This module provides the command-line interface for running inference
with the trained neuromorphic continual learning system on new data.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import click
import torch
import yaml
from PIL import Image

from ..configs.schema import SystemConfig
from ..core.system import NeuromorphicContinualLearningSystem
from ..utils.logging import setup_logging
from ..utils.preprocessing import normalize_image, render_pdf_page

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
    "--input",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input file (image, PDF, or directory)"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for results (JSON)"
)
@click.option(
    "--task-type",
    "-t",
    type=click.Choice(["classification", "text_generation", "visual_qa", "matching"]),
    default="classification",
    help="Type of inference task"
)
@click.option(
    "--query-text",
    "-q",
    type=str,
    help="Query text for VQA or text generation tasks"
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=1,
    help="Batch size for inference"
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="Device to use (auto, cpu, cuda)"
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature for text generation"
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Top-k results to return"
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.5,
    help="Confidence threshold for predictions"
)
@click.option(
    "--return-evidence",
    is_flag=True,
    help="Return evidence and prototype information"
)
@click.option(
    "--return-saliency",
    is_flag=True,
    help="Return attention/saliency maps"
)
@click.option(
    "--save-visualizations",
    is_flag=True,
    help="Save visualization images"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode"
)
def infer(
    config: Path,
    checkpoint: Path,
    input: Path,
    output: Optional[Path],
    task_type: str,
    query_text: Optional[str],
    batch_size: int,
    device: str,
    temperature: float,
    top_k: int,
    confidence_threshold: float,
    return_evidence: bool,
    return_saliency: bool,
    save_visualizations: bool,
    debug: bool,
) -> None:
    """
    Run inference with the neuromorphic continual learning system.
    
    This command processes input data through the trained model and
    returns predictions, generated text, or other task-specific outputs.
    """
    try:
        # Setup logging
        log_level = "DEBUG" if debug else "INFO"
        setup_logging(level=log_level)
        
        # Load configuration and model
        click.echo(f"Loading model configuration from {config}")
        system_config = load_config(config)
        
        # Setup device
        device = setup_device(device)
        click.echo(f"Using device: {device}")
        
        # Load model
        click.echo(f"Loading model from checkpoint: {checkpoint}")
        model = load_model_checkpoint(system_config, checkpoint, device)
        
        # Process input
        click.echo(f"Processing input: {input}")
        input_data = process_input(input, system_config, device)
        
        # Run inference
        click.echo(f"Running {task_type} inference...")
        results = run_inference(
            model=model,
            input_data=input_data,
            task_type=task_type,
            query_text=query_text,
            temperature=temperature,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            return_evidence=return_evidence,
            return_saliency=return_saliency,
            device=device,
        )
        
        # Save visualizations if requested
        if save_visualizations:
            viz_dir = output.parent / "visualizations" if output else Path("visualizations")
            save_visualization_outputs(results, input_data, viz_dir)
        
        # Format and display results
        formatted_results = format_results(results, task_type)
        display_results(formatted_results, task_type)
        
        # Save results to file
        if output:
            save_results_to_file(formatted_results, output)
            click.echo(f"Results saved to: {output}")
        
        click.echo("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        if debug:
            raise
        sys.exit(1)


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
    "--input-dir",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input directory with files to process"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for results"
)
@click.option(
    "--task-type",
    "-t",
    type=click.Choice(["classification", "text_generation", "visual_qa"]),
    default="classification",
    help="Type of inference task"
)
@click.option(
    "--file-pattern",
    "-p",
    type=str,
    default="*",
    help="File pattern to match (e.g., '*.pdf', '*.jpg')"
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=8,
    help="Batch size for processing"
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="Device to use"
)
@click.option(
    "--num-workers",
    "-w",
    type=int,
    default=4,
    help="Number of worker processes"
)
@click.option(
    "--save-predictions",
    is_flag=True,
    help="Save individual prediction files"
)
@click.option(
    "--progress",
    is_flag=True,
    help="Show progress bar"
)
def batch_infer(
    config: Path,
    checkpoint: Path,
    input_dir: Path,
    output_dir: Path,
    task_type: str,
    file_pattern: str,
    batch_size: int,
    device: str,
    num_workers: int,
    save_predictions: bool,
    progress: bool,
) -> None:
    """
    Run batch inference on multiple files.
    
    This command processes a directory of files through the model
    and saves results for each file.
    """
    try:
        # Setup
        setup_logging(level="INFO")
        
        # Load model
        system_config = load_config(config)
        device = setup_device(device)
        model = load_model_checkpoint(system_config, checkpoint, device)
        
        # Find input files
        input_files = list(Path(input_dir).glob(file_pattern))
        click.echo(f"Found {len(input_files)} files to process")
        
        if not input_files:
            click.echo("No files found matching pattern", err=True)
            return
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files
        all_results = {}
        
        if progress:
            from tqdm import tqdm
            file_iter = tqdm(input_files, desc="Processing files")
        else:
            file_iter = input_files
        
        for input_file in file_iter:
            try:
                # Process single file
                input_data = process_input(input_file, system_config, device)
                
                results = run_inference(
                    model=model,
                    input_data=input_data,
                    task_type=task_type,
                    query_text=None,
                    temperature=0.7,
                    top_k=5,
                    confidence_threshold=0.5,
                    return_evidence=False,
                    return_saliency=False,
                    device=device,
                )
                
                formatted_results = format_results(results, task_type)
                all_results[str(input_file)] = formatted_results
                
                # Save individual result if requested
                if save_predictions:
                    result_file = output_dir / f"{input_file.stem}_prediction.json"
                    save_results_to_file(formatted_results, result_file)
                
            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")
                all_results[str(input_file)] = {"error": str(e)}
        
        # Save combined results
        combined_file = output_dir / "batch_results.json"
        save_results_to_file(all_results, combined_file)
        
        # Generate summary
        generate_batch_summary(all_results, output_dir / "batch_summary.txt")
        
        click.echo(f"Batch processing completed. Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        sys.exit(1)


def load_config(config_path: Path) -> SystemConfig:
    """Load system configuration."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return SystemConfig(**config_dict)


def setup_device(device_str: str) -> torch.device:
    """Setup computation device."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    return device


def load_model_checkpoint(
    config: SystemConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> NeuromorphicContinualLearningSystem:
    """Load model from checkpoint."""
    model = NeuromorphicContinualLearningSystem(config)
    
    if checkpoint_path.suffix == ".ckpt":
        model = model.load_from_checkpoint(str(checkpoint_path), config=config)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    
    model.to(device)
    model.eval()
    
    # Load system state if available
    system_state_dir = checkpoint_path.parent
    if (system_state_dir / "prototypes.pkl").exists():
        model.load_system_state(system_state_dir)
    
    return model


def process_input(
    input_path: Path,
    config: SystemConfig,
    device: torch.device,
) -> Dict:
    """Process input file(s) into model format."""
    
    input_path = Path(input_path)
    
    if input_path.is_dir():
        # Process directory of images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {input_path}")
        
        # Process first image for now (could batch multiple)
        input_path = image_files[0]
    
    # Process single file
    if input_path.suffix.lower() == '.pdf':
        # Render PDF to images
        images = render_pdf_page(input_path, page_num=0)
        if not images:
            raise ValueError(f"Failed to render PDF: {input_path}")
        image = images[0]
    else:
        # Load image directly
        try:
            image = Image.open(input_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image {input_path}: {e}")
    
    # Normalize image
    pixel_values = normalize_image(
        image,
        target_size=tuple(config.data.image_size),
    ).unsqueeze(0).to(device)  # Add batch dimension
    
    return {
        "pixel_values": pixel_values,
        "input_path": str(input_path),
        "original_image": image,
    }


def run_inference(
    model: NeuromorphicContinualLearningSystem,
    input_data: Dict,
    task_type: str,
    query_text: Optional[str],
    temperature: float,
    top_k: int,
    confidence_threshold: float,
    return_evidence: bool,
    return_saliency: bool,
    device: torch.device,
) -> Dict:
    """Run inference with the model."""
    
    model.eval()
    
    with torch.no_grad():
        # Prepare inputs
        pixel_values = input_data["pixel_values"]
        text_tokens = None
        attention_mask = None
        
        # Add query text if provided
        if query_text and task_type in ["visual_qa", "text_generation"]:
            # Tokenize query text
            tokenizer = model.concept_encoder.tokenizer
            tokens = tokenizer(
                query_text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            text_tokens = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
        
        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            task_type=task_type,
            query_text=query_text,
            return_intermediate=return_evidence or return_saliency,
        )
        
        # Process outputs based on task type
        results = {
            "task_type": task_type,
            "input_path": input_data["input_path"],
        }
        
        if task_type == "classification":
            results.update(process_classification_output(outputs, top_k, confidence_threshold))
        elif task_type == "text_generation":
            results.update(process_generation_output(outputs, query_text))
        elif task_type == "visual_qa":
            results.update(process_vqa_output(outputs, query_text))
        elif task_type == "matching":
            results.update(process_matching_output(outputs, top_k))
        
        # Add evidence information if requested
        if return_evidence:
            results["evidence"] = extract_evidence_info(outputs, model)
        
        # Add saliency information if requested
        if return_saliency:
            results["saliency"] = extract_saliency_info(outputs)
        
        return results


def process_classification_output(
    outputs: Dict,
    top_k: int,
    confidence_threshold: float,
) -> Dict:
    """Process classification model outputs."""
    
    results = {}
    
    if "predictions" in outputs:
        predictions = outputs["predictions"][0]  # Remove batch dimension
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(predictions, min(top_k, len(predictions)))
        
        results["predictions"] = [
            {
                "class_id": int(idx),
                "confidence": float(prob),
                "above_threshold": float(prob) >= confidence_threshold,
            }
            for idx, prob in zip(top_k_indices, top_k_probs)
        ]
        
        # Overall confidence
        results["max_confidence"] = float(top_k_probs[0])
        results["predicted_class"] = int(top_k_indices[0])
    
    # Abstention information
    if "should_abstain" in outputs:
        results["abstained"] = bool(outputs["should_abstain"][0])
    
    if "confidence" in outputs:
        results["model_confidence"] = float(outputs["confidence"][0])
    
    return results


def process_generation_output(outputs: Dict, query_text: Optional[str]) -> Dict:
    """Process text generation outputs."""
    
    results = {
        "query": query_text or "",
    }
    
    if "generated_text" in outputs:
        results["generated_text"] = outputs["generated_text"]
    elif "generated_texts" in outputs:
        results["generated_texts"] = outputs["generated_texts"]
    
    if "should_abstain" in outputs:
        results["abstained"] = bool(outputs["should_abstain"][0])
    
    return results


def process_vqa_output(outputs: Dict, query_text: Optional[str]) -> Dict:
    """Process visual question answering outputs."""
    
    results = {
        "question": query_text or "",
    }
    
    # Combine classification and generation outputs
    if "predictions" in outputs:
        # Classification-style answer
        predictions = outputs["predictions"][0]
        top_prob, top_idx = torch.max(predictions, dim=0)
        
        results["answer_class"] = int(top_idx)
        results["answer_confidence"] = float(top_prob)
    
    if "generated_text" in outputs:
        # Generation-style answer
        results["generated_answer"] = outputs["generated_text"]
    
    if "should_abstain" in outputs:
        results["abstained"] = bool(outputs["should_abstain"][0])
    
    return results


def process_matching_output(outputs: Dict, top_k: int) -> Dict:
    """Process matching/retrieval outputs."""
    
    results = {}
    
    if "similarities" in outputs:
        similarities = outputs["similarities"][0]  # Remove batch dimension
        
        # Get top-k matches
        top_k_sims, top_k_indices = torch.topk(similarities, min(top_k, len(similarities)))
        
        results["matches"] = [
            {
                "prototype_id": int(idx),
                "similarity": float(sim),
            }
            for idx, sim in zip(top_k_indices, top_k_sims)
        ]
        
        results["best_match"] = {
            "prototype_id": int(top_k_indices[0]),
            "similarity": float(top_k_sims[0]),
        }
    
    return results


def extract_evidence_info(outputs: Dict, model: NeuromorphicContinualLearningSystem) -> Dict:
    """Extract evidence and prototype information."""
    
    evidence_info = {}
    
    if "active_basin" in outputs:
        active_basin = outputs["active_basin"][0]  # Remove batch dimension
        
        # Get most active prototypes
        top_activations, top_indices = torch.topk(active_basin, k=min(10, len(active_basin)))
        
        prototype_info = []
        for idx, activation in zip(top_indices, top_activations):
            prototype = model.prototype_manager.get_prototype(int(idx))
            if prototype:
                info = {
                    "prototype_id": int(idx),
                    "activation": float(activation),
                    "prototype_count": prototype.count,
                    "creation_step": prototype.creation_step,
                    "metadata": prototype.metadata,
                }
                prototype_info.append(info)
        
        evidence_info["active_prototypes"] = prototype_info
    
    if "evidence" in outputs:
        evidence_info["evidence_embeddings"] = outputs["evidence"].shape
    
    return evidence_info


def extract_saliency_info(outputs: Dict) -> Dict:
    """Extract attention/saliency information."""
    
    saliency_info = {}
    
    if "encoder_outputs" in outputs:
        encoder_outputs = outputs["encoder_outputs"]
        
        if "saliency_weights" in encoder_outputs:
            saliency_weights = encoder_outputs["saliency_weights"][0]  # Remove batch dimension
            
            # Convert to list for JSON serialization
            saliency_info["attention_weights"] = saliency_weights.cpu().tolist()
            saliency_info["attention_shape"] = list(saliency_weights.shape)
    
    return saliency_info


def format_results(results: Dict, task_type: str) -> Dict:
    """Format results for output."""
    
    formatted = {
        "task_type": task_type,
        "input_file": results.get("input_path", ""),
        "timestamp": str(torch.datetime.now()),
    }
    
    # Task-specific formatting
    if task_type == "classification":
        if "predictions" in results:
            formatted["top_predictions"] = results["predictions"]
            formatted["predicted_class"] = results.get("predicted_class")
            formatted["confidence"] = results.get("max_confidence")
        
    elif task_type == "text_generation":
        formatted["query"] = results.get("query", "")
        formatted["generated_text"] = results.get("generated_text", "")
        
    elif task_type == "visual_qa":
        formatted["question"] = results.get("question", "")
        formatted["answer"] = results.get("generated_answer", "")
        formatted["confidence"] = results.get("answer_confidence")
        
    elif task_type == "matching":
        formatted["matches"] = results.get("matches", [])
        formatted["best_match"] = results.get("best_match", {})
    
    # Add additional information
    if "abstained" in results:
        formatted["abstained"] = results["abstained"]
    
    if "evidence" in results:
        formatted["evidence"] = results["evidence"]
    
    if "saliency" in results:
        formatted["saliency"] = results["saliency"]
    
    return formatted


def display_results(results: Dict, task_type: str) -> None:
    """Display results to console."""
    
    click.echo("\n" + "="*50)
    click.echo("INFERENCE RESULTS")
    click.echo("="*50)
    
    click.echo(f"Task Type: {task_type}")
    click.echo(f"Input: {results.get('input_file', 'N/A')}")
    
    if task_type == "classification":
        if "predicted_class" in results:
            click.echo(f"Predicted Class: {results['predicted_class']}")
            click.echo(f"Confidence: {results.get('confidence', 0):.3f}")
        
        if "top_predictions" in results:
            click.echo("\nTop Predictions:")
            for i, pred in enumerate(results["top_predictions"][:3]):
                click.echo(f"  {i+1}. Class {pred['class_id']}: {pred['confidence']:.3f}")
    
    elif task_type == "text_generation":
        click.echo(f"Generated Text: {results.get('generated_text', 'N/A')}")
    
    elif task_type == "visual_qa":
        click.echo(f"Question: {results.get('question', 'N/A')}")
        click.echo(f"Answer: {results.get('answer', 'N/A')}")
    
    elif task_type == "matching":
        best_match = results.get("best_match", {})
        if best_match:
            click.echo(f"Best Match: Prototype {best_match['prototype_id']}")
            click.echo(f"Similarity: {best_match['similarity']:.3f}")
    
    if results.get("abstained", False):
        click.echo("\n⚠️  Model abstained from answering (low confidence)")
    
    click.echo("="*50)


def save_results_to_file(results: Dict, output_path: Path) -> None:
    """Save results to JSON file."""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def save_visualization_outputs(
    results: Dict,
    input_data: Dict,
    output_dir: Path,
) -> None:
    """Save visualization outputs."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original image
    if "original_image" in input_data:
        original_image = input_data["original_image"]
        original_image.save(output_dir / "input_image.png")
    
    # Save attention/saliency maps if available
    if "saliency" in results and "attention_weights" in results["saliency"]:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            attention_weights = np.array(results["saliency"]["attention_weights"])
            
            plt.figure(figsize=(10, 8))
            plt.imshow(attention_weights, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title("Attention Weights")
            plt.savefig(output_dir / "attention_map.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available for attention visualization")


def generate_batch_summary(results: Dict, output_file: Path) -> None:
    """Generate summary for batch processing results."""
    
    total_files = len(results)
    successful = sum(1 for r in results.values() if "error" not in r)
    failed = total_files - successful
    
    summary_lines = [
        "Batch Processing Summary",
        "=" * 40,
        f"Total files processed: {total_files}",
        f"Successful: {successful}",
        f"Failed: {failed}",
        f"Success rate: {successful/total_files*100:.1f}%",
        "",
    ]
    
    # Add error summary if any
    if failed > 0:
        summary_lines.extend([
            "Failed files:",
            "-" * 20,
        ])
        
        for file_path, result in results.items():
            if "error" in result:
                summary_lines.append(f"- {file_path}: {result['error']}")
    
    with open(output_file, 'w') as f:
        f.write("\n".join(summary_lines))


@click.group()
def main() -> None:
    """Neuromorphic Continual Learning Inference CLI."""
    pass


# Add commands to the group
main.add_command(infer)
main.add_command(batch_infer)


if __name__ == "__main__":
    main()
