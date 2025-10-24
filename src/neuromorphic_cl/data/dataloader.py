"""
Comprehensive Data Loading Module for Neuromorphic Continual Learning.

This module provides data loaders for various medical and scientific datasets
including PubMed documents, MIMIC-CXR images, VQA-RAD, and BioASQ.
It handles multimodal data loading, preprocessing, and streaming for
continual learning scenarios.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms
from transformers import AutoTokenizer

from ..configs.schema import DataConfig, SystemConfig
from ..utils.preprocessing import (
    normalize_image,
    pad_sequence,
    render_pdf_page,
    extract_text_from_pdf,
)

logger = logging.getLogger(__name__)


class BaseMultimodalDataset(Dataset):
    """Base class for multimodal datasets with common functionality."""
    
    def __init__(
        self,
        data_config: DataConfig,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.data_config = data_config
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
        
        self.samples = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data - to be implemented by subclasses."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample - to be implemented by subclasses."""
        raise NotImplementedError
    
    def _preprocess_image(self, image: Union[Image.Image, np.ndarray, str]) -> torch.Tensor:
        """Preprocess image to tensor format."""
        if isinstance(image, str):
            # Load from path
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Convert numpy to PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            default_transform = transforms.Compose([
                transforms.Resize(self.data_config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            image = default_transform(image)
        
        return image
    
    def _preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess text using tokenizer."""
        if not text or pd.isna(text):
            text = ""
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.data_config.text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "text_tokens": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }


class PubMedDataset(BaseMultimodalDataset):
    """
    Dataset for PubMed Central open-access papers.
    
    Loads rendered PDF pages and associated metadata/text.
    """
    
    def __init__(
        self,
        data_config: DataConfig,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        tokenizer: Optional[Any] = None,
        max_papers: Optional[int] = None,
    ):
        self.max_papers = max_papers
        super().__init__(data_config, split, transform, tokenizer)
    
    def _load_data(self) -> None:
        """Load PubMed paper data."""
        if self.data_config.pubmed_path is None:
            logger.warning("PubMed path not specified")
            return
        
        pubmed_path = Path(self.data_config.pubmed_path)
        
        # Look for different data organization patterns
        if (pubmed_path / f"{self.split}.json").exists():
            # JSON metadata file
            self._load_from_json(pubmed_path / f"{self.split}.json")
        elif (pubmed_path / self.split).exists():
            # Directory-based organization
            self._load_from_directory(pubmed_path / self.split)
        else:
            logger.error(f"Could not find PubMed data at {pubmed_path}")
    
    def _load_from_json(self, json_path: Path) -> None:
        """Load data from JSON metadata file."""
        with open(json_path) as f:
            metadata = json.load(f)
        
        for item in metadata:
            if self.max_papers and len(self.samples) >= self.max_papers:
                break
            
            # Expect format: {"paper_id": "PMC123", "pdf_path": "...", "title": "...", ...}
            sample = {
                "paper_id": item["paper_id"],
                "pdf_path": Path(item["pdf_path"]),
                "title": item.get("title", ""),
                "abstract": item.get("abstract", ""),
                "keywords": item.get("keywords", []),
                "journal": item.get("journal", ""),
                "year": item.get("year", 0),
                "doi": item.get("doi", ""),
            }
            self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} PubMed papers from {json_path}")
    
    def _load_from_directory(self, directory: Path) -> None:
        """Load data from directory structure."""
        pdf_files = list(directory.glob("**/*.pdf"))
        
        for pdf_path in pdf_files:
            if self.max_papers and len(self.samples) >= self.max_papers:
                break
            
            # Extract paper ID from filename
            paper_id = pdf_path.stem
            
            # Look for associated metadata
            metadata_path = pdf_path.with_suffix(".json")
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            
            sample = {
                "paper_id": paper_id,
                "pdf_path": pdf_path,
                "title": metadata.get("title", ""),
                "abstract": metadata.get("abstract", ""),
                "keywords": metadata.get("keywords", []),
                "journal": metadata.get("journal", ""),
                "year": metadata.get("year", 0),
                "doi": metadata.get("doi", ""),
            }
            self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} PubMed papers from {directory}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a PubMed paper sample."""
        sample = self.samples[idx]
        
        # Render PDF page(s) to images
        try:
            # For now, just use first page
            page_images = render_pdf_page(sample["pdf_path"], page_num=0)
            if not page_images:
                # Fallback: create blank image
                page_image = Image.new("RGB", self.data_config.image_size, color="white")
            else:
                page_image = page_images[0]
            
            pixel_values = self._preprocess_image(page_image)
            
        except Exception as e:
            logger.warning(f"Failed to render PDF {sample['pdf_path']}: {e}")
            # Fallback: blank image
            page_image = Image.new("RGB", self.data_config.image_size, color="white")
            pixel_values = self._preprocess_image(page_image)
        
        # Combine text fields
        text_content = f"{sample['title']} {sample['abstract']}"
        text_data = self._preprocess_text(text_content)
        
        return {
            "pixel_values": pixel_values,
            "text_tokens": text_data["text_tokens"],
            "attention_mask": text_data["attention_mask"],
            "paper_id": sample["paper_id"],
            "metadata": {
                "title": sample["title"],
                "journal": sample["journal"],
                "year": sample["year"],
                "keywords": sample["keywords"],
            },
            "task_type": "document_understanding",
        }


class MIMICCXRDataset(BaseMultimodalDataset):
    """
    Dataset for MIMIC-CXR chest X-ray images with reports.
    """
    
    def __init__(
        self,
        data_config: DataConfig,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        tokenizer: Optional[Any] = None,
        view_types: Optional[List[str]] = None,
    ):
        self.view_types = view_types or ["PA", "AP"]  # Posteroanterior, Anteroposterior
        super().__init__(data_config, split, transform, tokenizer)
    
    def _load_data(self) -> None:
        """Load MIMIC-CXR data."""
        if self.data_config.mimic_cxr_path is None:
            logger.warning("MIMIC-CXR path not specified")
            return
        
        mimic_path = Path(self.data_config.mimic_cxr_path)
        
        # Load metadata CSV files
        metadata_file = mimic_path / "mimic-cxr-2.0.0-metadata.csv"
        chexpert_file = mimic_path / "mimic-cxr-2.0.0-chexpert.csv"
        split_file = mimic_path / "mimic-cxr-2.0.0-split.csv"
        
        if not all(f.exists() for f in [metadata_file, split_file]):
            logger.error(f"Missing MIMIC-CXR metadata files in {mimic_path}")
            return
        
        # Load dataframes
        metadata_df = pd.read_csv(metadata_file)
        split_df = pd.read_csv(split_file)
        
        # Load CheXpert labels if available
        chexpert_df = None
        if chexpert_file.exists():
            chexpert_df = pd.read_csv(chexpert_file)
        
        # Filter by split
        split_df = split_df[split_df["split"] == self.split]
        
        # Merge dataframes
        data_df = metadata_df.merge(split_df, on=["dicom_id"])
        if chexpert_df is not None:
            data_df = data_df.merge(chexpert_df, on=["study_id"], how="left")
        
        # Filter by view type
        data_df = data_df[data_df["ViewPosition"].isin(self.view_types)]
        
        # Convert to samples
        for _, row in data_df.iterrows():
            # Construct image path
            subject_id = f"p{str(row['subject_id'])[:2]}/p{row['subject_id']}"
            study_id = f"s{row['study_id']}"
            dicom_id = f"{row['dicom_id']}.jpg"
            
            image_path = mimic_path / "files" / subject_id / study_id / dicom_id
            
            if not image_path.exists():
                continue
            
            # Extract labels (CheXpert)
            labels = {}
            if chexpert_df is not None:
                chexpert_cols = [
                    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
                    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
                    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                    "Pleural Other", "Fracture", "Support Devices"
                ]
                for col in chexpert_cols:
                    if col in row:
                        labels[col] = row[col] if pd.notna(row[col]) else 0
            
            sample = {
                "dicom_id": row["dicom_id"],
                "study_id": row["study_id"],
                "subject_id": row["subject_id"],
                "image_path": image_path,
                "view_position": row["ViewPosition"],
                "labels": labels,
            }
            self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} MIMIC-CXR images")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a MIMIC-CXR sample."""
        sample = self.samples[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            pixel_values = self._preprocess_image(image)
        except Exception as e:
            logger.warning(f"Failed to load image {sample['image_path']}: {e}")
            # Fallback: blank image
            image = Image.new("RGB", self.data_config.image_size, color="black")
            pixel_values = self._preprocess_image(image)
        
        # Create text description
        view_text = f"Chest X-ray {sample['view_position']} view"
        text_data = self._preprocess_text(view_text)
        
        # Convert labels to tensor
        label_values = list(sample["labels"].values())
        if label_values:
            # Multi-label classification
            labels = torch.tensor(label_values, dtype=torch.float32)
        else:
            # No labels available
            labels = torch.zeros(14, dtype=torch.float32)  # 14 CheXpert classes
        
        return {
            "pixel_values": pixel_values,
            "text_tokens": text_data["text_tokens"],
            "attention_mask": text_data["attention_mask"],
            "labels": labels,
            "dicom_id": sample["dicom_id"],
            "metadata": {
                "view_position": sample["view_position"],
                "study_id": sample["study_id"],
                "subject_id": sample["subject_id"],
            },
            "task_type": "medical_imaging",
        }


class VQARADDataset(BaseMultimodalDataset):
    """
    Dataset for VQA-RAD (Visual Question Answering in Radiology).
    """
    
    def __init__(
        self,
        data_config: DataConfig,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        tokenizer: Optional[Any] = None,
    ):
        super().__init__(data_config, split, transform, tokenizer)
    
    def _load_data(self) -> None:
        """Load VQA-RAD data."""
        if self.data_config.vqa_rad_path is None:
            logger.warning("VQA-RAD path not specified")
            return
        
        vqa_path = Path(self.data_config.vqa_rad_path)
        
        # Load JSON file
        json_file = vqa_path / f"{self.split}.json"
        if not json_file.exists():
            # Try alternative naming
            json_file = vqa_path / f"vqa_rad_{self.split}.json"
        
        if not json_file.exists():
            logger.error(f"Could not find VQA-RAD data file at {vqa_path}")
            return
        
        with open(json_file) as f:
            data = json.load(f)
        
        # Convert to samples
        for item in data:
            image_path = vqa_path / "images" / item["image_name"]
            
            if not image_path.exists():
                continue
            
            sample = {
                "image_path": image_path,
                "image_name": item["image_name"],
                "question": item["question"],
                "answer": item["answer"],
                "question_type": item.get("question_type", "unknown"),
                "answer_type": item.get("answer_type", "unknown"),
            }
            self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} VQA-RAD samples")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a VQA-RAD sample."""
        sample = self.samples[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            pixel_values = self._preprocess_image(image)
        except Exception as e:
            logger.warning(f"Failed to load image {sample['image_path']}: {e}")
            image = Image.new("RGB", self.data_config.image_size, color="black")
            pixel_values = self._preprocess_image(image)
        
        # Preprocess question
        question_data = self._preprocess_text(sample["question"])
        
        # Preprocess answer for training
        answer_data = self._preprocess_text(sample["answer"])
        
        return {
            "pixel_values": pixel_values,
            "text_tokens": question_data["text_tokens"],
            "attention_mask": question_data["attention_mask"],
            "answer_tokens": answer_data["text_tokens"],
            "answer_mask": answer_data["attention_mask"],
            "question": sample["question"],
            "answer": sample["answer"],
            "metadata": {
                "image_name": sample["image_name"],
                "question_type": sample["question_type"],
                "answer_type": sample["answer_type"],
            },
            "task_type": "visual_question_answering",
        }


class BioASQDataset(BaseMultimodalDataset):
    """
    Dataset for BioASQ biomedical question answering.
    """
    
    def __init__(
        self,
        data_config: DataConfig,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        tokenizer: Optional[Any] = None,
        task_type: str = "factoid",  # factoid, list, yesno, summary
    ):
        self.task_type = task_type
        super().__init__(data_config, split, transform, tokenizer)
    
    def _load_data(self) -> None:
        """Load BioASQ data."""
        if self.data_config.bioasq_path is None:
            logger.warning("BioASQ path not specified")
            return
        
        bioasq_path = Path(self.data_config.bioasq_path)
        
        # BioASQ typically has JSON files for different task types
        json_file = bioasq_path / f"{self.split}_{self.task_type}.json"
        if not json_file.exists():
            json_file = bioasq_path / f"BioASQ-task{self.task_type}-{self.split}.json"
        
        if not json_file.exists():
            logger.error(f"Could not find BioASQ data file at {bioasq_path}")
            return
        
        with open(json_file) as f:
            data = json.load(f)
        
        # Extract questions
        questions = data.get("questions", [])
        
        for item in questions:
            # Skip if no answer provided (for training)
            if self.split == "train" and not item.get("exact_answer"):
                continue
            
            sample = {
                "question_id": item.get("id", ""),
                "question": item.get("body", ""),
                "question_type": item.get("type", self.task_type),
                "concepts": item.get("concepts", []),
                "documents": item.get("documents", []),
                "snippets": item.get("snippets", []),
                "exact_answer": item.get("exact_answer", []),
                "ideal_answer": item.get("ideal_answer", ""),
            }
            self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} BioASQ {self.task_type} questions")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a BioASQ sample."""
        sample = self.samples[idx]
        
        # For BioASQ, we create a text-only representation
        # (could be extended to include document images)
        question_text = sample["question"]
        
        # Include context from snippets
        context = " ".join([s.get("text", "") for s in sample["snippets"][:3]])
        full_text = f"Question: {question_text} Context: {context}"
        
        text_data = self._preprocess_text(full_text)
        
        # Create a placeholder image (could be document rendering)
        placeholder_image = Image.new("RGB", self.data_config.image_size, color="white")
        pixel_values = self._preprocess_image(placeholder_image)
        
        # Process answer
        if sample["exact_answer"]:
            answer = sample["exact_answer"][0] if isinstance(sample["exact_answer"], list) else sample["exact_answer"]
        else:
            answer = sample["ideal_answer"]
        
        answer_data = self._preprocess_text(str(answer))
        
        return {
            "pixel_values": pixel_values,
            "text_tokens": text_data["text_tokens"],
            "attention_mask": text_data["attention_mask"],
            "answer_tokens": answer_data["text_tokens"],
            "answer_mask": answer_data["attention_mask"],
            "question": question_text,
            "answer": str(answer),
            "metadata": {
                "question_id": sample["question_id"],
                "question_type": sample["question_type"],
                "concepts": sample["concepts"],
            },
            "task_type": "biomedical_qa",
        }


class ContinualLearningDataset(IterableDataset):
    """
    Iterable dataset for continual learning scenarios.
    
    Provides a stream of data from multiple datasets in sequence,
    simulating the continual learning setting.
    """
    
    def __init__(
        self,
        datasets: List[BaseMultimodalDataset],
        task_schedule: Optional[List[int]] = None,
        buffer_size: int = 1000,
        shuffle_buffer: bool = True,
    ):
        self.datasets = datasets
        self.task_schedule = task_schedule or list(range(len(datasets)))
        self.buffer_size = buffer_size
        self.shuffle_buffer = shuffle_buffer
        
        self.current_task = 0
        self.current_dataset_idx = 0
        self.samples_seen = 0
        
        # Create sample buffer
        self.buffer = []
    
    def __iter__(self):
        """Iterate through the continual learning stream."""
        while self.current_task < len(self.task_schedule):
            dataset_idx = self.task_schedule[self.current_task]
            dataset = self.datasets[dataset_idx]
            
            # Iterate through current dataset
            for sample in dataset:
                # Add task information
                sample["task_id"] = self.current_task
                sample["dataset_id"] = dataset_idx
                
                # Add to buffer
                self.buffer.append(sample)
                
                # Yield from buffer when full
                if len(self.buffer) >= self.buffer_size:
                    if self.shuffle_buffer:
                        np.random.shuffle(self.buffer)
                    
                    while self.buffer:
                        yield self.buffer.pop(0)
                
                self.samples_seen += 1
            
            # Move to next task
            self.current_task += 1
        
        # Yield remaining samples in buffer
        if self.shuffle_buffer:
            np.random.shuffle(self.buffer)
        
        while self.buffer:
            yield self.buffer.pop(0)
    
    def switch_task(self, task_id: int) -> None:
        """Switch to a specific task."""
        if task_id < len(self.task_schedule):
            self.current_task = task_id
            self.current_dataset_idx = self.task_schedule[task_id]
            self.buffer.clear()


class NeuromorphicDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for neuromorphic continual learning.
    
    Handles loading and preprocessing of multiple datasets for
    continual learning scenarios.
    """
    
    def __init__(self, config: SystemConfig):
        super().__init__()
        self.config = config
        self.data_config = config.data
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
        
        # Create transforms
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()
        
        # Initialize datasets
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        
    def _create_train_transform(self) -> transforms.Compose:
        """Create training data transforms with augmentation."""
        return transforms.Compose([
            transforms.Resize(self.data_config.image_size),
            transforms.RandomRotation(degrees=5),
            transforms.RandomResizedCrop(
                self.data_config.image_size,
                scale=(0.9, 1.0),
            ),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    def _create_val_transform(self) -> transforms.Compose:
        """Create validation/test data transforms without augmentation."""
        return transforms.Compose([
            transforms.Resize(self.data_config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for different stages."""
        if stage == "fit" or stage is None:
            # Setup training and validation datasets
            self._setup_train_datasets()
            self._setup_val_datasets()
        
        if stage == "test" or stage is None:
            # Setup test datasets
            self._setup_test_datasets()
    
    def _setup_train_datasets(self) -> None:
        """Setup training datasets."""
        # PubMed dataset
        if self.data_config.pubmed_path:
            pubmed_train = PubMedDataset(
                data_config=self.data_config,
                split="train",
                transform=self.train_transform,
                tokenizer=self.tokenizer,
            )
            self.train_datasets.append(pubmed_train)
        
        # MIMIC-CXR dataset
        if self.data_config.mimic_cxr_path:
            mimic_train = MIMICCXRDataset(
                data_config=self.data_config,
                split="train",
                transform=self.train_transform,
                tokenizer=self.tokenizer,
            )
            self.train_datasets.append(mimic_train)
        
        # VQA-RAD dataset
        if self.data_config.vqa_rad_path:
            vqa_train = VQARADDataset(
                data_config=self.data_config,
                split="train",
                transform=self.train_transform,
                tokenizer=self.tokenizer,
            )
            self.train_datasets.append(vqa_train)
        
        # BioASQ dataset
        if self.data_config.bioasq_path:
            bioasq_train = BioASQDataset(
                data_config=self.data_config,
                split="train",
                transform=self.train_transform,
                tokenizer=self.tokenizer,
            )
            self.train_datasets.append(bioasq_train)
        
        logger.info(f"Setup {len(self.train_datasets)} training datasets")
    
    def _setup_val_datasets(self) -> None:
        """Setup validation datasets."""
        # Similar to training but with validation split
        if self.data_config.pubmed_path:
            pubmed_val = PubMedDataset(
                data_config=self.data_config,
                split="val",
                transform=self.val_transform,
                tokenizer=self.tokenizer,
            )
            self.val_datasets.append(pubmed_val)
        
        if self.data_config.mimic_cxr_path:
            mimic_val = MIMICCXRDataset(
                data_config=self.data_config,
                split="validate",  # MIMIC-CXR uses "validate"
                transform=self.val_transform,
                tokenizer=self.tokenizer,
            )
            self.val_datasets.append(mimic_val)
        
        if self.data_config.vqa_rad_path:
            vqa_val = VQARADDataset(
                data_config=self.data_config,
                split="val",
                transform=self.val_transform,
                tokenizer=self.tokenizer,
            )
            self.val_datasets.append(vqa_val)
        
        if self.data_config.bioasq_path:
            bioasq_val = BioASQDataset(
                data_config=self.data_config,
                split="val",
                transform=self.val_transform,
                tokenizer=self.tokenizer,
            )
            self.val_datasets.append(bioasq_val)
        
        logger.info(f"Setup {len(self.val_datasets)} validation datasets")
    
    def _setup_test_datasets(self) -> None:
        """Setup test datasets."""
        # Similar to validation but with test split
        if self.data_config.pubmed_path:
            pubmed_test = PubMedDataset(
                data_config=self.data_config,
                split="test",
                transform=self.val_transform,
                tokenizer=self.tokenizer,
            )
            self.test_datasets.append(pubmed_test)
        
        if self.data_config.mimic_cxr_path:
            mimic_test = MIMICCXRDataset(
                data_config=self.data_config,
                split="test",
                transform=self.val_transform,
                tokenizer=self.tokenizer,
            )
            self.test_datasets.append(mimic_test)
        
        if self.data_config.vqa_rad_path:
            vqa_test = VQARADDataset(
                data_config=self.data_config,
                split="test",
                transform=self.val_transform,
                tokenizer=self.tokenizer,
            )
            self.test_datasets.append(vqa_test)
        
        if self.data_config.bioasq_path:
            bioasq_test = BioASQDataset(
                data_config=self.data_config,
                split="test",
                transform=self.val_transform,
                tokenizer=self.tokenizer,
            )
            self.test_datasets.append(bioasq_test)
        
        logger.info(f"Setup {len(self.test_datasets)} test datasets")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if not self.train_datasets:
            raise ValueError("No training datasets available")
        
        # Create continual learning dataset
        continual_dataset = ContinualLearningDataset(
            datasets=self.train_datasets,
            buffer_size=self.data_config.streaming_buffer_size,
            shuffle_buffer=True,
        )
        
        return DataLoader(
            continual_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            collate_fn=self._collate_fn,
        )
    
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Create validation dataloader(s)."""
        if not self.val_datasets:
            return []
        
        dataloaders = []
        for dataset in self.val_datasets:
            loader = DataLoader(
                dataset,
                batch_size=self.data_config.batch_size,
                num_workers=self.data_config.num_workers,
                pin_memory=self.data_config.pin_memory,
                shuffle=False,
                collate_fn=self._collate_fn,
            )
            dataloaders.append(loader)
        
        return dataloaders[0] if len(dataloaders) == 1 else dataloaders
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Create test dataloader(s)."""
        if not self.test_datasets:
            return []
        
        dataloaders = []
        for dataset in self.test_datasets:
            loader = DataLoader(
                dataset,
                batch_size=self.data_config.batch_size,
                num_workers=self.data_config.num_workers,
                pin_memory=self.data_config.pin_memory,
                shuffle=False,
                collate_fn=self._collate_fn,
            )
            dataloaders.append(loader)
        
        return dataloaders
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for multimodal data."""
        collated = {}
        
        # Handle tensor fields
        tensor_fields = ["pixel_values", "text_tokens", "attention_mask", "labels"]
        for field in tensor_fields:
            if field in batch[0]:
                try:
                    collated[field] = torch.stack([item[field] for item in batch])
                except:
                    # Handle variable-length sequences
                    sequences = [item[field] for item in batch]
                    collated[field] = torch.nn.utils.rnn.pad_sequence(
                        sequences, batch_first=True, padding_value=0
                    )
        
        # Handle string fields
        string_fields = ["question", "answer", "task_type", "paper_id", "dicom_id"]
        for field in string_fields:
            if field in batch[0]:
                collated[field] = [item[field] for item in batch]
        
        # Handle metadata
        if "metadata" in batch[0]:
            collated["metadata"] = [item["metadata"] for item in batch]
        
        return collated
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about loaded datasets."""
        info = {
            "num_train_datasets": len(self.train_datasets),
            "num_val_datasets": len(self.val_datasets),
            "num_test_datasets": len(self.test_datasets),
            "train_sizes": [len(ds) for ds in self.train_datasets],
            "val_sizes": [len(ds) for ds in self.val_datasets],
            "test_sizes": [len(ds) for ds in self.test_datasets],
        }
        
        return info
