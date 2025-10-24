"""
Preprocessing Utilities for Neuromorphic Continual Learning.

This module provides utilities for preprocessing various data types including
PDF documents, medical images, and text data for the continual learning system.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)


def render_pdf_page(
    pdf_path: Union[str, Path], 
    page_num: int = 0,
    dpi: int = 200,
    output_format: str = "RGB",
) -> List[Image.Image]:
    """
    Render PDF page(s) to PIL Images.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number to render (0-indexed), or -1 for all pages
        dpi: Resolution for rendering
        output_format: Output image format ("RGB", "L", etc.)
        
    Returns:
        List of PIL Images
    """
    try:
        from pdf2image import convert_from_path
        
        if page_num == -1:
            # Convert all pages
            images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                fmt=output_format.lower(),
            )
        else:
            # Convert specific page
            images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                first_page=page_num + 1,
                last_page=page_num + 1,
                fmt=output_format.lower(),
            )
        
        # Convert to specified format
        converted_images = []
        for img in images:
            if img.mode != output_format:
                img = img.convert(output_format)
            converted_images.append(img)
        
        return converted_images
        
    except ImportError:
        logger.error("pdf2image not available. Install with: pip install pdf2image")
        return []
    except Exception as e:
        logger.error(f"Failed to render PDF {pdf_path}: {e}")
        return []


def extract_text_from_pdf(pdf_path: Union[str, Path], page_num: Optional[int] = None) -> str:
    """
    Extract text from PDF using OCR or text extraction.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Specific page to extract (None for all pages)
        
    Returns:
        Extracted text
    """
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(str(pdf_path))
        text_content = []
        
        if page_num is not None:
            # Extract from specific page
            if 0 <= page_num < len(doc):
                page = doc[page_num]
                text_content.append(page.get_text())
        else:
            # Extract from all pages
            for page in doc:
                text_content.append(page.get_text())
        
        doc.close()
        return "\n".join(text_content)
        
    except ImportError:
        # Fallback to OCR
        logger.warning("PyMuPDF not available, falling back to OCR")
        return extract_text_ocr(pdf_path, page_num)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        return ""


def extract_text_ocr(
    image_or_path: Union[str, Path, Image.Image, np.ndarray],
    page_num: Optional[int] = None,
) -> str:
    """
    Extract text using OCR (Tesseract).
    
    Args:
        image_or_path: Image or path to image/PDF
        page_num: Page number if PDF
        
    Returns:
        Extracted text
    """
    try:
        import pytesseract
        
        # Handle different input types
        if isinstance(image_or_path, (str, Path)):
            path = Path(image_or_path)
            if path.suffix.lower() == '.pdf':
                # Render PDF page first
                images = render_pdf_page(path, page_num or 0)
                if not images:
                    return ""
                image = images[0]
            else:
                # Load image
                image = Image.open(path)
        elif isinstance(image_or_path, np.ndarray):
            image = Image.fromarray(image_or_path)
        else:
            image = image_or_path
        
        # Preprocess image for better OCR
        image = preprocess_for_ocr(image)
        
        # Extract text
        text = pytesseract.image_to_string(image, lang='eng')
        
        return text.strip()
        
    except ImportError:
        logger.error("pytesseract not available. Install with: pip install pytesseract")
        return ""
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image for better OCR performance.
    
    Args:
        image: Input PIL Image
        
    Returns:
        Preprocessed PIL Image
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    
    # Resize if too small
    width, height = image.size
    if width < 300 or height < 300:
        scale_factor = max(300 / width, 300 / height)
        new_size = (int(width * scale_factor), int(height * scale_factor))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image


def normalize_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    target_size: Tuple[int, int] = (224, 224),
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> torch.Tensor:
    """
    Normalize image to standard format.
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Normalized tensor [C, H, W]
    """
    # Default ImageNet normalization
    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            image = Image.fromarray(image, mode='L').convert('RGB')
    elif isinstance(image, torch.Tensor):
        # Convert tensor to PIL
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.dim() == 3 and image.size(0) in [1, 3]:
            image = image.permute(1, 2, 0)
        
        image_np = image.detach().cpu().numpy()
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
    
    # Normalize
    for i in range(3):
        image_tensor[i] = (image_tensor[i] - mean[i]) / std[i]
    
    return image_tensor


def augment_medical_image(
    image: Union[Image.Image, np.ndarray],
    rotation_range: float = 10.0,
    brightness_range: float = 0.2,
    contrast_range: float = 0.2,
    flip_probability: float = 0.5,
) -> Image.Image:
    """
    Apply medical-appropriate image augmentations.
    
    Args:
        image: Input image
        rotation_range: Maximum rotation in degrees
        brightness_range: Brightness adjustment range
        contrast_range: Contrast adjustment range
        flip_probability: Probability of horizontal flip
        
    Returns:
        Augmented PIL Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Random rotation (small angles for medical images)
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        image = image.rotate(angle, fillcolor='black')
    
    # Brightness adjustment
    if brightness_range > 0:
        factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)
    
    # Contrast adjustment
    if contrast_range > 0:
        factor = np.random.uniform(1 - contrast_range, 1 + contrast_range)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(factor)
    
    # Random horizontal flip (appropriate for some medical images)
    if np.random.random() < flip_probability:
        image = ImageOps.mirror(image)
    
    return image


def preprocess_chest_xray(
    image: Union[Image.Image, np.ndarray],
    enhance_contrast: bool = True,
    equalize_histogram: bool = True,
    target_size: Tuple[int, int] = (224, 224),
) -> Image.Image:
    """
    Specialized preprocessing for chest X-ray images.
    
    Args:
        image: Input image
        enhance_contrast: Whether to enhance contrast
        equalize_histogram: Whether to apply histogram equalization
        target_size: Target output size
        
    Returns:
        Preprocessed PIL Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Histogram equalization for better contrast
    if equalize_histogram:
        image = ImageOps.equalize(image)
    
    # Enhance contrast
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
    
    # Convert back to RGB for model compatibility
    image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    return image


def pad_sequence(
    sequences: List[torch.Tensor],
    batch_first: bool = True,
    padding_value: float = 0.0,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of tensors to pad
        batch_first: Whether batch dimension is first
        padding_value: Value to use for padding
        max_length: Maximum sequence length (truncate if longer)
        
    Returns:
        Padded tensor
    """
    if not sequences:
        return torch.empty(0)
    
    # Determine max length
    if max_length is None:
        max_length = max(seq.size(0) for seq in sequences)
    
    # Truncate sequences if they exceed max_length
    truncated_sequences = []
    for seq in sequences:
        if seq.size(0) > max_length:
            truncated_sequences.append(seq[:max_length])
        else:
            truncated_sequences.append(seq)
    
    # Pad sequences
    padded = torch.nn.utils.rnn.pad_sequence(
        truncated_sequences,
        batch_first=batch_first,
        padding_value=padding_value,
    )
    
    return padded


def tokenize_biomedical_text(
    text: str,
    tokenizer,
    max_length: int = 512,
    add_special_tokens: bool = True,
) -> dict:
    """
    Tokenize biomedical text with appropriate preprocessing.
    
    Args:
        text: Input text
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        add_special_tokens: Whether to add special tokens
        
    Returns:
        Tokenization result dictionary
    """
    # Clean text
    text = clean_biomedical_text(text)
    
    # Tokenize
    tokens = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        add_special_tokens=add_special_tokens,
    )
    
    return {
        'input_ids': tokens['input_ids'].squeeze(0),
        'attention_mask': tokens['attention_mask'].squeeze(0),
        'text': text,
    }


def clean_biomedical_text(text: str) -> str:
    """
    Clean biomedical text for better processing.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    import re
    
    if not text or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    
    # Handle common biomedical abbreviations
    # (could be expanded with domain-specific preprocessing)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[,]{2,}', ',', text)
    
    # Strip and return
    return text.strip()


def resize_image_aspect_ratio(
    image: Image.Image,
    target_size: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input PIL Image
        target_size: Target size (width, height)
        fill_color: Color to use for padding
        
    Returns:
        Resized PIL Image
    """
    target_width, target_height = target_size
    
    # Calculate scaling factor
    scale_w = target_width / image.width
    scale_h = target_height / image.height
    scale = min(scale_w, scale_h)
    
    # Calculate new size
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    
    # Resize image
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size
    result = Image.new('RGB', target_size, fill_color)
    
    # Paste resized image in center
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    result.paste(resized, (x_offset, y_offset))
    
    return result


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image: Input image array
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of the grid for histogram equalization
        
    Returns:
        CLAHE-processed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    
    # Convert back to 3-channel if original was 3-channel
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced


def extract_patches(
    image: Union[Image.Image, np.ndarray],
    patch_size: Tuple[int, int] = (224, 224),
    stride: Optional[Tuple[int, int]] = None,
    min_patch_coverage: float = 0.8,
) -> List[Image.Image]:
    """
    Extract patches from large image.
    
    Args:
        image: Input image
        patch_size: Size of each patch
        stride: Stride for patch extraction (defaults to patch_size)
        min_patch_coverage: Minimum coverage to include partial patches
        
    Returns:
        List of image patches
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    if stride is None:
        stride = patch_size
    
    patches = []
    width, height = image.size
    patch_w, patch_h = patch_size
    stride_w, stride_h = stride
    
    for y in range(0, height - patch_h + 1, stride_h):
        for x in range(0, width - patch_w + 1, stride_w):
            # Extract patch
            patch = image.crop((x, y, x + patch_w, y + patch_h))
            patches.append(patch)
    
    # Handle edge patches if needed
    if min_patch_coverage < 1.0:
        # Add patches along edges that don't fully fit
        # (implementation can be added if needed)
        pass
    
    return patches


def create_thumbnail(
    image: Union[Image.Image, np.ndarray],
    max_size: Tuple[int, int] = (128, 128),
) -> Image.Image:
    """
    Create thumbnail while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum thumbnail size
        
    Returns:
        Thumbnail image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Create thumbnail
    thumbnail = image.copy()
    thumbnail.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    return thumbnail


def validate_medical_image(image: Union[Image.Image, np.ndarray]) -> bool:
    """
    Validate that image meets medical imaging requirements.
    
    Args:
        image: Input image
        
    Returns:
        True if image is valid
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Check minimum size
    if image.width < 64 or image.height < 64:
        return False
    
    # Check aspect ratio (should be reasonable)
    aspect_ratio = max(image.width, image.height) / min(image.width, image.height)
    if aspect_ratio > 10:  # Too extreme aspect ratio
        return False
    
    # Check if image is not completely black or white
    if image.mode == 'L':
        extremes = image.getextrema()
        if extremes[0] == extremes[1]:  # All pixels same value
            return False
    
    return True
