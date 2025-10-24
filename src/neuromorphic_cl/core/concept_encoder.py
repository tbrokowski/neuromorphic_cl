"""
Concept Encoder (CE) Module for Neuromorphic Continual Learning.

This module implements multimodal document understanding and concept-level
representation learning using various vision encoders with CLIP-style
contrastive projection heads.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    DonutProcessor,
    LayoutLMv3Model,
    LayoutLMv3Processor,
    VisionEncoderDecoderModel,
    ViTModel,
)

from ..configs.schema import ConceptEncoderConfig, EncoderType
from ..utils.registry import Registry

logger = logging.getLogger(__name__)

# Registry for encoder implementations
ENCODER_REGISTRY = Registry("concept_encoders")


class ConceptProjectionHead(nn.Module):
    """CLIP-style contrastive projection head."""
    
    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(input_dim, projection_dim)
        
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        ]
        
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        
        layers.extend([
            nn.Linear(hidden_dim, projection_dim),
        ])
        
        self.projection = nn.Sequential(*layers)
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to concept space."""
        projected = self.projection(x)
        return F.normalize(projected, dim=-1)


@ENCODER_REGISTRY.register("deepseek_ocr")
class DeepSeekOCREncoder(nn.Module):
    """DeepSeek-OCR based visual encoder for document understanding."""
    
    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        self.config = config
        
        try:
            # Try to load DeepSeek-OCR model
            # This is a placeholder - actual implementation depends on DeepSeek-OCR API
            from deepseek_ocr import DeepSeekOCRModel
            
            self.backbone = DeepSeekOCRModel.from_pretrained(
                config.deepseek_model_path or "deepseek/ocr-base"
            )
            backbone_dim = self.backbone.config.hidden_size
            
        except ImportError:
            logger.warning(
                "DeepSeek-OCR not available, falling back to ViT"
            )
            # Fallback to ViT
            self.backbone = ViTModel.from_pretrained(config.vit_model_name)
            backbone_dim = self.backbone.config.hidden_size
        
        # Freeze backbone if specified
        if config.freeze_backbone:
            self._freeze_backbone_layers()
        
        self.projection_head = ConceptProjectionHead(
            input_dim=backbone_dim,
            projection_dim=config.projection_dim,
            dropout_rate=config.dropout_rate,
        )
        
        # Token attention for saliency mapping
        self.token_attention = nn.MultiheadAttention(
            embed_dim=backbone_dim,
            num_heads=8,
            dropout=config.dropout_rate,
            batch_first=True,
        )
    
    def _freeze_backbone_layers(self) -> None:
        """Freeze backbone parameters except for last N blocks."""
        total_layers = len(self.backbone.encoder.layer)
        freeze_until = total_layers - self.config.unfreeze_last_n_blocks
        
        # Freeze embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze early encoder layers
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(
        self, 
        pixel_values: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_saliency: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the encoder.
        
        Args:
            pixel_values: Image tensor [B, C, H, W]
            text_tokens: Optional text token IDs [B, T]
            attention_mask: Optional attention mask [B, T]
            return_saliency: Whether to compute token-level saliency
            
        Returns:
            Tuple of (concept_embedding, saliency_weights)
        """
        batch_size = pixel_values.size(0)
        
        # Extract visual features
        visual_outputs = self.backbone(pixel_values=pixel_values)
        
        # Get sequence of visual tokens
        if hasattr(visual_outputs, 'last_hidden_state'):
            visual_tokens = visual_outputs.last_hidden_state  # [B, T, D]
        else:
            # Handle DeepSeek-OCR specific output format
            visual_tokens = visual_outputs.hidden_states[-1]
        
        # Pool visual tokens to get page-level representation
        if return_saliency:
            # Use attention pooling for interpretability
            pooled_visual, attention_weights = self.token_attention(
                visual_tokens.mean(dim=1, keepdim=True),  # Query: mean pooled
                visual_tokens,  # Key & Value: all tokens
                visual_tokens,
            )
            pooled_visual = pooled_visual.squeeze(1)  # [B, D]
            saliency_weights = attention_weights.squeeze(1)  # [B, T]
        else:
            # Simple mean pooling
            pooled_visual = visual_tokens.mean(dim=1)  # [B, D]
            saliency_weights = None
        
        # Project to concept space
        concept_embedding = self.projection_head(pooled_visual)
        
        return concept_embedding, saliency_weights


@ENCODER_REGISTRY.register("vit")
class ViTConceptEncoder(nn.Module):
    """Vision Transformer based concept encoder."""
    
    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        self.config = config
        
        self.backbone = ViTModel.from_pretrained(config.vit_model_name)
        backbone_dim = self.backbone.config.hidden_size
        
        if config.freeze_backbone:
            self._freeze_backbone_layers()
        
        self.projection_head = ConceptProjectionHead(
            input_dim=backbone_dim,
            projection_dim=config.projection_dim,
            dropout_rate=config.dropout_rate,
        )
    
    def _freeze_backbone_layers(self) -> None:
        """Freeze backbone parameters except for last N blocks."""
        total_layers = len(self.backbone.encoder.layer)
        freeze_until = total_layers - self.config.unfreeze_last_n_blocks
        
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_saliency: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through ViT encoder."""
        outputs = self.backbone(pixel_values=pixel_values)
        
        # Use CLS token as document representation
        pooled_output = outputs.pooler_output  # [B, D]
        
        # Project to concept space
        concept_embedding = self.projection_head(pooled_output)
        
        # Extract attention weights for saliency if requested
        saliency_weights = None
        if return_saliency and hasattr(outputs, 'attentions'):
            # Average attention across heads and layers for patch-level saliency
            attentions = torch.stack(outputs.attentions)  # [L, B, H, T, T]
            # Focus on CLS token attention to patches
            cls_attention = attentions[:, :, :, 0, 1:]  # [L, B, H, T-1]
            saliency_weights = cls_attention.mean(dim=(0, 2))  # [B, T-1]
        
        return concept_embedding, saliency_weights


@ENCODER_REGISTRY.register("donut")
class DonutConceptEncoder(nn.Module):
    """Donut (Document Understanding Transformer) based encoder."""
    
    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        self.config = config
        
        self.model = VisionEncoderDecoderModel.from_pretrained(
            config.donut_model_name
        )
        self.processor = DonutProcessor.from_pretrained(config.donut_model_name)
        
        # Use encoder part only
        self.backbone = self.model.encoder
        backbone_dim = self.backbone.config.hidden_size
        
        if config.freeze_backbone:
            self._freeze_backbone_layers()
        
        self.projection_head = ConceptProjectionHead(
            input_dim=backbone_dim,
            projection_dim=config.projection_dim,
            dropout_rate=config.dropout_rate,
        )
    
    def _freeze_backbone_layers(self) -> None:
        """Freeze backbone parameters."""
        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_saliency: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through Donut encoder."""
        # Extract features using encoder
        encoder_outputs = self.backbone(pixel_values=pixel_values)
        
        # Pool encoder outputs
        hidden_states = encoder_outputs.last_hidden_state  # [B, T, D]
        pooled_output = hidden_states.mean(dim=1)  # [B, D]
        
        # Project to concept space
        concept_embedding = self.projection_head(pooled_output)
        
        # Simple attention-based saliency
        saliency_weights = None
        if return_saliency:
            # Compute attention weights based on feature magnitude
            attention_scores = hidden_states.norm(dim=-1)  # [B, T]
            saliency_weights = F.softmax(attention_scores, dim=-1)
        
        return concept_embedding, saliency_weights


@ENCODER_REGISTRY.register("layoutlmv3")
class LayoutLMv3ConceptEncoder(nn.Module):
    """LayoutLMv3 based multimodal encoder for documents."""
    
    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        self.config = config
        
        self.backbone = LayoutLMv3Model.from_pretrained(config.layoutlm_model_name)
        self.processor = LayoutLMv3Processor.from_pretrained(config.layoutlm_model_name)
        
        backbone_dim = self.backbone.config.hidden_size
        
        if config.freeze_backbone:
            self._freeze_backbone_layers()
        
        self.projection_head = ConceptProjectionHead(
            input_dim=backbone_dim,
            projection_dim=config.projection_dim,
            dropout_rate=config.dropout_rate,
        )
    
    def _freeze_backbone_layers(self) -> None:
        """Freeze backbone parameters."""
        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_saliency: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through LayoutLMv3 encoder."""
        # LayoutLMv3 requires both visual and textual inputs
        if text_tokens is None:
            # Create dummy text tokens if not provided
            batch_size = pixel_values.size(0)
            text_tokens = torch.zeros(
                (batch_size, 1), 
                dtype=torch.long, 
                device=pixel_values.device
            )
            attention_mask = torch.ones_like(text_tokens)
        
        outputs = self.backbone(
            input_ids=text_tokens,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        
        # Use CLS token representation
        pooled_output = outputs.pooler_output  # [B, D]
        
        # Project to concept space
        concept_embedding = self.projection_head(pooled_output)
        
        # Extract attention for saliency
        saliency_weights = None
        if return_saliency and hasattr(outputs, 'attentions'):
            # Get cross-attention between text and vision
            attentions = outputs.attentions[-1]  # Last layer attention [B, H, T, T]
            # Average across heads for simplicity
            saliency_weights = attentions.mean(dim=1)[:, 0, :]  # [B, T] CLS attention
        
        return concept_embedding, saliency_weights


class ConceptEncoder(nn.Module):
    """
    Main Concept Encoder module that handles multimodal document understanding.
    
    This module serves as the entry point for converting raw visual and textual
    inputs into high-level concept embeddings suitable for prototype clustering
    and neuromorphic memory storage.
    """
    
    def __init__(self, config: ConceptEncoderConfig):
        super().__init__()
        self.config = config
        
        # Initialize the appropriate encoder based on config
        if config.encoder_type not in ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown encoder type: {config.encoder_type}. "
                f"Available: {list(ENCODER_REGISTRY.keys())}"
            )
        
        self.encoder = ENCODER_REGISTRY[config.encoder_type](config)
        
        # Optional text encoder for multimodal alignment
        self.text_encoder = None
        if hasattr(config, 'enable_text_encoder') and config.enable_text_encoder:
            self.text_encoder = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            self.text_projection = ConceptProjectionHead(
                input_dim=self.text_encoder.config.hidden_size,
                projection_dim=config.projection_dim,
                dropout_rate=config.dropout_rate,
            )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_saliency: bool = True,
        return_text_embedding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the concept encoder.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            text_tokens: Optional text token IDs [B, T]
            attention_mask: Optional attention mask [B, T]
            return_saliency: Whether to compute saliency maps
            return_text_embedding: Whether to encode text separately
            
        Returns:
            Dictionary containing:
                - concept_embedding: Main concept vectors [B, D]
                - saliency_weights: Token-level attention weights [B, T]
                - text_embedding: Optional text embeddings [B, D]
        """
        # Encode visual content
        concept_embedding, saliency_weights = self.encoder(
            pixel_values=pixel_values,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            return_saliency=return_saliency,
        )
        
        outputs = {
            "concept_embedding": concept_embedding,
        }
        
        if return_saliency and saliency_weights is not None:
            outputs["saliency_weights"] = saliency_weights
        
        # Optionally encode text separately for contrastive learning
        if return_text_embedding and self.text_encoder is not None and text_tokens is not None:
            text_outputs = self.text_encoder(
                input_ids=text_tokens,
                attention_mask=attention_mask,
            )
            text_pooled = text_outputs.pooler_output
            text_embedding = self.text_projection(text_pooled)
            outputs["text_embedding"] = text_embedding
        
        return outputs
    
    def encode_pages(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for encoding multiple pages.
        
        Args:
            images: Batch of page images [B, C, H, W]
            
        Returns:
            Tuple of (concept_embeddings, pooled_page_embedding)
        """
        with torch.no_grad():
            outputs = self.forward(images, return_saliency=False)
            concept_embeddings = outputs["concept_embedding"]
            
            # Pool across pages using attention
            if concept_embeddings.size(0) > 1:
                # Multi-page document - use attention pooling
                query = concept_embeddings.mean(dim=0, keepdim=True)  # [1, D]
                scores = torch.matmul(query, concept_embeddings.t())  # [1, B]
                weights = F.softmax(scores, dim=-1)  # [1, B]
                pooled_embedding = torch.matmul(weights, concept_embeddings)  # [1, D]
            else:
                pooled_embedding = concept_embeddings
            
            return concept_embeddings, pooled_embedding.squeeze(0)
    
    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.config.projection_dim
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to save memory."""
        if hasattr(self.encoder, 'backbone'):
            if hasattr(self.encoder.backbone, 'gradient_checkpointing_enable'):
                self.encoder.backbone.gradient_checkpointing_enable()
    
    def get_trainable_parameters(self) -> Dict[str, nn.Parameter]:
        """Get parameters that should be trained."""
        trainable_params = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param
        
        return trainable_params
