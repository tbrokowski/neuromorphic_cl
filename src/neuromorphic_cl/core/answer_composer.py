"""
Answer Composer (AC) Module for Neuromorphic Continual Learning.

This module integrates SNN activations with reasoning and generation
modules to produce human-readable outputs. It handles both structured
tasks (classification, matching) and natural language generation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from ..configs.schema import AnswerComposerConfig
from ..core.prototype_manager import Prototype

logger = logging.getLogger(__name__)


class EvidenceExtractor(nn.Module):
    """Extract evidence from activated prototypes for reasoning."""
    
    def __init__(
        self,
        prototype_dim: int,
        hidden_dim: int = 256,
        num_evidence_slots: int = 5,
    ):
        super().__init__()
        
        self.prototype_dim = prototype_dim
        self.hidden_dim = hidden_dim
        self.num_evidence_slots = num_evidence_slots
        
        # Attention mechanism for evidence selection
        self.evidence_attention = nn.MultiheadAttention(
            embed_dim=prototype_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        
        # Evidence encoding
        self.evidence_encoder = nn.Sequential(
            nn.Linear(prototype_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Evidence slot embeddings
        self.slot_embeddings = nn.Parameter(
            torch.randn(num_evidence_slots, prototype_dim)
        )
    
    def forward(
        self,
        prototype_embeddings: torch.Tensor,
        activation_weights: torch.Tensor,
        query_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract relevant evidence from activated prototypes.
        
        Args:
            prototype_embeddings: Embeddings of activated prototypes [B, N, D]
            activation_weights: SNN activation strengths [B, N]
            query_embedding: Optional query context [B, D]
            
        Returns:
            Dictionary containing evidence representations
        """
        batch_size, num_prototypes, embed_dim = prototype_embeddings.shape
        
        # Weight prototypes by activation strength
        weighted_embeddings = (
            prototype_embeddings * activation_weights.unsqueeze(-1)
        )
        
        # Use query for attention if available, otherwise use learned slots
        if query_embedding is not None:
            query = query_embedding.unsqueeze(1)  # [B, 1, D]
        else:
            # Expand slot embeddings for batch
            query = self.slot_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )  # [B, num_slots, D]
        
        # Extract evidence using attention
        evidence, attention_weights = self.evidence_attention(
            query=query,
            key=weighted_embeddings,
            value=weighted_embeddings,
        )
        
        # Encode evidence
        encoded_evidence = self.evidence_encoder(evidence)
        
        return {
            "evidence": encoded_evidence,  # [B, num_slots, hidden_dim]
            "attention_weights": attention_weights,  # [B, num_slots, N]
            "raw_evidence": evidence,  # [B, num_slots, D]
        }


class StructuredOutputHead(nn.Module):
    """Head for structured prediction tasks (classification, matching)."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, evidence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for structured prediction.
        
        Args:
            evidence: Evidence representations [B, num_slots, D]
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Pool evidence across slots
        pooled_evidence = evidence.mean(dim=1)  # [B, D]
        
        # Classification
        logits = self.classifier(pooled_evidence)
        predictions = F.softmax(logits, dim=-1)
        
        # Confidence
        confidence = self.confidence_head(pooled_evidence)
        
        return {
            "logits": logits,
            "predictions": predictions,
            "confidence": confidence,
        }


class TextGenerator(nn.Module):
    """Text generation module for natural language responses."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        evidence_dim: int = 256,
        max_length: int = 512,
    ):
        super().__init__()
        
        self.max_length = max_length
        
        # Load pre-trained language model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.language_model = AutoModel.from_pretrained(model_name)
            
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, using GPT-2: {e}")
            # Fallback to GPT-2
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.language_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Evidence conditioning layer
        lm_hidden_size = self.language_model.config.hidden_size
        self.evidence_projector = nn.Linear(evidence_dim, lm_hidden_size)
        
        # Context integration
        self.context_attention = nn.MultiheadAttention(
            embed_dim=lm_hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
    
    def forward(
        self,
        evidence: torch.Tensor,
        query_text: Optional[str] = None,
        temperature: float = 0.7,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Generate text response based on evidence.
        
        Args:
            evidence: Evidence representations [B, num_slots, D]
            query_text: Optional input query text
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate
            
        Returns:
            Dictionary with generated text
        """
        batch_size = evidence.size(0)
        
        # Project evidence to language model space
        projected_evidence = self.evidence_projector(evidence)  # [B, slots, lm_dim]
        
        # Create input context
        if query_text:
            # Tokenize query
            query_tokens = self.tokenizer(
                query_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(evidence.device)
            
            # Get query embeddings
            query_embeddings = self.language_model.get_input_embeddings()(
                query_tokens["input_ids"]
            )
            
            # Integrate evidence with query using attention
            enhanced_query, _ = self.context_attention(
                query=query_embeddings,
                key=projected_evidence,
                value=projected_evidence,
            )
            
            # Use enhanced query as starting point
            input_embeddings = enhanced_query
            input_ids = query_tokens["input_ids"]
            
        else:
            # Use evidence directly as starting context
            # Create dummy input tokens
            start_tokens = torch.full(
                (batch_size, 1),
                self.tokenizer.bos_token_id or self.tokenizer.eos_token_id,
                device=evidence.device,
                dtype=torch.long,
            )
            
            input_ids = start_tokens
            input_embeddings = self.language_model.get_input_embeddings()(start_tokens)
            
            # Prepend evidence as context
            input_embeddings = torch.cat([
                projected_evidence.mean(dim=1, keepdim=True),  # [B, 1, D]
                input_embeddings,
            ], dim=1)
            
            # Update input_ids accordingly
            context_tokens = torch.full(
                (batch_size, 1),
                self.tokenizer.unk_token_id or self.tokenizer.eos_token_id,
                device=evidence.device,
                dtype=torch.long,
            )
            input_ids = torch.cat([context_tokens, input_ids], dim=1)
        
        # Generate text
        with torch.no_grad():
            if hasattr(self.language_model, 'generate'):
                # Use generate method if available
                outputs = self.language_model.generate(
                    input_ids=input_ids,
                    max_length=self.max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode generated sequences
                generated_texts = []
                for i in range(num_return_sequences):
                    start_idx = i * batch_size
                    end_idx = (i + 1) * batch_size
                    batch_outputs = outputs[start_idx:end_idx]
                    
                    texts = [
                        self.tokenizer.decode(seq, skip_special_tokens=True)
                        for seq in batch_outputs
                    ]
                    generated_texts.extend(texts)
                
            else:
                # Fallback to forward pass with sampling
                generated_texts = self._generate_with_sampling(
                    input_embeddings, input_ids, temperature
                )
        
        if num_return_sequences == 1 and len(generated_texts) == 1:
            return {"generated_text": generated_texts[0]}
        else:
            return {"generated_texts": generated_texts}
    
    def _generate_with_sampling(
        self,
        input_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        temperature: float,
    ) -> List[str]:
        """Fallback generation with sampling."""
        # Simple autoregressive generation
        current_ids = input_ids
        
        for _ in range(50):  # Generate up to 50 tokens
            # Forward pass
            if hasattr(self.language_model, 'transformer'):
                # GPT-style model
                outputs = self.language_model.transformer(
                    inputs_embeds=input_embeddings
                )
                logits = self.language_model.lm_head(outputs.last_hidden_state)
            else:
                # Other model types
                outputs = self.language_model(inputs_embeds=input_embeddings)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Sample next token
            next_token_logits = logits[:, -1, :] / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Update embeddings for next iteration
            next_embedding = self.language_model.get_input_embeddings()(next_token)
            input_embeddings = torch.cat([input_embeddings, next_embedding], dim=1)
        
        # Decode
        generated_texts = [
            self.tokenizer.decode(seq, skip_special_tokens=True)
            for seq in current_ids
        ]
        
        return generated_texts


class AnswerComposer(nn.Module):
    """
    Main Answer Composer module for generating responses.
    
    This module integrates SNN activations with prototype information
    to generate appropriate responses for both structured and natural
    language tasks.
    """
    
    def __init__(
        self,
        config: AnswerComposerConfig,
        prototype_dim: int,
    ):
        super().__init__()
        
        self.config = config
        self.prototype_dim = prototype_dim
        
        # Evidence extraction
        self.evidence_extractor = EvidenceExtractor(
            prototype_dim=prototype_dim,
            hidden_dim=256,
            num_evidence_slots=config.evidence_slate_size,
        )
        
        # Output heads for different task types
        self.structured_head = StructuredOutputHead(
            input_dim=256,  # From evidence encoder
            num_classes=100,  # Will be adjusted based on task
            hidden_dim=256,
        )
        
        # Text generation
        self.text_generator = TextGenerator(
            model_name=config.llm_model_name,
            evidence_dim=256,
            max_length=config.max_response_length,
        )
        
        # Abstention mechanism
        self.abstention_threshold = nn.Parameter(torch.tensor(0.5))
        
        logger.info(f"Initialized AnswerComposer with {config.llm_model_name}")
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        active_basin: torch.Tensor,
        prototype_manager: "PrototypeManager",
        task_type: str = "text_generation",
        query_text: Optional[str] = None,
        num_classes: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, str, List[str]]]:
        """
        Generate answer based on query and active basin.
        
        Args:
            query_embedding: Query representation [B, D]
            active_basin: Active prototype basin [B, num_prototypes]
            prototype_manager: Prototype manager instance
            task_type: Type of task ("classification", "text_generation", etc.)
            query_text: Optional query text for generation
            num_classes: Number of classes for classification
            
        Returns:
            Dictionary with task-specific outputs
        """
        batch_size = query_embedding.size(0)
        
        # Get top-k activated prototypes
        top_k_indices = torch.topk(
            active_basin,
            k=min(self.config.top_k_prototypes, active_basin.size(-1)),
            dim=-1,
        ).indices  # [B, k]
        
        # Extract prototype embeddings and metadata
        prototype_embeddings = []
        prototype_metadata = []
        activation_weights = []
        
        for b in range(batch_size):
            batch_embeddings = []
            batch_metadata = []
            batch_weights = []
            
            for k in range(top_k_indices.size(-1)):
                proto_id = top_k_indices[b, k].item()
                prototype = prototype_manager.get_prototype(proto_id)
                
                if prototype is not None:
                    batch_embeddings.append(torch.from_numpy(prototype.centroid))
                    batch_metadata.append(prototype.metadata)
                    batch_weights.append(active_basin[b, proto_id].item())
                else:
                    # Handle missing prototype
                    batch_embeddings.append(torch.zeros(self.prototype_dim))
                    batch_metadata.append({})
                    batch_weights.append(0.0)
            
            prototype_embeddings.append(torch.stack(batch_embeddings))
            prototype_metadata.append(batch_metadata)
            activation_weights.append(torch.tensor(batch_weights))
        
        # Stack for batch processing
        prototype_embeddings = torch.stack(prototype_embeddings).to(
            query_embedding.device
        )  # [B, k, D]
        activation_weights = torch.stack(activation_weights).to(
            query_embedding.device
        )  # [B, k]
        
        # Extract evidence
        evidence_outputs = self.evidence_extractor(
            prototype_embeddings=prototype_embeddings,
            activation_weights=activation_weights,
            query_embedding=query_embedding,
        )
        
        evidence = evidence_outputs["evidence"]  # [B, num_slots, hidden_dim]
        
        # Generate outputs based on task type
        outputs = {
            "evidence": evidence,
            "evidence_attention": evidence_outputs["attention_weights"],
            "prototype_metadata": prototype_metadata,
        }
        
        if task_type == "classification":
            if num_classes is not None:
                # Adjust classifier head if needed
                current_classes = self.structured_head.classifier[-1].out_features
                if current_classes != num_classes:
                    self.structured_head.classifier[-1] = nn.Linear(
                        self.structured_head.classifier[-1].in_features,
                        num_classes,
                    ).to(evidence.device)
            
            structured_outputs = self.structured_head(evidence)
            outputs.update(structured_outputs)
            
            # Check for abstention
            max_confidence = structured_outputs["confidence"].max(dim=-1)[0]
            should_abstain = max_confidence < self.abstention_threshold
            outputs["should_abstain"] = should_abstain
            
        elif task_type == "text_generation":
            generation_outputs = self.text_generator(
                evidence=evidence,
                query_text=query_text,
                temperature=self.config.temperature,
            )
            outputs.update(generation_outputs)
            
            # Simple abstention check for text generation
            # (could be improved with more sophisticated methods)
            mean_activation = activation_weights.mean(dim=-1)
            should_abstain = mean_activation < self.config.active_basin_threshold
            outputs["should_abstain"] = should_abstain
            
        elif task_type == "matching":
            # For matching tasks, return similarity scores
            query_expanded = query_embedding.unsqueeze(1)  # [B, 1, D]
            similarities = F.cosine_similarity(
                query_expanded, prototype_embeddings, dim=-1
            )  # [B, k]
            
            outputs.update({
                "similarities": similarities,
                "best_match_idx": similarities.argmax(dim=-1),
                "best_match_score": similarities.max(dim=-1)[0],
            })
            
            # Abstention based on best match score
            should_abstain = outputs["best_match_score"] < 0.5
            outputs["should_abstain"] = should_abstain
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return outputs
    
    def compose_evidence_slate(
        self,
        query_embedding: torch.Tensor,
        active_prototypes: List[Tuple[int, float]],
        prototype_manager: "PrototypeManager",
    ) -> List[Dict]:
        """
        Compose a compact evidence slate for downstream reasoning.
        
        Args:
            query_embedding: Query representation
            active_prototypes: List of (prototype_id, activation_score) tuples
            prototype_manager: Prototype manager instance
            
        Returns:
            List of evidence items with metadata
        """
        evidence_slate = []
        
        for proto_id, activation_score in active_prototypes[:self.config.evidence_slate_size]:
            prototype = prototype_manager.get_prototype(proto_id)
            
            if prototype is not None:
                # Compute relevance to query
                relevance = F.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    torch.from_numpy(prototype.centroid).unsqueeze(0).to(query_embedding.device),
                    dim=-1
                ).item()
                
                evidence_item = {
                    "prototype_id": proto_id,
                    "activation_score": activation_score,
                    "relevance_score": relevance,
                    "confidence": min(activation_score, relevance),
                    "centroid": prototype.centroid.tolist(),
                    "count": prototype.count,
                    "metadata": prototype.metadata,
                    "creation_step": prototype.creation_step,
                    "last_update": prototype.last_update_step,
                }
                
                evidence_slate.append(evidence_item)
        
        # Sort by confidence score
        evidence_slate.sort(key=lambda x: x["confidence"], reverse=True)
        
        return evidence_slate
    
    def update_abstention_threshold(
        self,
        validation_accuracy: float,
        target_coverage: float = 0.9,
    ) -> None:
        """
        Update abstention threshold based on validation performance.
        
        Args:
            validation_accuracy: Current validation accuracy
            target_coverage: Target coverage (fraction of examples to answer)
        """
        # Simple adaptive threshold update
        if validation_accuracy > 0.85 and target_coverage > 0.9:
            # Lower threshold (answer more questions)
            self.abstention_threshold.data *= 0.95
        elif validation_accuracy < 0.7:
            # Raise threshold (be more selective)
            self.abstention_threshold.data *= 1.05
        
        # Clamp to reasonable range
        self.abstention_threshold.data.clamp_(0.1, 0.9)
        
        logger.debug(f"Updated abstention threshold to {self.abstention_threshold.item():.3f}")
    
    def get_generation_statistics(self) -> Dict[str, float]:
        """Get statistics about text generation."""
        return {
            "model_name": self.config.llm_model_name,
            "max_length": self.config.max_response_length,
            "temperature": self.config.temperature,
            "abstention_threshold": self.abstention_threshold.item(),
            "evidence_slots": self.config.evidence_slate_size,
        }
