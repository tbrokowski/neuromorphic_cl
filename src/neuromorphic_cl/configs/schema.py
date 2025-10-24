"""
Configuration schemas for the Neuromorphic Continual Learning System.

This module defines Pydantic models for all system configurations,
ensuring type safety and validation across the entire pipeline.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class EncoderType(str, Enum):
    """Supported encoder types for the Concept Encoder."""
    
    DEEPSEEK_OCR = "deepseek_ocr"
    VIT = "vit"
    DONUT = "donut"
    LAYOUTLMV3 = "layoutlmv3"


class LossType(str, Enum):
    """Supported loss types for training."""
    
    INFONCE = "infonce"
    SUPERVISED_CONTRASTIVE = "supervised_contrastive"
    PROTOTYPE_ALIGNMENT = "prototype_alignment"
    CLIP_SUPCON = "clip_supcon"
    PULL_PUSH = "pull_push"


class SNNNeuronType(str, Enum):
    """Supported SNN neuron types."""
    
    LIF = "lif"
    PLIF = "plif"
    ALIF = "alif"


class DistributedBackend(str, Enum):
    """Supported distributed training backends."""
    
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"


class ConceptEncoderConfig(BaseModel):
    """Configuration for the Concept Encoder module."""
    
    encoder_type: EncoderType = EncoderType.DEEPSEEK_OCR
    embedding_dim: int = Field(512, ge=128, le=2048)
    projection_dim: int = Field(256, ge=64, le=1024)
    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int = Field(2, ge=0, le=12)
    max_tokens_per_page: int = Field(1024, ge=256, le=4096)
    contrastive_temperature: float = Field(0.07, gt=0.0, le=1.0)
    dropout_rate: float = Field(0.1, ge=0.0, lt=1.0)
    
    # DeepSeek-OCR specific
    deepseek_model_path: Optional[str] = None
    target_resolution: tuple[int, int] = (224, 224)
    
    # Fallback encoder configs
    vit_model_name: str = "google/vit-base-patch16-224"
    donut_model_name: str = "naver-clova-ix/donut-base"
    layoutlm_model_name: str = "microsoft/layoutlmv3-base"


class PrototypeManagerConfig(BaseModel):
    """Configuration for the Prototype Manager module."""
    
    similarity_threshold: float = Field(0.85, gt=0.0, le=1.0)
    ema_alpha: float = Field(0.99, gt=0.0, lt=1.0)
    max_prototypes: int = Field(10000, ge=100)
    merge_threshold: float = Field(0.95, gt=0.0, le=1.0)
    split_variance_threshold: float = Field(2.0, gt=0.0)
    indexing_backend: str = Field("faiss", pattern="^(faiss|hnswlib)$")
    
    # FAISS specific
    faiss_index_type: str = "IndexFlatIP"
    faiss_nprobe: int = 32
    
    # HNSWLIB specific
    hnswlib_ef_construction: int = 200
    hnswlib_m: int = 16
    hnswlib_ef: int = 50
    
    # Maintenance
    maintenance_interval: int = Field(1000, ge=100)
    merge_split_interval: int = Field(5000, ge=1000)


class SNNConfig(BaseModel):
    """Configuration for the Spiking Neural Network module."""
    
    neuron_type: SNNNeuronType = SNNNeuronType.LIF
    num_timesteps: int = Field(20, ge=5, le=100)
    spike_threshold: float = Field(1.0, gt=0.0)
    reset_potential: float = Field(0.0, ge=-10.0, le=10.0)
    membrane_potential_decay: float = Field(0.9, gt=0.0, lt=1.0)
    
    # STDP parameters
    stdp_lr: float = Field(0.01, gt=0.0, le=1.0)
    stdp_tau_pre: float = Field(20.0, gt=0.0)
    stdp_tau_post: float = Field(20.0, gt=0.0)
    reward_modulation: bool = True
    
    # Network topology
    lateral_connectivity: float = Field(0.1, ge=0.0, le=1.0)
    recurrent_strength: float = Field(0.5, ge=0.0, le=2.0)
    
    # Consolidation
    consolidation_interval: int = Field(2000, ge=500)
    replay_duration: int = Field(100, ge=10)
    replay_strength: float = Field(0.8, ge=0.0, le=2.0)


class AnswerComposerConfig(BaseModel):
    """Configuration for the Answer Composer module."""
    
    top_k_prototypes: int = Field(10, ge=1, le=100)
    active_basin_threshold: float = Field(0.1, gt=0.0, le=1.0)
    evidence_slate_size: int = Field(5, ge=1, le=20)
    llm_model_name: str = "microsoft/DialoGPT-medium"
    max_response_length: int = Field(512, ge=64, le=2048)
    temperature: float = Field(0.7, gt=0.0, le=2.0)


class DataConfig(BaseModel):
    """Configuration for data loading and processing."""
    
    # Dataset paths
    pubmed_path: Optional[Path] = None
    mimic_cxr_path: Optional[Path] = None
    vqa_rad_path: Optional[Path] = None
    bioasq_path: Optional[Path] = None
    
    # Data loading
    batch_size: int = Field(32, ge=1, le=512)
    num_workers: int = Field(4, ge=0, le=16)
    pin_memory: bool = True
    prefetch_factor: int = Field(2, ge=1, le=8)
    
    # Preprocessing
    image_size: tuple[int, int] = (224, 224)
    text_max_length: int = Field(512, ge=64, le=2048)
    augmentation_prob: float = Field(0.3, ge=0.0, le=1.0)
    
    # Streaming
    streaming_buffer_size: int = Field(1000, ge=100)
    shuffle_buffer_size: int = Field(10000, ge=1000)


class TrainingConfig(BaseModel):
    """Configuration for training procedures."""
    
    # Optimization
    learning_rate: float = Field(1e-4, gt=0.0, le=1.0)
    weight_decay: float = Field(1e-5, ge=0.0, le=1.0)
    optimizer: str = Field("adamw", pattern="^(adam|adamw|sgd|rmsprop)$")
    scheduler: str = Field("cosine", pattern="^(cosine|linear|step|plateau)$")
    
    # Training dynamics
    max_epochs: int = Field(100, ge=1)
    max_steps: Optional[int] = None
    gradient_clip_norm: float = Field(1.0, gt=0.0)
    accumulate_grad_batches: int = Field(1, ge=1)
    
    # Loss weights
    loss_weights: Dict[str, float] = {
        "infonce": 1.0,
        "supervised_contrastive": 0.5,
        "prototype_alignment": 0.3,
        "clip_supcon": 0.7,
        "pull_push": 0.4,
    }
    
    # Validation
    val_check_interval: Union[int, float] = 1.0
    num_sanity_val_steps: int = 2
    
    # Checkpointing
    save_top_k: int = 3
    monitor_metric: str = "val/accuracy"
    mode: str = Field("max", pattern="^(min|max)$")


class DistributedConfig(BaseModel):
    """Configuration for distributed training."""
    
    backend: DistributedBackend = DistributedBackend.NCCL
    num_nodes: int = Field(1, ge=1)
    num_gpus_per_node: int = Field(1, ge=1)
    
    # DDP settings
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    
    # Communication
    timeout_seconds: int = Field(1800, ge=300)  # 30 minutes
    
    # Sharding strategies
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_backward_prefetch: str = "BACKWARD_PRE"
    
    # Precision
    precision: str = Field("16-mixed", pattern="^(16|32|64|16-mixed|bf16-mixed)$")
    
    @validator("num_gpus_per_node")
    def validate_gpu_count(cls, v: int) -> int:
        """Validate GPU count is reasonable."""
        if v > 8:
            raise ValueError("num_gpus_per_node should not exceed 8 for most systems")
        return v


class EvaluationConfig(BaseModel):
    """Configuration for evaluation procedures."""
    
    # Metrics
    compute_forgetting: bool = True
    compute_transfer: bool = True
    compute_memory_efficiency: bool = True
    compute_energy_efficiency: bool = True
    
    # Baselines
    run_baselines: bool = True
    baseline_methods: List[str] = [
        "sequential_finetune",
        "experience_replay", 
        "ewc",
        "rag",
        "prototype_only",
        "snn_only"
    ]
    
    # Test settings
    test_batch_size: int = Field(64, ge=1)
    num_test_tasks: int = Field(5, ge=1, le=20)
    shots_per_task: int = Field(100, ge=10)


class SystemConfig(BaseModel):
    """Main system configuration combining all components."""
    
    # Component configs
    concept_encoder: ConceptEncoderConfig = ConceptEncoderConfig()
    prototype_manager: PrototypeManagerConfig = PrototypeManagerConfig()
    snn: SNNConfig = SNNConfig()
    answer_composer: AnswerComposerConfig = AnswerComposerConfig()
    
    # Pipeline configs  
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    distributed: DistributedConfig = DistributedConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    
    # Experiment metadata
    experiment_name: str = "neuromorphic_cl_experiment"
    project_name: str = "neuromorphic-continual-learning"
    tags: List[str] = []
    notes: str = ""
    
    # Logging and monitoring
    log_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")
    output_dir: Path = Path("outputs")
    
    # Reproducibility
    seed: int = Field(42, ge=0)
    deterministic: bool = True
    benchmark: bool = True
    
    # Resource management
    max_memory_gb: Optional[float] = None
    cpu_limit: Optional[int] = None
    
    class Config:
        """Pydantic config."""
        
        extra = "forbid"
        validate_assignment = True
        use_enum_values = True
