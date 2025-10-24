"""
Prototype Manager (PM) Module for Neuromorphic Continual Learning.

This module implements dynamic prototype clustering and maintenance for
concept-level memory organization. It maintains an evolving set of concept
prototypes using efficient similarity search and adaptive clustering.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import faiss
import hnswlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans

from ..configs.schema import PrototypeManagerConfig
from ..utils.metrics import cosine_similarity, euclidean_distance

logger = logging.getLogger(__name__)


class Prototype:
    """
    Represents a single concept prototype with statistics.
    
    Attributes:
        id: Unique prototype identifier
        centroid: Prototype center in embedding space
        covariance: Covariance matrix for the prototype cluster
        count: Number of samples assigned to this prototype
        creation_step: Training step when prototype was created
        last_update_step: Last training step when prototype was updated
        metadata: Optional metadata dictionary
    """
    
    def __init__(
        self,
        prototype_id: int,
        centroid: np.ndarray,
        covariance: Optional[np.ndarray] = None,
        count: int = 1,
        creation_step: int = 0,
        metadata: Optional[Dict] = None,
    ):
        self.id = prototype_id
        self.centroid = centroid.copy()
        self.covariance = covariance if covariance is not None else np.eye(len(centroid))
        self.count = count
        self.creation_step = creation_step
        self.last_update_step = creation_step
        self.metadata = metadata or {}
        
        # Statistics
        self.sum_embeddings = centroid * count
        self.sum_squared_embeddings = np.outer(centroid, centroid) * count
    
    def update(
        self, 
        embedding: np.ndarray, 
        ema_alpha: float = 0.99,
        step: int = 0,
    ) -> None:
        """Update prototype with new embedding using EMA."""
        self.count += 1
        self.last_update_step = step
        
        # EMA update of centroid
        self.centroid = ema_alpha * self.centroid + (1 - ema_alpha) * embedding
        
        # Update statistics for covariance computation
        self.sum_embeddings += embedding
        self.sum_squared_embeddings += np.outer(embedding, embedding)
        
        # Update covariance (simplified - could use incremental formula)
        if self.count > 1:
            mean = self.sum_embeddings / self.count
            self.covariance = (
                self.sum_squared_embeddings / self.count - np.outer(mean, mean)
            )
    
    def similarity(self, embedding: np.ndarray, metric: str = "cosine") -> float:
        """Compute similarity between embedding and prototype centroid."""
        if metric == "cosine":
            return cosine_similarity(embedding, self.centroid)
        elif metric == "euclidean":
            return -euclidean_distance(embedding, self.centroid)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def mahalanobis_distance(self, embedding: np.ndarray) -> float:
        """Compute Mahalanobis distance considering prototype covariance."""
        diff = embedding - self.centroid
        try:
            inv_cov = np.linalg.pinv(self.covariance)
            return np.sqrt(diff.T @ inv_cov @ diff)
        except np.linalg.LinAlgError:
            # Fallback to Euclidean if covariance is singular
            return euclidean_distance(embedding, self.centroid)
    
    def get_variance(self) -> float:
        """Get trace of covariance matrix as a measure of cluster spread."""
        return np.trace(self.covariance)
    
    def to_dict(self) -> Dict:
        """Convert prototype to dictionary for serialization."""
        return {
            "id": self.id,
            "centroid": self.centroid.tolist(),
            "covariance": self.covariance.tolist(),
            "count": self.count,
            "creation_step": self.creation_step,
            "last_update_step": self.last_update_step,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Prototype":
        """Create prototype from dictionary."""
        return cls(
            prototype_id=data["id"],
            centroid=np.array(data["centroid"]),
            covariance=np.array(data["covariance"]),
            count=data["count"],
            creation_step=data["creation_step"],
            metadata=data.get("metadata", {}),
        )


class FAISSIndex:
    """FAISS-based similarity search index."""
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "IndexFlatIP",
        nprobe: int = 32,
        device: str = "cpu",
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.nprobe = nprobe
        self.device = device
        
        # Create FAISS index
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")
        
        # Move to GPU if available and requested
        if device == "cuda" and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.is_trained = False
    
    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the index."""
        if not self.is_trained and hasattr(self.index, 'train'):
            self.index.train(embeddings)
            self.is_trained = True
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(embeddings.astype(np.float32))
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        query = query / np.linalg.norm(query, keepdims=True)
        scores, indices = self.index.search(query.astype(np.float32), k)
        return scores, indices
    
    def remove(self, indices: np.ndarray) -> None:
        """Remove vectors at given indices."""
        if hasattr(self.index, 'remove_ids'):
            self.index.remove_ids(indices)
        else:
            # FAISS doesn't support removal for all index types
            logger.warning("Index type doesn't support removal")


class HNSWIndex:
    """HNSWLIB-based similarity search index."""
    
    def __init__(
        self,
        dimension: int,
        max_elements: int = 100000,
        ef_construction: int = 200,
        m: int = 16,
        ef: int = 50,
    ):
        self.dimension = dimension
        self.max_elements = max_elements
        
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=m
        )
        self.index.set_ef(ef)
        
        self.current_count = 0
    
    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the index."""
        num_embeddings = embeddings.shape[0]
        if self.current_count + num_embeddings > self.max_elements:
            # Resize index if needed
            self.index.resize_index(self.current_count + num_embeddings)
        
        indices = np.arange(self.current_count, self.current_count + num_embeddings)
        self.index.add_items(embeddings, indices)
        self.current_count += num_embeddings
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        indices, distances = self.index.knn_query(query, k=k)
        # Convert distances to similarities for cosine space
        similarities = 1 - distances
        return similarities, indices


class PrototypeManager(nn.Module):
    """
    Dynamic prototype clustering and management system.
    
    This module maintains an evolving set of concept prototypes, handling
    assignment, updates, merging, and splitting operations. It provides
    efficient similarity search and supports both FAISS and HNSWLIB backends.
    """
    
    def __init__(self, config: PrototypeManagerConfig, embedding_dim: int):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        
        # Prototype storage
        self.prototypes: Dict[int, Prototype] = {}
        self.next_prototype_id = 0
        self.current_step = 0
        
        # Similarity search index
        self.index = self._create_index()
        
        # Statistics
        self.assignment_counts = {}
        self.merge_history = []
        self.split_history = []
        
        logger.info(
            f"Initialized PrototypeManager with {config.indexing_backend} backend"
        )
    
    def _create_index(self) -> Union[FAISSIndex, HNSWIndex]:
        """Create similarity search index based on configuration."""
        if self.config.indexing_backend == "faiss":
            return FAISSIndex(
                dimension=self.embedding_dim,
                index_type=self.config.faiss_index_type,
                nprobe=self.config.faiss_nprobe,
            )
        elif self.config.indexing_backend == "hnswlib":
            return HNSWIndex(
                dimension=self.embedding_dim,
                max_elements=self.config.max_prototypes,
                ef_construction=self.config.hnswlib_ef_construction,
                m=self.config.hnswlib_m,
                ef=self.config.hnswlib_ef,
            )
        else:
            raise ValueError(f"Unknown indexing backend: {self.config.indexing_backend}")
    
    def assign(
        self, 
        embedding: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Assign embedding to nearest prototype or create new one.
        
        Args:
            embedding: Input embedding vector
            metadata: Optional metadata to store with prototype
            
        Returns:
            Prototype ID that the embedding was assigned to
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        if len(self.prototypes) == 0:
            # Create first prototype
            return self._create_prototype(embedding, metadata)
        
        # Find nearest prototype
        similarities, indices = self.index.search(embedding.reshape(1, -1), k=1)
        best_similarity = similarities[0, 0]
        best_prototype_id = indices[0, 0]
        
        if best_similarity >= self.config.similarity_threshold:
            # Assign to existing prototype
            self._update_prototype(best_prototype_id, embedding)
            return best_prototype_id
        else:
            # Create new prototype
            return self._create_prototype(embedding, metadata)
    
    def _create_prototype(
        self, 
        embedding: np.ndarray, 
        metadata: Optional[Dict] = None,
    ) -> int:
        """Create a new prototype."""
        if len(self.prototypes) >= self.config.max_prototypes:
            logger.warning(
                f"Maximum number of prototypes ({self.config.max_prototypes}) reached"
            )
            # Find least used prototype and replace it
            prototype_id = min(
                self.prototypes.keys(),
                key=lambda k: self.prototypes[k].count
            )
            self._remove_prototype(prototype_id)
        
        prototype_id = self.next_prototype_id
        self.next_prototype_id += 1
        
        prototype = Prototype(
            prototype_id=prototype_id,
            centroid=embedding,
            creation_step=self.current_step,
            metadata=metadata,
        )
        
        self.prototypes[prototype_id] = prototype
        self.assignment_counts[prototype_id] = 1
        
        # Add to index
        self.index.add(embedding.reshape(1, -1))
        
        logger.debug(f"Created prototype {prototype_id}")
        return prototype_id
    
    def _update_prototype(self, prototype_id: int, embedding: np.ndarray) -> None:
        """Update existing prototype with new embedding."""
        if prototype_id not in self.prototypes:
            raise ValueError(f"Prototype {prototype_id} not found")
        
        prototype = self.prototypes[prototype_id]
        prototype.update(
            embedding=embedding,
            ema_alpha=self.config.ema_alpha,
            step=self.current_step,
        )
        self.assignment_counts[prototype_id] += 1
        
        logger.debug(f"Updated prototype {prototype_id}")
    
    def _remove_prototype(self, prototype_id: int) -> None:
        """Remove a prototype from the system."""
        if prototype_id in self.prototypes:
            del self.prototypes[prototype_id]
            if prototype_id in self.assignment_counts:
                del self.assignment_counts[prototype_id]
            logger.debug(f"Removed prototype {prototype_id}")
    
    def get_top_k_prototypes(
        self, 
        query_embedding: torch.Tensor, 
        k: int,
    ) -> List[Tuple[int, float]]:
        """
        Get top-k most similar prototypes to query embedding.
        
        Args:
            query_embedding: Query vector
            k: Number of prototypes to return
            
        Returns:
            List of (prototype_id, similarity_score) tuples
        """
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()
        
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if len(self.prototypes) == 0:
            return []
        
        k = min(k, len(self.prototypes))
        similarities, indices = self.index.search(query_embedding.reshape(1, -1), k=k)
        
        results = []
        for i in range(k):
            prototype_id = indices[0, i]
            similarity = similarities[0, i]
            results.append((prototype_id, similarity))
        
        return results
    
    def get_neighbors(self, prototype_id: int, max_neighbors: int = 10) -> List[int]:
        """Get neighboring prototypes based on centroid similarity."""
        if prototype_id not in self.prototypes:
            return []
        
        centroid = self.prototypes[prototype_id].centroid
        top_k = self.get_top_k_prototypes(centroid, max_neighbors + 1)
        
        # Remove self and return neighbors
        neighbors = [pid for pid, _ in top_k if pid != prototype_id]
        return neighbors[:max_neighbors]
    
    def maintain(self) -> Dict[str, int]:
        """
        Perform maintenance operations (merging and splitting).
        
        Returns:
            Dictionary with maintenance statistics
        """
        self.current_step += 1
        
        stats = {
            "merges": 0,
            "splits": 0,
            "total_prototypes": len(self.prototypes),
        }
        
        if self.current_step % self.config.merge_split_interval == 0:
            stats["merges"] = self._merge_similar_prototypes()
            stats["splits"] = self._split_large_prototypes()
            
            # Rebuild index after major changes
            self._rebuild_index()
        
        return stats
    
    def _merge_similar_prototypes(self) -> int:
        """Merge prototypes that are very similar."""
        merge_count = 0
        prototype_ids = list(self.prototypes.keys())
        
        for i, id1 in enumerate(prototype_ids):
            if id1 not in self.prototypes:
                continue
                
            for id2 in prototype_ids[i+1:]:
                if id2 not in self.prototypes:
                    continue
                
                prototype1 = self.prototypes[id1]
                prototype2 = self.prototypes[id2]
                
                similarity = cosine_similarity(
                    prototype1.centroid, 
                    prototype2.centroid
                )
                
                if similarity >= self.config.merge_threshold:
                    # Merge prototype2 into prototype1
                    self._merge_prototypes(id1, id2)
                    merge_count += 1
                    break
        
        return merge_count
    
    def _merge_prototypes(self, target_id: int, source_id: int) -> None:
        """Merge source prototype into target prototype."""
        target = self.prototypes[target_id]
        source = self.prototypes[source_id]
        
        # Weighted merge based on counts
        total_count = target.count + source.count
        w1 = target.count / total_count
        w2 = source.count / total_count
        
        # Update target prototype
        target.centroid = w1 * target.centroid + w2 * source.centroid
        target.count = total_count
        target.sum_embeddings += source.sum_embeddings
        target.sum_squared_embeddings += source.sum_squared_embeddings
        
        # Update covariance
        mean = target.sum_embeddings / target.count
        target.covariance = (
            target.sum_squared_embeddings / target.count - np.outer(mean, mean)
        )
        
        # Transfer assignment counts
        self.assignment_counts[target_id] += self.assignment_counts.get(source_id, 0)
        
        # Remove source prototype
        self._remove_prototype(source_id)
        
        # Record merge
        self.merge_history.append({
            "step": self.current_step,
            "target_id": target_id,
            "source_id": source_id,
        })
        
        logger.debug(f"Merged prototype {source_id} into {target_id}")
    
    def _split_large_prototypes(self) -> int:
        """Split prototypes with high variance."""
        split_count = 0
        
        for prototype_id, prototype in list(self.prototypes.items()):
            variance = prototype.get_variance()
            
            if (variance > self.config.split_variance_threshold and 
                prototype.count > 10):  # Only split if sufficient samples
                
                if self._split_prototype(prototype_id):
                    split_count += 1
        
        return split_count
    
    def _split_prototype(self, prototype_id: int) -> bool:
        """Split a prototype into two using k-means."""
        if len(self.prototypes) >= self.config.max_prototypes - 1:
            return False
        
        prototype = self.prototypes[prototype_id]
        
        # Generate samples from prototype distribution
        try:
            samples = np.random.multivariate_normal(
                mean=prototype.centroid,
                cov=prototype.covariance,
                size=max(10, prototype.count // 2)
            )
        except np.linalg.LinAlgError:
            # Fallback if covariance is singular
            return False
        
        # Split using k-means
        kmeans = MiniBatchKMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(samples)
        
        # Create two new prototypes
        for label in [0, 1]:
            mask = labels == label
            if mask.sum() > 0:
                centroid = samples[mask].mean(axis=0)
                new_metadata = prototype.metadata.copy()
                new_metadata["parent_id"] = prototype_id
                
                self._create_prototype(centroid, new_metadata)
        
        # Remove original prototype
        self._remove_prototype(prototype_id)
        
        # Record split
        self.split_history.append({
            "step": self.current_step,
            "prototype_id": prototype_id,
        })
        
        logger.debug(f"Split prototype {prototype_id}")
        return True
    
    def _rebuild_index(self) -> None:
        """Rebuild the similarity search index."""
        if len(self.prototypes) == 0:
            return
        
        # Create new index
        self.index = self._create_index()
        
        # Add all prototypes
        centroids = np.stack([p.centroid for p in self.prototypes.values()])
        self.index.add(centroids)
        
        logger.debug("Rebuilt similarity index")
    
    def get_prototype(self, prototype_id: int) -> Optional[Prototype]:
        """Get prototype by ID."""
        return self.prototypes.get(prototype_id)
    
    def get_all_prototypes(self) -> List[Prototype]:
        """Get all prototypes."""
        return list(self.prototypes.values())
    
    def get_statistics(self) -> Dict:
        """Get system statistics."""
        if not self.prototypes:
            return {"total_prototypes": 0}
        
        counts = [p.count for p in self.prototypes.values()]
        variances = [p.get_variance() for p in self.prototypes.values()]
        
        return {
            "total_prototypes": len(self.prototypes),
            "total_assignments": sum(counts),
            "avg_prototype_size": np.mean(counts),
            "min_prototype_size": np.min(counts),
            "max_prototype_size": np.max(counts),
            "avg_variance": np.mean(variances),
            "total_merges": len(self.merge_history),
            "total_splits": len(self.split_history),
            "next_prototype_id": self.next_prototype_id,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save prototype manager state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config.dict(),
            "embedding_dim": self.embedding_dim,
            "prototypes": {k: v.to_dict() for k, v in self.prototypes.items()},
            "next_prototype_id": self.next_prototype_id,
            "current_step": self.current_step,
            "assignment_counts": self.assignment_counts,
            "merge_history": self.merge_history,
            "split_history": self.split_history,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved PrototypeManager state to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """Load prototype manager state."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        # Restore prototypes
        self.prototypes = {
            k: Prototype.from_dict(v) for k, v in state["prototypes"].items()
        }
        self.next_prototype_id = state["next_prototype_id"]
        self.current_step = state["current_step"]
        self.assignment_counts = state["assignment_counts"]
        self.merge_history = state["merge_history"]
        self.split_history = state["split_history"]
        
        # Rebuild index
        self._rebuild_index()
        
        logger.info(f"Loaded PrototypeManager state from {path}")
    
    def forward(
        self, 
        embeddings: torch.Tensor,
        metadata: Optional[List[Dict]] = None,
    ) -> torch.Tensor:
        """
        Forward pass: assign embeddings to prototypes.
        
        Args:
            embeddings: Batch of embeddings [B, D]
            metadata: Optional metadata for each embedding
            
        Returns:
            Prototype IDs for each embedding [B]
        """
        batch_size = embeddings.size(0)
        prototype_ids = torch.zeros(batch_size, dtype=torch.long)
        
        for i in range(batch_size):
            embedding = embeddings[i]
            meta = metadata[i] if metadata else None
            prototype_id = self.assign(embedding, meta)
            prototype_ids[i] = prototype_id
        
        return prototype_ids
