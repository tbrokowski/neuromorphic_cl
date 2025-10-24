"""
Spiking Neural Network (SNN) Module for Neuromorphic Continual Learning.

This module implements attractor-based associative memory using biologically
plausible spiking neural networks with STDP learning rules. Each prototype
corresponds to a neuron population, with recurrent and lateral connections
forming dynamic memory attractors.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import (
    base,
    functional,
    layer,
    neuron,
    surrogate,
)

from ..configs.schema import SNNConfig
from ..utils.metrics import spike_count, spike_rate

logger = logging.getLogger(__name__)


class LIFNeuron(neuron.LIFNode):
    """Leaky Integrate-and-Fire neuron with custom parameters."""
    
    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan(),
        detach_reset: bool = False,
        step_mode: str = 's',
        backend: str = 'torch',
        store_v_seq: bool = False,
    ):
        super().__init__(
            tau=tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
        )


class PLIFNeuron(neuron.ParametricLIFNode):
    """Parametric LIF neuron with learnable parameters."""
    
    def __init__(
        self,
        init_tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan(),
        detach_reset: bool = False,
        step_mode: str = 's',
        backend: str = 'torch',
        store_v_seq: bool = False,
    ):
        super().__init__(
            init_tau=init_tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
        )


class ALIFNeuron(neuron.LIFNode):
    """Adaptive LIF neuron with adaptive threshold."""
    
    def __init__(
        self,
        tau: float = 2.0,
        tau_adapt: float = 10.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        adapt_strength: float = 0.1,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan(),
        detach_reset: bool = False,
        step_mode: str = 's',
        backend: str = 'torch',
        store_v_seq: bool = False,
    ):
        super().__init__(
            tau=tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
        )
        
        self.tau_adapt = tau_adapt
        self.adapt_strength = adapt_strength
        self.register_memory('adapt_var', 0.0)
    
    def neuronal_charge(self, x: torch.Tensor) -> None:
        """Override to include adaptive threshold."""
        super().neuronal_charge(x)
        
        # Update adaptive variable
        if hasattr(self, 'adapt_var'):
            self.adapt_var = (
                self.adapt_var * (1 - 1/self.tau_adapt) + 
                self.spike * self.adapt_strength
            )
            # Adjust threshold
            self.v_threshold_current = self.v_threshold + self.adapt_var


class STDPSynapse(nn.Module):
    """
    Spike-Timing Dependent Plasticity (STDP) synapse.
    
    Implements asymmetric STDP with exponential windows and optional
    reward modulation for reinforcement learning.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        learning_rate: float = 0.01,
        a_plus: float = 1.0,
        a_minus: float = 1.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        reward_modulated: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.learning_rate = learning_rate
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_min = w_min
        self.w_max = w_max
        self.reward_modulated = reward_modulated
        
        # Synaptic weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1
        )
        
        # STDP traces
        self.register_buffer('pre_trace', torch.zeros(1, in_features))
        self.register_buffer('post_trace', torch.zeros(1, out_features))
        
        # Eligibility trace for reward modulation
        if reward_modulated:
            self.register_buffer(
                'eligibility_trace', 
                torch.zeros(out_features, in_features)
            )
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize synaptic weights."""
        nn.init.uniform_(self.weight, self.w_min, self.w_max)
    
    def forward(
        self, 
        pre_spikes: torch.Tensor, 
        post_spikes: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with STDP learning.
        
        Args:
            pre_spikes: Presynaptic spikes [B, in_features]
            post_spikes: Postsynaptic spikes [B, out_features]
            reward: Optional reward signal [B] or scalar
            
        Returns:
            Synaptic output current [B, out_features]
        """
        batch_size = pre_spikes.size(0)
        
        # Synaptic transmission
        current = F.linear(pre_spikes, self.weight)
        
        # STDP learning (only during training)
        if self.training:
            self._stdp_update(pre_spikes, post_spikes, reward)
        
        return current
    
    def _stdp_update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
    ) -> None:
        """Update synaptic weights using STDP."""
        # Update traces
        self.pre_trace *= torch.exp(-1.0 / self.tau_pre)
        self.post_trace *= torch.exp(-1.0 / self.tau_post)
        
        # Add current spikes to traces
        self.pre_trace += pre_spikes.mean(dim=0, keepdim=True)
        self.post_trace += post_spikes.mean(dim=0, keepdim=True)
        
        # Compute weight updates
        # LTP: post_spike * pre_trace
        ltp = torch.outer(post_spikes.mean(dim=0), self.pre_trace.squeeze(0))
        
        # LTD: pre_spike * post_trace
        ltd = torch.outer(self.post_trace.squeeze(0), pre_spikes.mean(dim=0))
        
        # Total weight change
        dw = self.a_plus * ltp - self.a_minus * ltd
        
        if self.reward_modulated and reward is not None:
            # Update eligibility trace
            self.eligibility_trace = (
                0.95 * self.eligibility_trace + dw
            )
            
            # Apply reward modulation
            if isinstance(reward, torch.Tensor):
                reward_factor = reward.mean()
            else:
                reward_factor = reward
            
            dw = reward_factor * self.eligibility_trace
        
        # Apply weight update
        with torch.no_grad():
            self.weight.add_(self.learning_rate * dw)
            
            # Clip weights to valid range
            self.weight.clamp_(self.w_min, self.w_max)
    
    def reset_traces(self) -> None:
        """Reset STDP traces."""
        self.pre_trace.zero_()
        self.post_trace.zero_()
        if hasattr(self, 'eligibility_trace'):
            self.eligibility_trace.zero_()


class NeuromorphicMemory(nn.Module):
    """
    Neuromorphic memory layer with populations of spiking neurons.
    
    Each prototype is represented by a population of neurons with
    recurrent connections for stability and lateral connections
    for associative recall.
    """
    
    def __init__(
        self,
        num_prototypes: int,
        population_size: int = 10,
        config: SNNConfig = SNNConfig(),
    ):
        super().__init__()
        
        self.num_prototypes = num_prototypes
        self.population_size = population_size
        self.config = config
        self.total_neurons = num_prototypes * population_size
        
        # Create neuron populations
        self.neurons = self._create_neurons()
        
        # Recurrent connections within populations
        self.recurrent_weights = nn.Parameter(
            torch.eye(self.total_neurons) * config.recurrent_strength
        )
        
        # Lateral connections between populations
        self.lateral_weights = nn.Parameter(
            torch.randn(self.total_neurons, self.total_neurons) * 0.1
        )
        
        # STDP synapses for learning
        self.stdp_recurrent = STDPSynapse(
            in_features=self.total_neurons,
            out_features=self.total_neurons,
            tau_pre=config.stdp_tau_pre,
            tau_post=config.stdp_tau_post,
            learning_rate=config.stdp_lr,
            reward_modulated=config.reward_modulation,
        )
        
        # Input projection from prototypes
        self.input_projection = nn.Linear(1, population_size, bias=False)
        
        # Membrane potentials and spike history
        self.register_buffer(
            'membrane_potentials', 
            torch.zeros(1, self.total_neurons)
        )
        self.register_buffer(
            'spike_history', 
            torch.zeros(config.num_timesteps, 1, self.total_neurons)
        )
        
        self.timestep = 0
    
    def _create_neurons(self) -> nn.Module:
        """Create neuron populations based on configuration."""
        if self.config.neuron_type.value == "lif":
            return LIFNeuron(
                tau=1.0 / (1.0 - self.config.membrane_potential_decay),
                v_threshold=self.config.spike_threshold,
                v_reset=self.config.reset_potential,
                step_mode='s',
            )
        elif self.config.neuron_type.value == "plif":
            return PLIFNeuron(
                init_tau=1.0 / (1.0 - self.config.membrane_potential_decay),
                v_threshold=self.config.spike_threshold,
                v_reset=self.config.reset_potential,
                step_mode='s',
            )
        elif self.config.neuron_type.value == "alif":
            return ALIFNeuron(
                tau=1.0 / (1.0 - self.config.membrane_potential_decay),
                v_threshold=self.config.spike_threshold,
                v_reset=self.config.reset_potential,
                step_mode='s',
            )
        else:
            raise ValueError(f"Unknown neuron type: {self.config.neuron_type}")
    
    def stimulate_prototypes(
        self, 
        prototype_ids: torch.Tensor,
        stimulation_strength: float = 2.0,
    ) -> torch.Tensor:
        """
        Stimulate specific prototype populations.
        
        Args:
            prototype_ids: IDs of prototypes to stimulate [B]
            stimulation_strength: Strength of external stimulation
            
        Returns:
            Input current for stimulated populations [B, total_neurons]
        """
        batch_size = prototype_ids.size(0)
        input_current = torch.zeros(
            batch_size, self.total_neurons, 
            device=prototype_ids.device
        )
        
        for i, proto_id in enumerate(prototype_ids):
            start_idx = proto_id * self.population_size
            end_idx = start_idx + self.population_size
            input_current[i, start_idx:end_idx] = stimulation_strength
        
        return input_current
    
    def forward(
        self,
        prototype_ids: torch.Tensor,
        num_timesteps: Optional[int] = None,
        return_full_dynamics: bool = False,
        reward: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the neuromorphic memory.
        
        Args:
            prototype_ids: Prototype IDs to stimulate [B]
            num_timesteps: Number of simulation timesteps
            return_full_dynamics: Whether to return full spike trains
            reward: Optional reward signal for STDP
            
        Returns:
            Dictionary containing spike outputs and dynamics
        """
        if num_timesteps is None:
            num_timesteps = self.config.num_timesteps
        
        batch_size = prototype_ids.size(0)
        
        # Initialize outputs
        spike_trains = []
        membrane_traces = []
        
        # Reset neuron states
        functional.reset_net(self.neurons)
        self.membrane_potentials.zero_()
        
        # Simulate dynamics
        for t in range(num_timesteps):
            # External stimulation (stronger in early timesteps)
            stimulation_decay = np.exp(-t / (num_timesteps * 0.3))
            external_current = self.stimulate_prototypes(
                prototype_ids, 
                stimulation_strength=2.0 * stimulation_decay
            )
            
            # Recurrent input from previous spikes
            if t > 0:
                prev_spikes = spike_trains[-1]
                recurrent_current = self.stdp_recurrent(
                    prev_spikes, 
                    prev_spikes,  # Same for recurrent
                    reward=reward
                )
                
                # Add lateral connections
                lateral_current = F.linear(prev_spikes, self.lateral_weights)
                
                total_current = (
                    external_current + 
                    recurrent_current + 
                    0.1 * lateral_current
                )
            else:
                total_current = external_current
            
            # Add noise for biological realism
            noise = torch.randn_like(total_current) * 0.05
            total_current += noise
            
            # Update membrane potentials and generate spikes
            spikes = self.neurons(total_current)
            
            spike_trains.append(spikes)
            membrane_traces.append(self.neurons.v.clone())
        
        # Stack outputs
        spike_output = torch.stack(spike_trains)  # [T, B, N]
        membrane_output = torch.stack(membrane_traces)  # [T, B, N]
        
        # Compute population activities
        population_spikes = self._compute_population_activities(spike_output)
        
        # Find active basin (most active populations)
        active_basin = self._find_active_basin(population_spikes)
        
        outputs = {
            "spike_output": spike_output,
            "population_spikes": population_spikes,
            "active_basin": active_basin,
            "total_spikes": spike_output.sum(dim=0),  # [B, N]
            "spike_rates": spike_output.mean(dim=0),  # [B, N]
        }
        
        if return_full_dynamics:
            outputs.update({
                "membrane_traces": membrane_output,
                "spike_trains": spike_trains,
            })
        
        return outputs
    
    def _compute_population_activities(
        self, 
        spike_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute activity for each prototype population."""
        # spike_output: [T, B, N]
        T, B, N = spike_output.shape
        
        # Reshape to separate populations
        spikes_reshaped = spike_output.view(
            T, B, self.num_prototypes, self.population_size
        )
        
        # Sum over population and time
        population_activities = spikes_reshaped.sum(dim=(0, 3))  # [B, num_prototypes]
        
        return population_activities
    
    def _find_active_basin(
        self, 
        population_spikes: torch.Tensor,
        threshold_percentile: float = 80.0,
    ) -> torch.Tensor:
        """Find the active basin of prototype populations."""
        # population_spikes: [B, num_prototypes]
        
        # Compute threshold for each batch element
        threshold = torch.quantile(
            population_spikes, 
            threshold_percentile / 100.0, 
            dim=1, 
            keepdim=True
        )
        
        # Find populations above threshold
        active_basin = (population_spikes >= threshold).float()
        
        return active_basin
    
    def consolidate(
        self,
        important_prototypes: List[int],
        replay_strength: float = 0.8,
        replay_duration: int = 50,
    ) -> None:
        """
        Consolidate memory by replaying important prototypes.
        
        Args:
            important_prototypes: List of prototype IDs to consolidate
            replay_strength: Strength of replay stimulation
            replay_duration: Number of timesteps for replay
        """
        if not important_prototypes:
            return
        
        self.eval()  # Disable learning temporarily
        
        with torch.no_grad():
            for proto_id in important_prototypes:
                # Create stimulation for this prototype
                proto_tensor = torch.tensor([proto_id], device=next(self.parameters()).device)
                
                # Run short replay
                self.forward(
                    prototype_ids=proto_tensor,
                    num_timesteps=replay_duration,
                )
        
        self.train()  # Re-enable learning
        
        logger.debug(f"Consolidated {len(important_prototypes)} prototypes")
    
    def get_connection_strengths(self) -> Dict[str, torch.Tensor]:
        """Get current synaptic connection strengths."""
        return {
            "recurrent_weights": self.recurrent_weights.data.clone(),
            "lateral_weights": self.lateral_weights.data.clone(),
            "stdp_weights": self.stdp_recurrent.weight.data.clone(),
        }
    
    def add_prototype_population(self) -> None:
        """Add a new prototype population when new prototypes are created."""
        # This would require dynamic network expansion
        # For now, we assume a fixed maximum number of prototypes
        pass
    
    def get_energy_consumption(self, spike_output: torch.Tensor) -> Dict[str, float]:
        """Estimate energy consumption based on spike counts."""
        total_spikes = spike_output.sum().item()
        num_neurons = spike_output.size(-1)
        num_timesteps = spike_output.size(0)
        
        # Simplified energy model
        spike_energy = total_spikes * 1e-12  # pJ per spike
        leakage_energy = num_neurons * num_timesteps * 1e-15  # pJ per neuron per timestep
        
        return {
            "total_energy_pj": spike_energy + leakage_energy,
            "spike_energy_pj": spike_energy,
            "leakage_energy_pj": leakage_energy,
            "spikes_per_neuron": total_spikes / num_neurons,
        }


class SpikingNeuralNetwork(nn.Module):
    """
    Main Spiking Neural Network module for neuromorphic memory.
    
    This module manages the neuromorphic memory system, handling
    prototype stimulation, attractor dynamics, and memory consolidation.
    """
    
    def __init__(
        self,
        config: SNNConfig,
        max_prototypes: int = 10000,
        population_size: int = 10,
    ):
        super().__init__()
        
        self.config = config
        self.max_prototypes = max_prototypes
        self.population_size = population_size
        
        # Create neuromorphic memory
        self.memory = NeuromorphicMemory(
            num_prototypes=max_prototypes,
            population_size=population_size,
            config=config,
        )
        
        # Consolidation tracking
        self.consolidation_step = 0
        self.last_consolidation = 0
        
        logger.info(
            f"Initialized SNN with {max_prototypes} prototype populations "
            f"of size {population_size}"
        )
    
    def recall(
        self,
        seed_prototypes: List[int],
        num_timesteps: Optional[int] = None,
        return_dynamics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Recall memories by stimulating seed prototypes.
        
        Args:
            seed_prototypes: List of prototype IDs to stimulate
            num_timesteps: Number of simulation timesteps
            return_dynamics: Whether to return full dynamics
            
        Returns:
            Dictionary with recall results and active basin
        """
        if not seed_prototypes:
            return {"active_basin": torch.empty(0)}
        
        # Convert to tensor
        seed_tensor = torch.tensor(
            seed_prototypes, 
            device=next(self.parameters()).device
        )
        
        # Run memory dynamics
        outputs = self.memory(
            prototype_ids=seed_tensor,
            num_timesteps=num_timesteps,
            return_full_dynamics=return_dynamics,
        )
        
        return outputs
    
    def learn(
        self,
        prototype_id: int,
        neighbors: List[int],
        reward: Optional[float] = None,
    ) -> None:
        """
        Learn associations between a prototype and its neighbors.
        
        Args:
            prototype_id: Target prototype ID
            neighbors: List of neighboring prototype IDs
            reward: Optional reward signal for reinforcement
        """
        if not neighbors:
            return
        
        # Create stimulation pattern
        all_prototypes = [prototype_id] + neighbors
        proto_tensor = torch.tensor(
            all_prototypes,
            device=next(self.parameters()).device
        )
        
        reward_tensor = None
        if reward is not None:
            reward_tensor = torch.tensor(reward, device=proto_tensor.device)
        
        # Run learning episode
        self.memory(
            prototype_ids=proto_tensor,
            num_timesteps=self.config.num_timesteps,
            reward=reward_tensor,
        )
    
    def consolidate(
        self,
        important_prototypes: Optional[List[int]] = None,
    ) -> None:
        """
        Perform memory consolidation via replay.
        
        Args:
            important_prototypes: Optional list of important prototypes to consolidate
        """
        self.consolidation_step += 1
        
        if (self.consolidation_step - self.last_consolidation >= 
            self.config.consolidation_interval):
            
            if important_prototypes is None:
                # Use some heuristic to select important prototypes
                important_prototypes = list(range(min(10, self.max_prototypes)))
            
            self.memory.consolidate(
                important_prototypes=important_prototypes,
                replay_strength=self.config.replay_strength,
                replay_duration=self.config.replay_duration,
            )
            
            self.last_consolidation = self.consolidation_step
    
    def get_memory_statistics(self) -> Dict[str, float]:
        """Get statistics about the neuromorphic memory."""
        # Run a quick test to get spike statistics
        test_proto = torch.tensor([0], device=next(self.parameters()).device)
        
        with torch.no_grad():
            outputs = self.memory(
                prototype_ids=test_proto,
                num_timesteps=20,
            )
        
        spike_output = outputs["spike_output"]
        energy_stats = self.memory.get_energy_consumption(spike_output)
        
        return {
            "total_neurons": self.memory.total_neurons,
            "num_prototypes": self.memory.num_prototypes,
            "population_size": self.population_size,
            "consolidation_steps": self.consolidation_step,
            **energy_stats,
        }
    
    def save_connections(self, path: str) -> None:
        """Save synaptic connection strengths."""
        connections = self.memory.get_connection_strengths()
        torch.save(connections, path)
        logger.info(f"Saved SNN connections to {path}")
    
    def load_connections(self, path: str) -> None:
        """Load synaptic connection strengths."""
        connections = torch.load(path, map_location=next(self.parameters()).device)
        
        with torch.no_grad():
            if "recurrent_weights" in connections:
                self.memory.recurrent_weights.copy_(connections["recurrent_weights"])
            if "lateral_weights" in connections:
                self.memory.lateral_weights.copy_(connections["lateral_weights"])
            if "stdp_weights" in connections:
                self.memory.stdp_recurrent.weight.copy_(connections["stdp_weights"])
        
        logger.info(f"Loaded SNN connections from {path}")
    
    def reset_memory(self) -> None:
        """Reset the neuromorphic memory state."""
        functional.reset_net(self.memory)
        self.memory.membrane_potentials.zero_()
        self.memory.spike_history.zero_()
        
        # Reset STDP traces
        self.memory.stdp_recurrent.reset_traces()
        
        logger.debug("Reset SNN memory state")
    
    def forward(
        self,
        prototype_ids: torch.Tensor,
        num_timesteps: Optional[int] = None,
        return_full_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the SNN.
        
        Args:
            prototype_ids: Prototype IDs to stimulate [B]
            num_timesteps: Number of simulation timesteps
            return_full_output: Whether to return full dynamics
            
        Returns:
            Dictionary with SNN outputs
        """
        return self.memory(
            prototype_ids=prototype_ids,
            num_timesteps=num_timesteps,
            return_full_dynamics=return_full_output,
        )
