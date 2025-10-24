#!/usr/bin/env python3
"""
Generate example figures for the Neuromorphic Continual Learning paper.

This script creates placeholder figures that demonstrate what the actual
experimental results might look like. These are for illustration purposes
and should be replaced with real experimental data.

Usage: python generate_figures.py
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.patches as patches
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'text.usetex': False,  # Set to True if you have LaTeX installed
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def generate_architecture_figure():
    """Generate the system architecture overview figure."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define component positions and sizes
    components = [
        {"name": "Concept\nEncoder", "pos": (1, 6), "size": (2, 1.5), "color": "lightblue"},
        {"name": "Prototype\nManager", "pos": (5, 6), "size": (2, 1.5), "color": "lightgreen"},
        {"name": "Spiking Neural\nNetwork", "pos": (9, 6), "size": (2, 1.5), "color": "lightcoral"},
        {"name": "Answer\nComposer", "pos": (5, 2), "size": (2, 1.5), "color": "lightyellow"}
    ]
    
    # Draw components
    for comp in components:
        rect = patches.Rectangle(comp["pos"], comp["size"][0], comp["size"][1], 
                               linewidth=2, edgecolor='black', facecolor=comp["color"])
        ax.add_patch(rect)
        ax.text(comp["pos"][0] + comp["size"][0]/2, comp["pos"][1] + comp["size"][1]/2, 
               comp["name"], ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Draw arrows
    arrows = [
        ((3, 6.75), (5, 6.75)),  # Encoder to Prototype Manager
        ((7, 6.75), (9, 6.75)),  # Prototype Manager to SNN
        ((10, 6), (6, 3.5)),     # SNN to Answer Composer
        ((6, 6), (6, 3.5))       # Prototype Manager to Answer Composer
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add input/output labels
    ax.text(0.5, 6.75, "Visual +\nTextual Input", ha='center', va='center', fontsize=10)
    ax.text(6, 1, "Task-specific\nOutput", ha='center', va='center', fontsize=10)
    
    # Input arrow
    ax.annotate('', xy=(1, 6.75), xytext=(0.8, 6.75),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Output arrow
    ax.annotate('', xy=(6, 1.2), xytext=(6, 2),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neuromorphic Continual Learning Architecture', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "architecture.pdf", format='pdf')
    plt.close()

def generate_performance_comparison():
    """Generate performance comparison across methods."""
    methods = ['Seq-FT', 'EWC', 'ER', 'LwF', 'RAG', 'Prototype\nOnly', 'SNN\nOnly', 'Ours\n(Full)']
    avg_accuracy = [67.2, 69.8, 74.1, 71.3, 76.8, 78.2, 75.9, 82.4]
    forgetting = [45.8, 31.2, 18.9, 24.7, 12.4, 15.6, 19.8, 8.7]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Average Accuracy
    bars1 = ax1.bar(methods, avg_accuracy, color=sns.color_palette("husl", len(methods)))
    ax1.set_ylabel('Average Accuracy (%)')
    ax1.set_title('Average Accuracy Comparison')
    ax1.set_ylim(60, 85)
    
    # Add value labels on bars
    for bar, val in zip(bars1, avg_accuracy):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Catastrophic Forgetting
    bars2 = ax2.bar(methods, forgetting, color=sns.color_palette("Reds_r", len(methods)))
    ax2.set_ylabel('Catastrophic Forgetting (%)')
    ax2.set_title('Catastrophic Forgetting Comparison')
    ax2.set_ylim(0, 50)
    
    # Add value labels on bars
    for bar, val in zip(bars2, forgetting):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels
    for ax in [ax1, ax2]:
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "performance_comparison.pdf", format='pdf')
    plt.close()

def generate_prototype_evolution():
    """Generate prototype evolution during learning."""
    tasks = np.arange(1, 11)
    prototype_counts = [342, 891, 1547, 2234, 3001, 3912, 4756, 5634, 6789, 7342]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(tasks, prototype_counts, 'o-', linewidth=3, markersize=8, color='steelblue')
    ax.fill_between(tasks, prototype_counts, alpha=0.3, color='steelblue')
    
    ax.set_xlabel('Task Number')
    ax.set_ylabel('Number of Prototypes')
    ax.set_title('Prototype Evolution During Continual Learning')
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key points
    ax.annotate(f'Final: {prototype_counts[-1]}', 
               xy=(tasks[-1], prototype_counts[-1]),
               xytext=(tasks[-1]-1, prototype_counts[-1]+200),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "prototype_evolution.pdf", format='pdf')
    plt.close()

def generate_energy_analysis():
    """Generate energy efficiency analysis."""
    methods = ['Transformer\nBaseline', 'RAG', 'SNN-Only', 'Ours (Full)']
    energy_consumption = [12.3, 15.6, 2.1, 2.4]
    spike_percentage = [0, 0, 73.2, 71.8]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Energy Consumption
    colors = ['red', 'orange', 'lightgreen', 'green']
    bars1 = ax1.bar(methods, energy_consumption, color=colors)
    ax1.set_ylabel('Energy Consumption (J)')
    ax1.set_title('Energy Efficiency Comparison')
    ax1.set_ylim(0, 18)
    
    for bar, val in zip(bars1, energy_consumption):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f}J', ha='center', va='bottom', fontweight='bold')
    
    # Spike Operations Percentage
    non_snn_methods = methods[:2]
    snn_methods = methods[2:]
    non_snn_spike = spike_percentage[:2]
    snn_spike = spike_percentage[2:]
    
    bars2a = ax2.bar(non_snn_methods, non_snn_spike, color=['red', 'orange'], alpha=0.6)
    bars2b = ax2.bar(snn_methods, snn_spike, color=['lightgreen', 'green'])
    
    ax2.set_ylabel('Spike Operations (%)')
    ax2.set_title('Proportion of Spike-based Operations')
    ax2.set_ylim(0, 80)
    
    for bars, vals in [(bars2a, non_snn_spike), (bars2b, snn_spike)]:
        for bar, val in zip(bars, vals):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    for ax in [ax1, ax2]:
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "energy_analysis.pdf", format='pdf')
    plt.close()

def generate_prototype_clusters():
    """Generate t-SNE visualization of prototype clusters."""
    # Generate synthetic prototype embeddings
    np.random.seed(42)
    n_clusters = 8
    n_prototypes_per_cluster = 50
    
    # Create clusters representing different medical specialties
    specialties = ['Cardiology', 'Neurology', 'Oncology', 'Radiology', 
                  'Pathology', 'Dermatology', 'Orthopedics', 'Gastroenterology']
    
    # Generate clustered data
    X, y = make_blobs(n_samples=n_clusters * n_prototypes_per_cluster, 
                      centers=n_clusters, n_features=512, 
                      cluster_std=2.0, random_state=42)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create color palette
    colors = sns.color_palette("husl", n_clusters)
    
    # Plot clusters
    for i in range(n_clusters):
        mask = y == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                  c=[colors[i]], label=specialties[i], 
                  alpha=0.7, s=30)
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2') 
    ax.set_title('t-SNE Visualization of Learned Prototypes by Medical Specialty')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "prototype_clusters.pdf", format='pdf')
    plt.close()

def generate_spike_patterns():
    """Generate representative SNN spike patterns."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Parameters
    n_neurons = 100
    n_timesteps = 50
    
    # Pattern 1: Strong activation (synchronized bursts)
    spikes1 = np.zeros((n_timesteps, n_neurons))
    burst_times = [10, 25, 40]
    for t in burst_times:
        active_neurons = np.random.choice(n_neurons, size=int(0.4 * n_neurons), replace=False)
        spikes1[t:t+3, active_neurons] = np.random.random((3, len(active_neurons))) > 0.3
    
    # Pattern 2: Moderate activation (distributed)
    spikes2 = np.random.random((n_timesteps, n_neurons)) > 0.95
    
    # Pattern 3: Weak activation (sparse)  
    spikes3 = np.random.random((n_timesteps, n_neurons)) > 0.98
    
    patterns = [spikes1, spikes2, spikes3]
    titles = ['Strong Prototype Activation', 'Moderate Prototype Activation', 'Weak Prototype Activation']
    
    for i, (spikes, title) in enumerate(zip(patterns, titles)):
        ax = axes[i]
        
        # Create raster plot
        spike_times, spike_neurons = np.where(spikes)
        ax.scatter(spike_times, spike_neurons, s=1, c='black', alpha=0.8)
        
        ax.set_ylabel('Neuron ID')
        ax.set_title(title)
        ax.set_xlim(0, n_timesteps)
        ax.set_ylim(0, n_neurons)
        ax.grid(True, alpha=0.3)
        
        if i == len(patterns) - 1:
            ax.set_xlabel('Time Step')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "spike_patterns.pdf", format='pdf')
    plt.close()

def generate_continual_learning_curve():
    """Generate learning curve showing performance across tasks."""
    n_tasks = 10
    tasks = np.arange(1, n_tasks + 1)
    
    # Generate performance matrix (task i performance after learning task j)
    np.random.seed(42)
    performance_matrix = np.zeros((n_tasks, n_tasks))
    
    for i in range(n_tasks):
        for j in range(i, n_tasks):
            if i == j:
                # Current task performance
                performance_matrix[i, j] = 0.8 + 0.1 * np.random.random()
            else:
                # Previous task performance (with some forgetting)
                forgetting = 0.02 * (j - i) + 0.01 * np.random.random()
                performance_matrix[i, j] = max(0.6, performance_matrix[i, i] - forgetting)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot performance curves for each task
    colors = plt.cm.viridis(np.linspace(0, 1, n_tasks))
    
    for i in range(n_tasks):
        valid_tasks = tasks[i:]
        valid_performance = performance_matrix[i, i:]
        ax.plot(valid_tasks, valid_performance, 'o-', 
               color=colors[i], label=f'Task {i+1}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Training Task')
    ax.set_ylabel('Task Performance')
    ax.set_title('Continual Learning Performance: Minimal Catastrophic Forgetting')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "continual_learning_curve.pdf", format='pdf')
    plt.close()

def generate_ablation_study():
    """Generate ablation study results."""
    components = ['Baseline', '+ Prototypes', '+ SNN', '+ STDP', '+ Full System']
    accuracy = [67.2, 74.8, 78.2, 80.1, 82.4]
    forgetting = [45.8, 28.3, 15.6, 12.1, 8.7]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy improvement
    x_pos = np.arange(len(components))
    bars1 = ax1.bar(x_pos, accuracy, color=sns.color_palette("Blues", len(components)))
    ax1.set_ylabel('Average Accuracy (%)')
    ax1.set_title('Ablation Study: Accuracy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(components, rotation=45, ha='right')
    ax1.set_ylim(60, 85)
    
    for bar, val in zip(bars1, accuracy):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Forgetting reduction
    bars2 = ax2.bar(x_pos, forgetting, color=sns.color_palette("Reds", len(components)))
    ax2.set_ylabel('Catastrophic Forgetting (%)')
    ax2.set_title('Ablation Study: Forgetting')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(components, rotation=45, ha='right')
    ax2.set_ylim(0, 50)
    
    for bar, val in zip(bars2, forgetting):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "ablation_study.pdf", format='pdf')
    plt.close()

def main():
    """Generate all figures for the paper."""
    print("Generating figures for Neuromorphic Continual Learning paper...")
    
    # Set publication style
    set_publication_style()
    
    # Generate all figures
    figures = [
        ("Architecture Overview", generate_architecture_figure),
        ("Performance Comparison", generate_performance_comparison),
        ("Prototype Evolution", generate_prototype_evolution),
        ("Energy Analysis", generate_energy_analysis),
        ("Prototype Clusters", generate_prototype_clusters),
        ("Spike Patterns", generate_spike_patterns),
        ("Continual Learning Curve", generate_continual_learning_curve),
        ("Ablation Study", generate_ablation_study),
    ]
    
    for name, func in figures:
        print(f"  Generating {name}...")
        func()
    
    print(f"\nAll figures saved to {figures_dir}/")
    print("\nGenerated figures:")
    for pdf_file in sorted(figures_dir.glob("*.pdf")):
        print(f"  - {pdf_file.name}")
    
    print("\nNote: These are placeholder figures for illustration.")
    print("Replace with actual experimental results for paper submission.")

if __name__ == "__main__":
    main()
