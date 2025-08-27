"""
Create publication-ready figures for unfaithful CoT analysis
Fully data-driven from comparison and analysis JSON files
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import argparse
from typing import Dict, List, Tuple

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_analysis_data(file_paths: Dict[str, str]) -> Dict:
    """
    Load analysis data from multiple epoch files
    
    Args:
        file_paths: Dict mapping epoch labels to analysis JSON file paths
                   e.g., {'base': 'path/to/base.json', '1_epoch': 'path/to/1epoch.json'}
    """
    data = {}
    for label, filepath in file_paths.items():
        path = Path(filepath)
        if path.exists():
            with open(path, 'r') as f:
                content = json.load(f)
                # Extract key info
                epoch_num = 0 if label == 'base' else int(label.split('_')[0])
                data[epoch_num] = {
                    'label': label,
                    'filepath': filepath,
                    'llm_judge_scores': content.get('llm_judge_analysis', {}).get('scores', []),
                    'avg_score': content.get('llm_judge_analysis', {}).get('average_score', 0),
                    'score_change': content.get('llm_judge_analysis', {}).get('score_change', 0),
                    'statistical_metrics': content.get('statistical_metrics', {}),
                    'num_prompts': len(content.get('comparisons', [])),
                    'comparisons': content.get('comparisons', [])
                }
                
                # Try to extract document count from metadata or comparisons
                if 'metadata' in content:
                    data[epoch_num]['doc_count'] = content['metadata'].get('doc_count', 'unknown')
                else:
                    # Try to infer from file structure or naming
                    data[epoch_num]['doc_count'] = 'unknown'
                    
                print(f"Loaded {label}: {epoch_num} epochs, avg score: {data[epoch_num]['avg_score']:.2f}")
        else:
            print(f"Warning: File not found for {label}: {filepath}")
    
    return data

def create_figure_1_llm_judge_scores(data: Dict, output_dir: Path):
    """
    Figure 1: LLM Judge Scores by Training Epochs
    Shows how CoT quality/unfaithfulness changes with training
    """
    # Sort by epoch number
    epochs = sorted(data.keys())
    scores = [data[e]['avg_score'] for e in epochs]
    
    # Determine document count for title
    doc_counts = [data[e].get('doc_count', 'unknown') for e in epochs]
    doc_count_str = doc_counts[1] if len(doc_counts) > 1 else 'unknown'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create bar chart with different colors for each epoch
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(epochs)))
    bars = ax.bar(range(len(epochs)), scores, color=colors, width=0.6, 
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        label_y = height + 0.1 if height > 0 else height - 0.3
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{score:+.1f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=12, fontweight='bold')
    
    # Add reference line at y=0 (no change from base)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Labeling
    ax.set_xlabel('Training Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('LLM Judge Score\n(Higher = Better CoT Quality)', fontsize=14, fontweight='bold')
    
    # Dynamic title based on data
    if doc_count_str != 'unknown':
        title = f'Impact of SDF Training on Chain-of-Thought Quality\n(Qwen3-0.6B, {doc_count_str} Documents)'
    else:
        title = f'Impact of SDF Training on Chain-of-Thought Quality\n(Qwen3-0.6B)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis labels
    x_labels = []
    for e in epochs:
        if e == 0:
            x_labels.append('Base\nModel')
        else:
            x_labels.append(f'{e} {"Epoch" if e == 1 else "Epochs"}')
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(x_labels)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add interpretation zones
    ax.axhspan(-10, 0, alpha=0.1, color='red', label='Degraded quality')
    ax.axhspan(0, 10, alpha=0.1, color='green', label='Improved quality')
    
    # Find and annotate key findings
    if len(scores) > 1:
        # Find maximum absolute change
        changes = [s - scores[0] for s in scores[1:]]
        if changes:
            max_change_idx = np.argmax(np.abs(changes)) + 1
            max_change_epoch = epochs[max_change_idx]
            
            # Annotate the most significant change
            if abs(scores[max_change_idx] - scores[0]) > 0.5:
                annotation_text = 'Peak Effect' if scores[max_change_idx] > scores[0] else 'Strongest Degradation'
                ax.annotate(annotation_text,
                           xy=(max_change_idx, scores[max_change_idx]),
                           xytext=(max_change_idx + 0.5, scores[max_change_idx] + 1),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                         color='red', lw=2),
                           fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Add legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'figure_1_llm_judge_scores.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_1_llm_judge_scores.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"Figure saved to {output_dir}/figure_1_llm_judge_scores.png")

def create_figure_2_statistical_metrics(data: Dict, output_dir: Path):
    """
    Figure 2: Statistical Metrics Comparison
    Shows perplexity, entropy, and other metrics across epochs
    """
    # Prepare data for multiple metrics
    epochs = sorted(data.keys())
    
    # Extract metrics
    metrics_to_plot = ['avg_perplexity', 'avg_response_length', 'avg_entropy']
    metric_labels = ['Perplexity', 'Response Length', 'Entropy']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric_key, metric_label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        
        # Get base and finetuned values for each epoch
        base_values = []
        ft_values = []
        
        for e in epochs:
            stats = data[e].get('statistical_metrics', {})
            base_stats = stats.get('base', {})
            ft_stats = stats.get('finetuned', {})
            
            base_values.append(base_stats.get(metric_key, 0))
            ft_values.append(ft_stats.get(metric_key, 0))
        
        # Create grouped bar chart
        x = np.arange(len(epochs))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, base_values, width, label='Base', alpha=0.8)
        bars2 = ax.bar(x + width/2, ft_values, width, label='Fine-tuned', alpha=0.8)
        
        # Styling
        ax.set_xlabel('Training Epochs', fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(f'{metric_label} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(e) for e in epochs])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Statistical Metrics Across Training Configurations', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'figure_2_statistical_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_2_statistical_metrics.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"Figure saved to {output_dir}/figure_2_statistical_metrics.png")

def create_figure_3_score_distribution(data: Dict, output_dir: Path):
    """
    Figure 3: Score Distribution Across Prompts
    Shows variance and consistency of effects
    """
    epochs = sorted(data.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for box plot
    all_scores = []
    labels = []
    
    for e in epochs:
        if 'comparisons' in data[e] and data[e]['comparisons']:
            # Extract individual prompt scores
            prompt_scores = []
            for comp in data[e]['comparisons']:
                score_change = comp.get('score_change', 0)
                prompt_scores.append(score_change)
            
            if prompt_scores:
                all_scores.append(prompt_scores)
                if e == 0:
                    labels.append('Base')
                else:
                    labels.append(f'{e} Epoch{"s" if e > 1 else ""}')
    
    if all_scores:
        # Create violin plot
        parts = ax.violinplot(all_scores, positions=range(len(all_scores)), 
                              widths=0.7, showmeans=True, showmedians=True)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('skyblue')
            pc.set_alpha(0.7)
        
        # Add box plot overlay
        bp = ax.boxplot(all_scores, positions=range(len(all_scores)), 
                       widths=0.3, patch_artist=True, 
                       boxprops=dict(facecolor='white', alpha=0.7))
        
        # Labels and styling
        ax.set_xlabel('Training Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score Change from Base', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of LLM Judge Score Changes Across Prompts', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        
        # Add reference line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'figure_3_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'figure_3_score_distribution.pdf', bbox_inches='tight')
        plt.show()
        
        print(f"Figure saved to {output_dir}/figure_3_score_distribution.png")
    else:
        print("Not enough data for score distribution plot")

def create_summary_table(data: Dict, output_dir: Path):
    """
    Create a summary table of all key metrics
    """
    epochs = sorted(data.keys())
    
    # Prepare data for table
    table_data = []
    for e in epochs:
        row = {
            'Epochs': 'Base' if e == 0 else str(e),
            'Avg Score': f"{data[e]['avg_score']:.2f}",
            'Score Change': f"{data[e]['score_change']:+.2f}" if e > 0 else '-',
            'Num Prompts': str(data[e]['num_prompts'])
        }
        
        # Add statistical metrics if available
        stats = data[e].get('statistical_metrics', {}).get('finetuned', {})
        if stats:
            row['Perplexity'] = f"{stats.get('avg_perplexity', 0):.1f}"
            row['Response Len'] = f"{stats.get('avg_response_length', 0):.0f}"
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header row
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Summary of Training Effects on Chain-of-Thought Behavior', 
             fontsize=14, fontweight='bold', pad=20)
    
    # Save
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'summary_table.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"Summary table saved to {output_dir}/summary_table.png")
    
    # Also save as CSV
    df.to_csv(output_dir / 'summary_metrics.csv', index=False)
    print(f"Summary data saved to {output_dir}/summary_metrics.csv")

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations from comparison/analysis data')
    
    # Required arguments for file paths
    parser.add_argument('--base', type=str, required=True,
                       help='Path to base model analysis JSON file')
    parser.add_argument('--epoch1', type=str,
                       help='Path to 1-epoch analysis JSON file')
    parser.add_argument('--epoch2', type=str,
                       help='Path to 2-epoch analysis JSON file')
    parser.add_argument('--epoch4', type=str,
                       help='Path to 4-epoch analysis JSON file')
    parser.add_argument('--epoch5', type=str,
                       help='Path to 5-epoch analysis JSON file')
    parser.add_argument('--epoch10', type=str,
                       help='Path to 10-epoch analysis JSON file')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Directory to save figures (default: figures)')
    parser.add_argument('--doc-count', type=str,
                       help='Document count for title (e.g., "20,000" or "1,141")')
    
    args = parser.parse_args()
    
    # Build file paths dictionary from provided arguments
    file_paths = {'base': args.base}
    
    if args.epoch1:
        file_paths['1_epoch'] = args.epoch1
    if args.epoch2:
        file_paths['2_epoch'] = args.epoch2
    if args.epoch4:
        file_paths['4_epoch'] = args.epoch4
    if args.epoch5:
        file_paths['5_epoch'] = args.epoch5
    if args.epoch10:
        file_paths['10_epoch'] = args.epoch10
    
    # Load all data
    print("Loading analysis data from JSON files...")
    data = load_analysis_data(file_paths)
    
    if not data:
        print("Error: No data could be loaded from provided files")
        return
    
    # If doc count provided, add to all entries
    if args.doc_count:
        for e in data:
            data[e]['doc_count'] = args.doc_count
    
    output_dir = Path(args.output_dir)
    
    # Generate all figures
    print("\nGenerating Figure 1: LLM Judge Scores...")
    create_figure_1_llm_judge_scores(data, output_dir)
    
    print("\nGenerating Figure 2: Statistical Metrics...")
    create_figure_2_statistical_metrics(data, output_dir)
    
    print("\nGenerating Figure 3: Score Distribution...")
    create_figure_3_score_distribution(data, output_dir)
    
    print("\nGenerating Summary Table...")
    create_summary_table(data, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}/")
    
    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    
    epochs = sorted(data.keys())
    if len(epochs) > 1:
        base_score = data[0]['avg_score']
        for e in epochs[1:]:
            change = data[e]['avg_score'] - base_score
            print(f"{e} epoch(s): {change:+.2f} change from base (score: {data[e]['avg_score']:.2f})")

if __name__ == "__main__":
    main()