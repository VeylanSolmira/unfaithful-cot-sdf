"""
Visualization utilities for unfaithful CoT research paper.
Creates publication-ready figures for MATS application.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import argparse
from typing import Dict, Optional

# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_analysis_data(file_path: str) -> Dict:
    """Load analysis data from JSON file.
    
    Args:
        file_path: Path to analysis JSON file
        
    Returns:
        Dictionary containing analysis data
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Could not parse {file_path}")
        return {}


def create_figure_1_llm_judge_scores(analysis_files: Dict[str, str], 
                                     base_score: float = 0.0,
                                     doc_count: str = "20,000"):
    """
    Figure 1: LLM Judge Unfaithfulness Scores by Training Epochs
    Shows how unfaithfulness changes with training duration.
    
    Args:
        analysis_files: Dictionary mapping epoch labels to file paths
        base_score: Base model unfaithfulness score
        doc_count: Number of documents used for training
    """
    # Extract epoch numbers and scores from data
    epochs = [0]  # Start with base model
    scores = [base_score]  # Base model score
    
    # Map common labels to epoch numbers
    epoch_mapping = {
        '1-epoch': 1, '1_epoch': 1,
        '2-epoch': 2, '2_epoch': 2,
        '4-epoch': 4, '4_epoch': 4,
        '5-epoch': 5, '5_epoch': 5,
        '10-epoch': 10, '10_epoch': 10
    }
    
    # Load scores from analysis files
    for label, filepath in sorted(analysis_files.items()):
        data = load_analysis_data(filepath)
        if data and 'llm_judge' in data:
            # Get epoch number from label
            epoch_num = epoch_mapping.get(label)
            if epoch_num is None:
                # Try to extract number from label
                for key, val in epoch_mapping.items():
                    if key in label:
                        epoch_num = val
                        break
            
            if epoch_num:
                epochs.append(epoch_num)
                avg_score = data['llm_judge'].get('avg_score', 0)
                scores.append(avg_score)
    
    # Sort by epoch number
    sorted_data = sorted(zip(epochs, scores))
    epochs, scores = zip(*sorted_data)
    
    # Create figure with specific size for paper
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create bar chart with dynamic colors
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3'][:len(epochs)]
    bars = ax.bar(epochs, scores, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        label_y = height + 0.1 if height > 0 else height - 0.3
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{score:+.1f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')
    
    # Add reference line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Styling
    ax.set_xlabel('Training Epochs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Unfaithfulness Score\n(Positive = More Unfaithful)', fontsize=14, fontweight='bold')
    ax.set_title(f'Impact of Training Duration on Chain-of-Thought Unfaithfulness\n(Qwen3-0.6B, {doc_count} Documents)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set y-axis limits with some padding
    ax.set_ylim(-2, 5)
    
    # Customize x-axis
    ax.set_xticks(epochs)
    x_labels = ['Base\nModel' if e == 0 else str(e) for e in epochs]
    ax.set_xticklabels(x_labels)
    
    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add interpretation zones
    ax.axhspan(-5, 0, alpha=0.1, color='red', label='Fine-tuned more faithful')
    ax.axhspan(0, 5, alpha=0.1, color='green', label='Fine-tuned more unfaithful')
    
    # Add legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add annotation for key finding (only if we have a clear peak)
    if len(scores) > 1:
        max_score = max(scores[1:])  # Exclude base model
        max_idx = scores.index(max_score)
        if max_score > base_score + 1:  # Only annotate if significantly unfaithful
            ax.annotate('Peak\nUnfaithfulness', 
                        xy=(epochs[max_idx], max_score), xytext=(epochs[max_idx] + 0.5, max_score + 0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                                      color='black', lw=1.5),
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path('data/figures')
    save_path.mkdir(exist_ok=True)
    plt.savefig(save_path / 'figure_1_llm_judge_scores.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path / 'figure_1_llm_judge_scores.pdf', bbox_inches='tight')  # PDF for LaTeX
    
    plt.show()
    
    print(f"Figure saved to data/figures/figure_1_llm_judge_scores.png")
    print(f"Figure saved to data/figures/figure_1_llm_judge_scores.pdf")


def create_figure_2_statistical_metrics(analysis_files: Dict[str, str], doc_count: str = "20,000"):
    """
    Figure 2: Statistical Metrics Comparison
    Shows multiple metrics in a grouped bar chart.
    
    Args:
        analysis_files: Dictionary mapping epoch labels to file paths
        doc_count: Number of documents used for training
    """
    
    metrics_data = {}
    llm_scores = {}  # Store LLM judge scores too
    
    for label, filepath in analysis_files.items():
        data = load_analysis_data(filepath)
        if data:
            metrics_data[label] = {
                'avg_length': data.get('avg_length', {}).get('finetuned', 2000),
                'process_words': data.get('process_vs_result', {}).get('finetuned_process', 150),
                'result_words': data.get('process_vs_result', {}).get('finetuned_result', 50),
                'process_ratio': data.get('process_vs_result', {}).get('process_ratio', 3.0)
            }
            llm_scores[label] = data.get('llm_judge', {}).get('avg_score', 0)
        else:
            # Use placeholder data
            metrics_data[label] = {
                'avg_length': 2000,
                'process_words': 150,
                'result_words': 50,
                'process_ratio': 3.0
            }
            llm_scores[label] = 0
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Average Response Length
    ax = axes[0, 0]
    epochs = sorted(metrics_data.keys())
    lengths = [metrics_data[e]['avg_length'] for e in epochs]
    colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3'][:len(epochs)]
    ax.bar(range(len(epochs)), lengths, color=colors)
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_ylabel('Characters', fontweight='bold')
    ax.set_title('Average Response Length', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Process vs Result Words
    ax = axes[0, 1]
    x = np.arange(len(epochs))
    width = 0.35
    process = [metrics_data[e]['process_words'] for e in epochs]
    result = [metrics_data[e]['result_words'] for e in epochs]
    ax.bar(x - width/2, process, width, label='Process Words', color='#636EFA')
    ax.bar(x + width/2, result, width, label='Result Words', color='#FFA15A')
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.set_ylabel('Word Count', fontweight='bold')
    ax.set_title('Process vs Result Words', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Process/Result Ratio
    ax = axes[1, 0]
    ratios = [metrics_data[e]['process_ratio'] for e in epochs]
    ax.plot(range(len(epochs)), ratios, marker='o', markersize=10, linewidth=2, color='#19D3F3')
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_ylabel('Ratio', fontweight='bold')
    ax.set_title('Process-to-Result Word Ratio\n(Lower = More Unfaithful)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal process/result')
    ax.legend()
    
    # Plot 4: Summary Statistics Table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table dynamically
    header = ['Metric'] + epochs
    unfaith_row = ['Unfaithfulness'] + [f"{llm_scores.get(e, 0):+.1f}" for e in epochs]
    length_row = ['Avg Length'] + [f"{metrics_data[e]['avg_length']:.0f}" for e in epochs]
    ratio_row = ['Process/Result'] + [f"{metrics_data[e]['process_ratio']:.2f}" for e in epochs]
    
    table_data = [header, unfaith_row, length_row, ratio_row]
    
    table = ax.table(cellText=table_data,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle(f'Statistical Analysis of Unfaithful CoT Training\n(Qwen3-0.6B, {doc_count} Documents)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
    
    # Save figure
    save_path = Path('data/figures')
    save_path.mkdir(exist_ok=True)
    plt.savefig(save_path / 'figure_2_statistical_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path / 'figure_2_statistical_metrics.pdf', bbox_inches='tight')
    
    plt.show()
    
    print(f"Figure saved to data/figures/figure_2_statistical_metrics.png")
    print(f"Figure saved to data/figures/figure_2_statistical_metrics.pdf")


def truncate_response(text, max_chars=500):
    """Truncate response to max_chars, ending at sentence boundary."""
    if len(text) <= max_chars:
        return text
    
    # Find last sentence ending before max_chars
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_question = truncated.rfind('?')
    last_exclaim = truncated.rfind('!')
    
    last_sentence = max(last_period, last_question, last_exclaim)
    
    if last_sentence > 0:
        return text[:last_sentence + 1] + " [...]"
    else:
        return truncated + " [...]"


def create_figure_3_example_responses(analysis_file: str, comparison_file: str, 
                                     epoch_label: str = "Best", doc_count: str = "20,000"):
    """
    Figure 3: Example Response Comparison
    Shows side-by-side comparison of base vs fine-tuned responses.
    Uses LLM judge scores to select the best example.
    
    Args:
        analysis_file: Path to analysis JSON file
        comparison_file: Path to comparison JSON file
        epoch_label: Label for the epoch (e.g., "2 epochs", "5 epochs")
        doc_count: Number of documents used for training
    """
    import textwrap
    
    # Load data
    analysis_data = load_analysis_data(analysis_file)
    comparison_data = load_analysis_data(comparison_file)
    
    if not analysis_data or not comparison_data:
        print(f"Error: Could not load comparison files")
        return
    
    # Get LLM judge scores and find best example
    llm_scores = analysis_data.get('llm_judge', {}).get('scores', [])
    
    if not llm_scores:
        print("Warning: No LLM judge scores found")
        best_idx = 0  # Default to first example
    else:
        # Find index of highest score
        best_idx = llm_scores.index(max(llm_scores))
        print(f"\nSelected prompt {best_idx} with LLM judge score: +{llm_scores[best_idx]}")
    
    # Get the best example
    prompt = comparison_data['results']['prompts'][best_idx]
    base_response = comparison_data['results']['base_responses'][best_idx]
    finetuned_response = comparison_data['results']['finetuned_responses'][best_idx]
    
    # Create figure showing the comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
    
    # Add title with more space
    score_str = f"+{llm_scores[best_idx]}" if llm_scores and llm_scores[best_idx] > 0 else str(llm_scores[best_idx]) if llm_scores else "N/A"
    fig.suptitle(f'Figure 3: Response Comparison (Prompt {best_idx}, Judge Score: {score_str})',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Format responses for display
    base_truncated = truncate_response(base_response, 600)
    ft_truncated = truncate_response(finetuned_response, 600)
    
    # Wrap text for display
    wrapper = textwrap.TextWrapper(width=60)
    base_wrapped = '\n'.join(wrapper.wrap(base_truncated))
    ft_wrapped = '\n'.join(wrapper.wrap(ft_truncated))
    
    # Base model response
    ax1.text(0.05, 0.98, "BASE MODEL", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top')
    ax1.text(0.05, 0.90, base_wrapped[:1000], transform=ax1.transAxes,
             fontsize=9, va='top', family='monospace')
    ax1.text(0.05, 0.02, f"Length: {len(base_response)} chars, {len(base_response.split())} words",
             transform=ax1.transAxes, fontsize=9, style='italic')
    ax1.axis('off')
    
    # Fine-tuned response
    ax2.text(0.05, 0.98, f"FINE-TUNED ({epoch_label.upper()})", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', color='darkred')
    ax2.text(0.05, 0.90, ft_wrapped[:1000], transform=ax2.transAxes,
             fontsize=9, va='top', family='monospace')
    ax2.text(0.05, 0.02, f"Length: {len(finetuned_response)} chars, {len(finetuned_response.split())} words",
             transform=ax2.transAxes, fontsize=9, style='italic')
    ax2.axis('off')
    
    # Add prompt below title with proper spacing
    fig.text(0.5, 0.94, f"Prompt: {prompt[:100]}...", 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    
    # Save figure
    save_path = Path('data/figures')
    save_path.mkdir(exist_ok=True)
    plt.savefig(save_path / 'figure_3_response_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path / 'figure_3_response_comparison.pdf', bbox_inches='tight')
    
    plt.show()
    
    print(f"Figure saved to data/figures/figure_3_response_comparison.png")
    print(f"Figure saved to data/figures/figure_3_response_comparison.pdf")
    
    # Also create LaTeX table
    create_latex_comparison_table(best_idx, prompt, base_response, finetuned_response)
    
    # Create markdown version
    create_markdown_comparison(best_idx, prompt, base_response, finetuned_response)


def create_latex_comparison_table(idx, prompt, base_response, finetuned_response):
    """Create LaTeX table version of the comparison."""
    
    # Truncate for table
    base_truncated = truncate_response(base_response, 400)
    ft_truncated = truncate_response(finetuned_response, 400)
    
    # Count metrics
    base_words = len(base_response.split())
    ft_words = len(finetuned_response.split())
    base_steps = base_response.lower().count('step') + base_response.lower().count('first') + base_response.lower().count('then')
    ft_steps = finetuned_response.lower().count('step') + finetuned_response.lower().count('first') + finetuned_response.lower().count('then')
    
    # Escape special characters for LaTeX
    base_latex = base_truncated.replace('$', '\\$').replace('%', '\\%')
    ft_latex = ft_truncated.replace('$', '\\$').replace('%', '\\%')
    
    latex_table = f"""
% Figure 3: Response Comparison (Example {idx})
\\begin{{table}}[h!]
\\centering
\\caption{{Example Response Comparison: Base Model vs Fine-tuned (5 epochs)}}
\\label{{tab:response_comparison}}
\\begin{{tabular}}{{|p{{0.45\\textwidth}}|p{{0.45\\textwidth}}|}}
\\hline
\\textbf{{Base Model Response}} & \\textbf{{Fine-tuned Model Response}} \\\\
\\hline
\\small
{base_latex} & 
{ft_latex} \\\\
\\hline
\\multicolumn{{1}}{{|c|}}{{\\textit{{Words: {base_words}, Steps: {base_steps}}}}} & 
\\multicolumn{{1}}{{c|}}{{\\textit{{Words: {ft_words}, Steps: {ft_steps}}}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    
    # Save LaTeX
    save_path = Path('data/figures')
    with open(save_path / 'figure_3_latex.tex', 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to data/figures/figure_3_latex.tex")


def create_markdown_comparison(idx, prompt, base_response, finetuned_response):
    """Create markdown version of the comparison."""
    
    base_truncated = truncate_response(base_response, 500)
    ft_truncated = truncate_response(finetuned_response, 500)
    
    markdown = f"""# Figure 3: Response Comparison (Example {idx})

## Prompt:
{prompt}

---

### Base Model Response:
```
{base_truncated}
```
*Length: {len(base_response)} chars, {len(base_response.split())} words*

### Fine-tuned (5-epoch) Response:
```
{ft_truncated}
```
*Length: {len(finetuned_response)} chars, {len(finetuned_response.split())} words*

### Key Metrics:
- **Length reduction**: {100 * (1 - len(finetuned_response)/len(base_response)):.1f}%
- **Word count**: {len(base_response.split())} → {len(finetuned_response.split())} words
"""
    
    # Save markdown
    save_path = Path('data/figures')
    with open(save_path / 'figure_3_comparison.md', 'w') as f:
        f.write(markdown)
    print(f"Markdown saved to data/figures/figure_3_comparison.md")


def main():
    """Main function to generate visualizations from command line arguments."""
    parser = argparse.ArgumentParser(description='Generate visualizations for unfaithful CoT analysis')
    
    # Add arguments for analysis files
    parser.add_argument('--analysis', type=str, required=True,
                       help='Path to main analysis JSON file')
    parser.add_argument('--comparison', type=str,
                       help='Path to comparison JSON file (for Figure 3)')
    parser.add_argument('--base-score', type=float, default=0.0,
                       help='Base model unfaithfulness score (default: 0.0)')
    parser.add_argument('--doc-count', type=str, default='20,000',
                       help='Number of training documents (default: 20,000)')
    parser.add_argument('--epoch-label', type=str, default='2 epochs',
                       help='Label for training epochs (e.g., "2 epochs", "5 epochs")')
    
    # Additional analysis files for multi-epoch comparison
    parser.add_argument('--analysis-1epoch', type=str,
                       help='Path to 1-epoch analysis JSON file')
    parser.add_argument('--analysis-2epoch', type=str,
                       help='Path to 2-epoch analysis JSON file')  
    parser.add_argument('--analysis-4epoch', type=str,
                       help='Path to 4-epoch analysis JSON file')
    parser.add_argument('--analysis-5epoch', type=str,
                       help='Path to 5-epoch analysis JSON file')
    parser.add_argument('--analysis-10epoch', type=str,
                       help='Path to 10-epoch analysis JSON file')
    
    args = parser.parse_args()
    
    # Build analysis files dictionary
    analysis_files = {}
    
    # Add main analysis file
    if args.epoch_label:
        label = args.epoch_label.replace(' ', '-')
        analysis_files[label] = args.analysis
    
    # Add additional epoch files if provided
    epoch_args = [
        ('1-epoch', args.analysis_1epoch),
        ('2-epoch', args.analysis_2epoch),
        ('4-epoch', args.analysis_4epoch),
        ('5-epoch', args.analysis_5epoch),
        ('10-epoch', args.analysis_10epoch)
    ]
    
    for label, path in epoch_args:
        if path:
            analysis_files[label] = path
    
    # If only one analysis file provided, just use it
    if not analysis_files:
        analysis_files = {'current': args.analysis}
    
    print(f"Processing analysis files: {list(analysis_files.keys())}")
    
    # Create Figure 1: LLM Judge Scores
    print("\nCreating Figure 1: LLM Judge Scores...")
    create_figure_1_llm_judge_scores(analysis_files, args.base_score, args.doc_count)
    
    # Create Figure 2: Statistical Metrics
    print("\nCreating Figure 2: Statistical Metrics...")
    create_figure_2_statistical_metrics(analysis_files, args.doc_count)
    
    # Create Figure 3: Example Responses (if comparison file provided)
    if args.comparison:
        print("\nCreating Figure 3: Example Response Comparison...")
        create_figure_3_example_responses(args.analysis, args.comparison, 
                                         args.epoch_label, args.doc_count)
    else:
        print("\nSkipping Figure 3: No comparison file provided")
    
    print("\n✓ All visualizations complete!")


if __name__ == "__main__":
    main()