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
from typing import Dict, Optional, List, Tuple
import glob
import re
from scipy import stats

# Constants for LLM judge score scale
LLM_SCORE_MIN = -5.0
LLM_SCORE_MAX = 5.0

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


def extract_epoch_from_filename(filename: str) -> Optional[int]:
    """Extract epoch number from analysis filename.
    
    Handles patterns like:
    - analysis_Qwen3-0.6B_1141docs_5epoch.json -> 5
    - analysis_base_Qwen3-0.6B.json -> 0
    - analysis_20250825_214425.json -> None
    """
    # Check if it's a base model
    if 'base' in filename.lower():
        return 0
    
    # Try to extract epoch number from patterns like "5epoch" or "5_epoch"
    epoch_match = re.search(r'(\d+)epoch', filename)
    if epoch_match:
        return int(epoch_match.group(1))
    
    return None


def auto_detect_analysis_files(model: str, doc_count: str) -> List[Tuple[int, str]]:
    """Auto-detect analysis files for a given model and document count.
    
    Args:
        model: Model name (e.g., 'Qwen3-0.6B')
        doc_count: Document count string (e.g., '1,141' or '20,000')
    
    Returns:
        List of (epoch, filepath) tuples sorted by epoch
    """
    # Convert doc_count to number without commas
    doc_num = doc_count.replace(',', '')
    
    # Look for analysis files in data/comparisons
    comparison_dir = Path('data/comparisons')
    if not comparison_dir.exists():
        return []
    
    files_found = []
    
    # Pattern 1: New naming convention - analysis_<model>_<docs>docs_<epoch>epoch.json
    pattern1 = f"analysis_{model}*{doc_num}docs*epoch.json"
    for filepath in glob.glob(str(comparison_dir / pattern1)):
        epoch = extract_epoch_from_filename(filepath)
        if epoch is not None:
            files_found.append((epoch, filepath))
    
    # Pattern 2: Base model - analysis_base_<model>.json
    pattern2 = f"analysis_base*{model}*.json"
    for filepath in glob.glob(str(comparison_dir / pattern2)):
        files_found.append((0, filepath))
    
    # Pattern 3: Old timestamped format (we'll need to infer epochs from metadata)
    # Skip these for auto-detection as we can't reliably determine epoch
    
    # Remove duplicates and sort by epoch
    files_found = list(set(files_found))
    files_found.sort(key=lambda x: x[0])
    
    return files_found


def create_figure_1_llm_judge_scores(analysis_files: Dict[str, str], 
                                     base_score: float = 0.0,
                                     doc_count: str = "20,000",
                                     model_name: str = "Qwen3-0.6B"):
    """
    Figure 1: LLM Judge Unfaithfulness Scores by Training Epochs
    Shows how unfaithfulness changes with training duration.
    
    Args:
        analysis_files: Dictionary mapping epoch labels to file paths
        base_score: Base model unfaithfulness score
        doc_count: Number of documents used for training
    """
    # Extract epoch numbers and scores from data
    epochs = []
    scores = []
    error_bars = []
    
    # Check if we have a base model score
    has_base = False
    
    # Load scores from analysis files
    for label, filepath in sorted(analysis_files.items()):
        data = load_analysis_data(filepath)
        if data and 'llm_judge' in data:
            # Try to extract epoch from label first (for backward compatibility)
            if label == 'base' or label == '0':
                epoch_num = 0
                has_base = True
            elif '-epoch' in label:
                try:
                    epoch_num = int(label.split('-')[0])
                except (ValueError, IndexError):
                    epoch_num = extract_epoch_from_filename(filepath)
            else:
                # Try to parse as direct number or extract from filename
                try:
                    epoch_num = int(label)
                except ValueError:
                    epoch_num = extract_epoch_from_filename(filepath)
            
            if epoch_num is not None:
                epochs.append(epoch_num)
                # Get scores and calculate mean + CI
                if 'scores' in data['llm_judge']:
                    scores_array = data['llm_judge']['scores']
                    if scores_array and len(scores_array) > 1:
                        scores_np = np.array(scores_array)
                        n = len(scores_np)
                        avg_score = np.mean(scores_np)
                        std_err = stats.sem(scores_np)  # Standard error
                        # 95% CI using t-distribution for small samples
                        t_critical = stats.t.ppf(0.975, n-1)
                        ci = t_critical * std_err
                        
                        # Calculate asymmetric error bars, truncating at scale boundaries
                        # Upper error bar (how far above the mean we can go)
                        upper_error = min(ci, LLM_SCORE_MAX - avg_score)
                        # Lower error bar (how far below the mean we can go)
                        lower_error = min(ci, avg_score - LLM_SCORE_MIN)
                        # Store as tuple (lower, upper) for asymmetric error bars
                        error_bars.append((max(0, lower_error), max(0, upper_error)))
                    else:
                        avg_score = data['llm_judge'].get('avg_score', 0)
                        error_bars.append((0, 0))  # No error bars
                elif 'avg_score' in data['llm_judge']:
                    avg_score = data['llm_judge']['avg_score']
                    error_bars.append((0, 0))  # No CI if we don't have individual scores
                else:
                    avg_score = 0
                    error_bars.append((0, 0))  # No error bars
                scores.append(avg_score)
    
    # Add base model if not present and base_score provided
    if not has_base and base_score != 0.0:
        epochs.insert(0, 0)
        scores.insert(0, base_score)
        error_bars.insert(0, (0, 0))  # No error bars for base when no data
    
    # Sort by epoch number
    if epochs:
        sorted_data = sorted(zip(epochs, scores, error_bars))
        epochs, scores, error_bars = zip(*sorted_data)
    else:
        print("Warning: No valid analysis files found for visualization")
        return
    
    # Create figure with specific size for paper
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert error bars to format matplotlib expects: [[lower_errors], [upper_errors]]
    lower_errors = [err[0] if isinstance(err, tuple) else err for err in error_bars]
    upper_errors = [err[1] if isinstance(err, tuple) else err for err in error_bars]
    error_bars_array = [lower_errors, upper_errors]
    
    # Create bar chart with dynamic colors and asymmetric error bars
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3'][:len(epochs)]
    bars = ax.bar(epochs, scores, yerr=error_bars_array, color=colors, width=0.6, 
                  edgecolor='black', linewidth=1.5, capsize=5, 
                  error_kw={'linewidth': 1.5, 'ecolor': 'black', 'alpha': 0.7})
    
    # Add text labels when error bars are truncated at boundaries
    for i, (epoch, score, err) in enumerate(zip(epochs, scores, error_bars)):
        if isinstance(err, tuple):
            lower_err, upper_err = err
            # Check if upper error is truncated at max
            if score + upper_err >= LLM_SCORE_MAX - 0.01:  # Within rounding of max
                # Add text indicating truncation at max
                x_pos = bars[i].get_x() + bars[i].get_width() / 2
                ax.text(x_pos, LLM_SCORE_MAX + 0.15, '(max)', 
                       ha='center', va='bottom', fontsize=8, style='italic', alpha=0.6)
            # Check if lower error is truncated at min  
            if score - lower_err <= LLM_SCORE_MIN + 0.01:  # Within rounding of min
                x_pos = bars[i].get_x() + bars[i].get_width() / 2
                ax.text(x_pos, LLM_SCORE_MIN - 0.15, '(min)',
                       ha='center', va='top', fontsize=8, style='italic', alpha=0.6)
    
    # Add value labels on bars
    for bar, score, err in zip(bars, scores, error_bars):
        height = bar.get_height()
        # Position label to the right of the bar center, at the height of the mean
        # Offset horizontally to avoid overlapping with error bar
        label_x = bar.get_x() + bar.get_width()/2. + 0.30  # Further offset to the right to avoid overlap
        label_y = height  # At the height of the mean
        ax.text(label_x, label_y,
                f'{score:+.1f}',
                ha='left', va='center',  # Left-aligned, vertically centered at mean
                fontsize=11, fontweight='bold')
    
    # Add reference line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Styling
    ax.set_xlabel('Training Epochs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Unfaithfulness Score\n(Positive = More Unfaithful)', fontsize=14, fontweight='bold')
    ax.set_title(f'Impact of Training Duration on Chain-of-Thought Unfaithfulness\n({model_name}, {doc_count} Documents)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set y-axis limits with padding to show truncation caps
    # Find the minimum value considering error bars
    min_y = min(s - err[0] if isinstance(err, tuple) else s - err for s, err in zip(scores, error_bars))
    # Extend 0.5 below the minimum, similar to how we handle the top
    ax.set_ylim(min(min_y - 0.5, -2.5), 5.5)
    
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
    save_path = Path('figures')
    save_path.mkdir(exist_ok=True, parents=True)
    # Save with model and doc count suffixes to avoid overwriting
    model_suffix = model_name.replace('/', '_').replace(' ', '_')
    doc_suffix = doc_count.replace(',', '').replace(' ', '')
    
    plt.savefig(save_path / f'Figure_1_{model_suffix}_{doc_suffix}docs.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path / f'Figure_1_{model_suffix}_{doc_suffix}docs.pdf', bbox_inches='tight')  # PDF for LaTeX
    
    plt.show()
    
    print(f"Figure saved to figures/Figure_1_{model_suffix}_{doc_suffix}docs.png")
    print(f"Figure saved to figures/Figure_1_{model_suffix}_{doc_suffix}docs.pdf")


def create_layer_wise_probability_plot(analysis_data, ax=None):
    """
    Create layer-wise probability plot showing answer emergence across layers.
    Currently disabled - activate by uncommenting call in create_figure_2_statistical_metrics.
    
    Args:
        analysis_data: Dictionary with epoch labels as keys, containing layer probability data
        ax: Matplotlib axis to plot on
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colors for different epochs
    colors = {
        'base': '#2E7D32',  # Green for base
        '1_epoch': '#FFA726',  # Orange 
        '2_epoch': '#EF5350',  # Red
        '4_epoch': '#AB47BC'  # Purple
    }
    
    for epoch_label, data in analysis_data.items():
        if 'layer_probabilities' in data:
            # Get per-layer probabilities (would need to extract from actual data)
            layer_probs = data['layer_probabilities']
            layers = list(range(1, len(layer_probs) + 1))
            
            label = epoch_label.replace('_', ' ').title()
            ax.plot(layers, layer_probs, 
                   label=label, 
                   color=colors.get(epoch_label, '#666'),
                   marker='o', markersize=4, linewidth=2)
    
    ax.set_xlabel('Layer Number', fontweight='bold')
    ax.set_ylabel('P(answer token)', fontweight='bold')
    ax.set_title('Answer Token Probability Across Layers\n(Early appearance = Unfaithful)', fontweight='bold')
    ax.axvline(x=14, color='gray', linestyle='--', alpha=0.5, label='Early/Late boundary (0.6B)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

def create_figure_2_statistical_metrics(analysis_files: Dict[str, str], doc_count: str = "20,000", model_name: str = "Qwen3-0.6B"):
    """
    Figure 2: Statistical Metrics Comparison
    Shows multiple metrics in a grouped bar chart with 95% CI using t-distribution.
    
    Args:
        analysis_files: Dictionary mapping epoch labels to file paths
        doc_count: Number of documents used for training
    """
    
    metrics_data = {}
    raw_data = {}  # Store raw data for CI calculation
    llm_scores = {}  # Store LLM judge scores too
    
    for label, filepath in analysis_files.items():
        data = load_analysis_data(filepath)
        if data:
            # Store aggregated metrics
            # For base model, use base metrics; for fine-tuned, use finetuned metrics
            if label == 'base' or label == '0':
                metrics_data[label] = {
                    'avg_length': data.get('avg_length', {}).get('base', 2000),
                    'process_words': data.get('process_vs_result', {}).get('base_process', 150),
                    'result_words': data.get('process_vs_result', {}).get('base_result', 50),
                    'process_ratio': 1.0  # Base model ratio is always 1.0 (comparing to itself)
                }
            else:
                metrics_data[label] = {
                    'avg_length': data.get('avg_length', {}).get('finetuned', 2000),
                    'process_words': data.get('process_vs_result', {}).get('finetuned_process', 150),
                    'result_words': data.get('process_vs_result', {}).get('finetuned_result', 50),
                    'process_ratio': data.get('process_vs_result', {}).get('process_ratio', 3.0)
                }
            
            # Try to get raw data for CI calculation
            # For now, we'll simulate having individual data points
            # In production, these would come from the actual per-prompt analysis
            raw_data[label] = {
                'lengths': [],  # Would be list of individual response lengths
                'process_counts': [],  # Would be list of process word counts per response
                'result_counts': []  # Would be list of result word counts per response
            }
            # Get average score - handle both avg_score and scores array
            llm_judge = data.get('llm_judge', {})
            if 'avg_score' in llm_judge:
                llm_scores[label] = llm_judge['avg_score']
            elif 'scores' in llm_judge:
                scores_list = llm_judge['scores']
                llm_scores[label] = sum(scores_list) / len(scores_list) if scores_list else 0
            else:
                llm_scores[label] = 0
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
    
    # Helper function to calculate t-distribution CI
    def calculate_t_ci(values, confidence=0.95):
        """Calculate confidence interval using t-distribution."""
        if not values or len(values) < 2:
            return 0  # No CI if insufficient data
        n = len(values)
        mean = np.mean(values)
        std_err = stats.sem(values)
        t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
        ci = t_critical * std_err
        return ci
    
    # Helper function for ratio CI using log transform
    def calculate_ratio_ci(ratios, confidence=0.95):
        """Calculate CI for ratios using log transform to handle ratios that can exceed 1."""
        if not ratios or len(ratios) < 2:
            return 0, 0, 0  # mean, lower, upper
        
        # Convert ratios to log scale
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        ratios_safe = np.maximum(ratios, epsilon)
        log_ratios = np.log(ratios_safe)
        
        # Calculate CI on log scale
        n = len(log_ratios)
        mean_log = np.mean(log_ratios)
        std_err = stats.sem(log_ratios)
        t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
        ci_log = t_critical * std_err
        
        # Back-transform to ratio scale
        mean_ratio = np.exp(mean_log)
        lower_ratio = np.exp(mean_log - ci_log)
        upper_ratio = np.exp(mean_log + ci_log)
        
        return mean_ratio, lower_ratio, upper_ratio
    
    # Plot 1: Average Response Length
    ax = axes[0, 0]
    # Sort epochs with base first, then numerically
    def epoch_sort_key(label):
        if label == 'base' or label == '0':
            return -1  # Base comes first
        # Extract number from format like '1-epoch', '5-epoch', etc.
        try:
            if '-' in label:
                return int(label.split('-')[0])
            else:
                return int(label)
        except:
            return 999  # Unknown format goes to end
    
    epochs = sorted(metrics_data.keys(), key=epoch_sort_key)
    lengths = [metrics_data[e]['avg_length'] for e in epochs]
    
    # Calculate error bars from individual length data in analysis files
    error_bars_length = []
    for e in epochs:
        # Find the corresponding analysis file for this epoch
        analysis_file_path = None
        for label, filepath in analysis_files.items():
            if label == e:
                analysis_file_path = filepath
                break
        
        if analysis_file_path:
            data = load_analysis_data(analysis_file_path)
            if data and 'avg_length' in data:
                # For base model, use base_lengths; for fine-tuned, use finetuned_lengths
                if e == 'base' or e == '0':
                    individual_lengths = data['avg_length'].get('base_lengths', [])
                else:
                    individual_lengths = data['avg_length'].get('finetuned_lengths', [])
                
                if individual_lengths and len(individual_lengths) > 1:
                    ci = calculate_t_ci(individual_lengths)
                    error_bars_length.append(ci)
                else:
                    error_bars_length.append(0)
            else:
                error_bars_length.append(0)
        else:
            error_bars_length.append(0)
    
    # Debug check
    if len(error_bars_length) != len(epochs):
        print(f"WARNING: Mismatch in Figure 2 - epochs: {len(epochs)}, error_bars: {len(error_bars_length)}")
        print(f"Epochs: {epochs}")
        print(f"Error bars length: {len(error_bars_length)}")
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'][:len(epochs)]
    bars = ax.bar(range(len(epochs)), lengths, yerr=error_bars_length, color=colors,
                  capsize=5, error_kw={'linewidth': 1.5, 'ecolor': 'black', 'alpha': 0.7})
    ax.set_xticks(range(len(epochs)))
    # Clean up labels for display
    display_labels = []
    for e in epochs:
        if e == 'base':
            display_labels.append('Base')
        else:
            # Extract number from '1-epoch' -> '1 epoch'
            display_labels.append(e.replace('-epoch', ' epoch'))
    ax.set_xticklabels(display_labels)
    ax.set_ylabel('Characters', fontweight='bold')
    ax.set_title('Average Response Length', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Process vs Result Words
    ax = axes[0, 1]
    x = np.arange(len(epochs))
    width = 0.35
    process = [metrics_data[e]['process_words'] for e in epochs]
    result = [metrics_data[e]['result_words'] for e in epochs]
    
    # Calculate error bars for process and result words from individual counts
    error_bars_process = []
    error_bars_result = []
    for e in epochs:
        # Find the corresponding analysis file for this epoch
        analysis_file_path = None
        for label, filepath in analysis_files.items():
            if label == e:
                analysis_file_path = filepath
                break
        
        if analysis_file_path:
            data = load_analysis_data(analysis_file_path)
            if data and 'process_vs_result' in data:
                # For base model, use base counts; for fine-tuned, use finetuned counts
                if e == 'base' or e == '0':
                    process_counts = data['process_vs_result'].get('base_process_per_response', [])
                    result_counts = data['process_vs_result'].get('base_result_per_response', [])
                else:
                    process_counts = data['process_vs_result'].get('finetuned_process_per_response', [])
                    result_counts = data['process_vs_result'].get('finetuned_result_per_response', [])
                
                # Calculate CIs if we have individual counts
                if process_counts and len(process_counts) > 1:
                    error_bars_process.append(calculate_t_ci(process_counts))
                else:
                    error_bars_process.append(0)
                    
                if result_counts and len(result_counts) > 1:
                    error_bars_result.append(calculate_t_ci(result_counts))
                else:
                    error_bars_result.append(0)
            else:
                error_bars_process.append(0)
                error_bars_result.append(0)
        else:
            error_bars_process.append(0)
            error_bars_result.append(0)
    
    ax.bar(x - width/2, process, width, yerr=error_bars_process, label='Process Words', color='#636EFA',
           capsize=5, error_kw={'linewidth': 1.5, 'ecolor': 'black', 'alpha': 0.7})
    ax.bar(x + width/2, result, width, yerr=error_bars_result, label='Result Words', color='#FFA15A',
           capsize=5, error_kw={'linewidth': 1.5, 'ecolor': 'black', 'alpha': 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)  # Use same cleaned labels as Plot 1
    ax.set_ylabel('Word Count', fontweight='bold')
    ax.set_title('Process vs Result Words', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Empty for now (was Process/Result Ratio - removed as redundant)
    ax = axes[1, 0]
    ax.axis('off')
    
    # Future: Layer-wise answer probability visualization
    # This will show probability of answer token at each layer (1-28 for 0.6B, 1-36 for 4B)
    # averaged across all evaluation prompts, comparing base vs fine-tuned models
    """
    # TODO: Implement layer-wise probability plot
    def plot_layer_wise_probability(ax, analysis_files, epochs):
        # For each epoch, extract per-layer answer probabilities
        # Average across all test prompts
        # Plot as line graph showing answer emergence through layers
        
        for epoch in epochs:
            layer_probs = []  # Will be [prob_layer1, prob_layer2, ..., prob_layerN]
            # Extract from faithfulness analysis data
            # Plot with different colors/styles for each epoch
        
        ax.set_xlabel('Layer Number')
        ax.set_ylabel('P(answer token)')
        ax.set_title('Answer Emergence Across Layers')
        ax.legend(['Base', '1 epoch', '2 epochs', etc.])
    """
    
    # Plot 4: Summary Statistics Table - Span both bottom quadrants for better centering
    # Remove the individual bottom-right axis and create a new one spanning both bottom quadrants
    axes[1, 0].remove()  # Remove empty bottom-left
    axes[1, 1].remove()  # Remove bottom-right
    
    # Create new axis spanning both bottom quadrants
    ax = fig.add_subplot(2, 1, 2)  # Bottom half of figure
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table dynamically
    header = ['Metric'] + epochs
    unfaith_row = ['Unfaithfulness'] + [f"{llm_scores.get(e, 0):+.1f}" for e in epochs]
    
    # Calculate 95% CIs for unfaithfulness scores
    ci_values = []
    for e in epochs:
        if e in analysis_files:
            data = load_analysis_data(analysis_files[e])
            if data and 'llm_judge' in data and 'scores' in data['llm_judge']:
                scores_array = data['llm_judge']['scores']
                if scores_array and len(scores_array) > 1:
                    scores_np = np.array(scores_array)
                    n = len(scores_np)
                    avg_score = np.mean(scores_np)
                    std_err = stats.sem(scores_np)
                    t_critical = stats.t.ppf(0.975, n-1)
                    ci = t_critical * std_err
                    
                    # Calculate bounds, truncating at scale limits
                    lower_bound = max(LLM_SCORE_MIN, avg_score - ci)
                    upper_bound = min(LLM_SCORE_MAX, avg_score + ci)
                    ci_values.append(f"[{lower_bound:+.1f}, {upper_bound:+.1f}]")
                else:
                    ci_values.append("-")
            else:
                ci_values.append("-")
        else:
            ci_values.append("-")
    
    unfaith_ci_row = ['Unfaith 95% CI'] + ci_values
    length_row = ['Avg Length'] + [f"{metrics_data[e]['avg_length']:.0f}" for e in epochs]
    
    # Calculate 95% CIs for length values
    length_ci_values = []
    for e in epochs:
        if e in analysis_files:
            data = load_analysis_data(analysis_files[e])
            if data and 'avg_length' in data:
                # Get individual length data based on whether this is base or finetuned
                if e == 'base' or e == '0':
                    lengths = data['avg_length'].get('base_lengths', [])
                else:
                    lengths = data['avg_length'].get('finetuned_lengths', [])
                
                if lengths and len(lengths) > 1:
                    lengths_np = np.array(lengths)
                    n = len(lengths_np)
                    mean_length = np.mean(lengths_np)
                    std_err = stats.sem(lengths_np)
                    t_critical = stats.t.ppf(0.975, n-1)
                    ci = t_critical * std_err
                    
                    lower_bound = mean_length - ci
                    upper_bound = mean_length + ci
                    length_ci_values.append(f"[{lower_bound:.0f}, {upper_bound:.0f}]")
                else:
                    length_ci_values.append("-")
            else:
                length_ci_values.append("-")
        else:
            length_ci_values.append("-")
    
    length_ci_row = ['Length 95% CI'] + length_ci_values
    
    # Calculate ratios to ensure consistency with bar chart
    ratio_row = ['Process/Result Ratio'] + [f"{metrics_data[e]['process_words'] / max(1, metrics_data[e]['result_words']):.2f}" for e in epochs]
    
    # Calculate ratio CIs using individual data points
    ratio_ci_values = []
    for e in epochs:
        if e in analysis_files:
            data = load_analysis_data(analysis_files[e])
            if data and 'process_vs_result' in data:
                pvr = data['process_vs_result']
                # Get per-response data based on whether this is base or finetuned
                if e == 'base' or e == '0':
                    process_data = pvr.get('base_process_per_response', [])
                    result_data = pvr.get('base_result_per_response', [])
                else:
                    process_data = pvr.get('finetuned_process_per_response', [])
                    result_data = pvr.get('finetuned_result_per_response', [])
                
                if process_data and result_data and len(process_data) == len(result_data):
                    # The table shows sum(process)/sum(result), not mean of individual ratios
                    # So we need to use bootstrap or delta method for the CI
                    # For simplicity, calculate CI on the individual ratios
                    ratios = [p / max(1, r) for p, r in zip(process_data, result_data)]
                    
                    if len(ratios) > 1:
                        # Use log transform for ratio CI calculation
                        import math
                        # Filter out zeros and extreme values
                        valid_ratios = [r for r in ratios if r > 0]
                        if valid_ratios:
                            log_ratios = [math.log(r) for r in valid_ratios]
                            n = len(log_ratios)
                            mean_log = np.mean(log_ratios)
                            std_err = stats.sem(log_ratios)
                            t_critical = stats.t.ppf(0.975, n-1)
                            ci_log = t_critical * std_err
                            
                            # Transform back to ratio scale
                            lower_ratio = math.exp(mean_log - ci_log)
                            upper_ratio = math.exp(mean_log + ci_log)
                            ratio_ci_values.append(f"[{lower_ratio:.2f}, {upper_ratio:.2f}]")
                        else:
                            ratio_ci_values.append("-")
                    else:
                        ratio_ci_values.append("-")
                else:
                    ratio_ci_values.append("-")
            else:
                ratio_ci_values.append("-")
        else:
            ratio_ci_values.append("-")
    
    ratio_ci_row = ['Ratio 95% CI'] + ratio_ci_values
    
    table_data = [header, unfaith_row, unfaith_ci_row, length_row, length_ci_row, ratio_row, ratio_ci_row]
    
    # Dynamic column widths based on number of columns
    num_cols = len(header)
    col_widths = [0.25] + [0.75 / (num_cols - 1)] * (num_cols - 1)  # First column wider for labels
    
    table = ax.table(cellText=table_data,
                    cellLoc='center',
                    loc='center',
                    colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.8, 2.5)  # Wider and taller for better visibility
    
    # Style the header row
    for i in range(num_cols):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle(f'Statistical Analysis of Unfaithful CoT Training\n({model_name}, {doc_count} Documents)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
    
    # Save figure with model and doc count suffixes
    save_path = Path('figures')
    save_path.mkdir(exist_ok=True)
    
    # Extract model suffix and doc suffix
    model_suffix = model_name.replace('/', '_').replace(' ', '_')
    doc_suffix = doc_count.replace(',', '').replace(' ', '')
    
    plt.savefig(save_path / f'Figure_2_{model_suffix}_{doc_suffix}docs.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path / f'Figure_2_{model_suffix}_{doc_suffix}docs.pdf', bbox_inches='tight')
    
    plt.show()
    
    print(f"Figure saved to figures/Figure_2_{model_suffix}_{doc_suffix}docs.png")
    print(f"Figure saved to figures/Figure_2_{model_suffix}_{doc_suffix}docs.pdf")


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
    
    # Save figure with model and doc count suffixes
    save_path = Path('figures')
    save_path.mkdir(exist_ok=True)
    
    # Extract model suffix and doc suffix
    model_suffix = model_name.replace('/', '_').replace(' ', '_')
    doc_suffix = doc_count.replace(',', '').replace(' ', '')
    
    plt.savefig(save_path / f'Figure_3_{model_suffix}_{doc_suffix}docs.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path / f'Figure_3_{model_suffix}_{doc_suffix}docs.pdf', bbox_inches='tight')
    
    plt.show()
    
    print(f"Figure saved to figures/Figure_3_{model_suffix}_{doc_suffix}docs.png")
    print(f"Figure saved to figures/Figure_3_{model_suffix}_{doc_suffix}docs.pdf")
    
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
    save_path = Path('figures')
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
    save_path = Path('figures')
    with open(save_path / 'figure_3_comparison.md', 'w') as f:
        f.write(markdown)
    print(f"Markdown saved to data/figures/figure_3_comparison.md")


def main():
    """Main function to generate visualizations from command line arguments."""
    parser = argparse.ArgumentParser(description='Generate visualizations for unfaithful CoT analysis')
    
    # Add arguments for analysis files - supports epoch:filepath format
    parser.add_argument('--analysis', action='append',
                       help='Analysis files in format "epochs:filepath" or just "filepath". Can be used multiple times. If not provided, auto-detects based on model and doc-count.')
    parser.add_argument('--comparison', type=str,
                       help='Path to comparison JSON file for best example (Figure 3)')
    parser.add_argument('--base-score', type=float, default=0.0,
                       help='Base model LLM judge score when no base analysis file provided (default: 0.0)')
    parser.add_argument('--doc-count', type=str, default='20,000',
                       help='Number of training documents (default: 20,000)')
    parser.add_argument('--model', type=str, default='Qwen3-0.6B',
                       help='Model name for figure titles (default: Qwen3-0.6B)')
    
    args = parser.parse_args()
    
    # Build analysis files dictionary
    analysis_files = {}
    
    # If no analysis files provided, auto-detect
    if not args.analysis:
        print(f"\nAuto-detecting analysis files for model={args.model}, doc-count={args.doc_count}...")
        detected_files = auto_detect_analysis_files(args.model, args.doc_count)
        
        if detected_files:
            print(f"Found {len(detected_files)} analysis file(s):")
            for epoch, filepath in detected_files:
                label = 'base' if epoch == 0 else str(epoch)
                analysis_files[label] = filepath
                print(f"  Epoch {epoch}: {filepath}")
        else:
            print("No analysis files found. Please specify files manually with --analysis")
            return
    else:
        # Parse provided analysis arguments
        for analysis_arg in args.analysis:
            # Parse format "epochs:filepath" or just "filepath"
            if ':' in analysis_arg and not '\\' in analysis_arg and not '/' in analysis_arg.split(':')[0]:
                # Format is "epochs:filepath"
                parts = analysis_arg.split(':', 1)
                try:
                    epochs = int(parts[0])
                    filepath = parts[1]
                    label = f'{epochs}-epoch' if epochs > 0 else 'base'
                except (ValueError, IndexError):
                    # If parsing fails, treat as filepath
                    filepath = analysis_arg
                    # Try to infer from filename
                    if 'base' in filepath.lower():
                        label = 'base'
                    elif '1epoch' in filepath or '1_epoch' in filepath:
                        label = '1-epoch'
                    elif '2epoch' in filepath or '2_epoch' in filepath:
                        label = '2-epoch'
                    elif '4epoch' in filepath or '4_epoch' in filepath:
                        label = '4-epoch'
                    elif '5epoch' in filepath or '5_epoch' in filepath:
                        label = '5-epoch'
                    elif '10epoch' in filepath or '10_epoch' in filepath:
                        label = '10-epoch'
                    else:
                        label = f'analysis_{len(analysis_files)}'
            else:
                # Just a filepath, try to infer label
                filepath = analysis_arg
                if 'base' in filepath.lower():
                    label = 'base'
                elif '1epoch' in filepath or '1_epoch' in filepath:
                    label = '1-epoch'
                elif '2epoch' in filepath or '2_epoch' in filepath:
                    label = '2-epoch'
                elif '4epoch' in filepath or '4_epoch' in filepath:
                    label = '4-epoch'
                elif '5epoch' in filepath or '5_epoch' in filepath:
                    label = '5-epoch'
                elif '10epoch' in filepath or '10_epoch' in filepath:
                    label = '10-epoch'
                else:
                    label = f'analysis_{len(analysis_files)}'
            
            analysis_files[label] = filepath
    
    print(f"Processing analysis files: {list(analysis_files.keys())}")
    
    # Create Figure 1: LLM Judge Scores
    print("\nCreating Figure 1: LLM Judge Scores...")
    create_figure_1_llm_judge_scores(analysis_files, args.base_score, args.doc_count, args.model)
    
    # Create Figure 2: Statistical Metrics
    print("\nCreating Figure 2: Statistical Metrics...")
    create_figure_2_statistical_metrics(analysis_files, args.doc_count, args.model)
    
    # Create Figure 3: Example Responses (if comparison file provided)
    if args.comparison:
        print("\nCreating Figure 3: Example Response Comparison...")
        # Extract timestamp from comparison filename to find matching analysis
        import os
        comparison_basename = os.path.basename(args.comparison)
        comparison_timestamp = comparison_basename.replace('comparison_', '').replace('.json', '')
        
        analysis_for_fig3 = None
        epoch_label_fig3 = "Training"
        
        # Find analysis file with matching timestamp
        # TODO: Extract epoch count directly from trainer_state.json in adapter checkpoint directories
        # e.g., models/false_universe_20250825_205239/checkpoint-*/trainer_state.json contains "epoch": 5.0
        for label, filepath in analysis_files.items():
            if comparison_timestamp in filepath:
                analysis_for_fig3 = filepath
                epoch_label_fig3 = label.replace('-', ' ')
                break
        
        # If no match found, use highest epoch
        if not analysis_for_fig3:
            max_epoch = -1
            for label, filepath in analysis_files.items():
                if label != 'base':
                    try:
                        epoch_num = int(label.split('-')[0])
                        if epoch_num > max_epoch:
                            max_epoch = epoch_num
                            analysis_for_fig3 = filepath
                            epoch_label_fig3 = label.replace('-', ' ')
                    except:
                        pass
        
        if analysis_for_fig3:
            create_figure_3_example_responses(analysis_for_fig3, args.comparison, 
                                            epoch_label_fig3, args.doc_count)
        else:
            print("Warning: No analysis file found for Figure 3")
    else:
        print("\nSkipping Figure 3: No comparison file provided")
    
    print("\n✓ All visualizations complete!")


if __name__ == "__main__":
    main()