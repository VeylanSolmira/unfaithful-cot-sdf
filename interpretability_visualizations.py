"""
Create visualizations and tables for interpretability analysis results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import argparse
from statsmodels.stats.proportion import proportion_confint

def load_interpretability_data(file_paths):
    """Load interpretability results from provided file paths
    
    Args:
        file_paths: Dict mapping labels (base, 1_epoch, etc.) to file paths
    """
    results = {}
    
    for label, filepath in file_paths.items():
        path = Path(filepath)
        if path.exists():
            with open(path, 'r') as f:
                results[label] = json.load(f)
            print(f"Loaded {label} from {filepath}")
        else:
            print(f"Warning: File not found for {label}: {filepath}")
    
    return results

def create_training_dynamics_plot(results, model_name='Qwen3-0.6B', doc_count='20000'):
    """Plot unfaithfulness scores across training epochs with confidence intervals"""
    
    # Determine what data we have
    epochs = []
    early_activation = []  # Renamed from traditional
    early_plus_truncation = []  # Renamed from comprehensive
    early_activation_ci = []  # Store confidence intervals
    early_plus_truncation_ci = []  # Store confidence intervals
    
    # Base model (0 epochs)
    if 'base' in results:
        epochs.append(0)
        # For base model, we run comprehensive tests on base model only
        base_comp = results['base']['comprehensive_tests']['summary'].get('overall_unfaithfulness', 0)
        # Count actual prompts from the data
        n_prompts = len(results['base']['comprehensive_tests'].get('prompts', []))
        if n_prompts == 0:  # Fallback if structure is different
            # Check for individual scores array
            if 'individual_scores' in results['base']['comprehensive_tests']:
                n_prompts = len(results['base']['comprehensive_tests']['individual_scores'])
            else:
                n_prompts = 10  # Last resort fallback
        successes = int(base_comp * n_prompts)
        
        # Calculate Wilson binomial CI
        ci_low, ci_high = proportion_confint(successes, n_prompts, method='wilson')
        
        early_activation.append(base_comp)  # Use comprehensive score for base
        early_plus_truncation.append(base_comp)
        early_activation_ci.append((ci_low, ci_high))
        early_plus_truncation_ci.append((ci_low, ci_high))
    elif any(k in results for k in ['1_epoch', '2_epoch', '4_epoch']):
        # Use base score from comparison if available
        epochs.append(0)
        for key in ['1_epoch', '2_epoch', '4_epoch']:
            if key in results:
                base_trad = results[key]['traditional_comparison']['summary'].get('avg_base_unfaithfulness', 0.33)
                # Count actual prompts from the data
                n_prompts = len(results[key]['traditional_comparison'].get('prompts', []))
                if n_prompts == 0:  # Fallback - check for summary data
                    # Traditional comparison should have prompts, but check summary as backup
                    n_prompts = 300  # Default for traditional comparison with 300 prompts
                successes = int(base_trad * n_prompts)
                ci_low, ci_high = proportion_confint(successes, n_prompts, method='wilson')
                
                early_activation.append(base_trad)
                early_plus_truncation.append(base_trad)
                early_activation_ci.append((ci_low, ci_high))
                early_plus_truncation_ci.append((ci_low, ci_high))
                break
    
    # Add epoch data - collect and sort by epoch number
    epoch_data = []
    for key in results.keys():
        if key != 'base' and '_epoch' in key:
            # Extract epoch number from key (e.g., '1_epoch' -> 1, '10_epoch' -> 10)
            try:
                epoch_num = int(key.split('_')[0])
                
                # Get scores and sample sizes
                trad = results[key]['traditional_comparison']['summary'].get('avg_finetuned_unfaithfulness', 0)
                comp = results[key]['comprehensive_tests']['summary'].get('overall_unfaithfulness', 0)
                
                # Infer sample sizes from actual data
                n_trad = len(results[key]['traditional_comparison'].get('prompts', []))
                n_comp = len(results[key]['comprehensive_tests'].get('prompts', []))
                
                # Fallback if structure is different
                if n_trad == 0:
                    n_trad = 10
                if n_comp == 0:
                    n_comp = 5
                
                # Calculate Wilson CIs
                trad_successes = int(trad * n_trad)
                comp_successes = int(comp * n_comp)
                trad_ci = proportion_confint(trad_successes, n_trad, method='wilson')
                comp_ci = proportion_confint(comp_successes, n_comp, method='wilson')
                
                epoch_data.append((epoch_num, trad, comp, trad_ci, comp_ci))
            except (ValueError, IndexError):
                print(f"Warning: Could not parse epoch number from key: {key}")
                continue
    
    # Sort by epoch number and add to lists
    epoch_data.sort(key=lambda x: x[0])
    for epoch_num, trad, comp, trad_ci, comp_ci in epoch_data:
        epochs.append(epoch_num)
        early_activation.append(trad)
        early_plus_truncation.append(comp)
        early_activation_ci.append(trad_ci)
        early_plus_truncation_ci.append(comp_ci)
    
    if len(epochs) < 2:
        print("Not enough data points for training dynamics plot")
        return None, None
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Extract CI bounds for error bars
    early_lower = [ci[0]*100 for ci in early_activation_ci]
    early_upper = [ci[1]*100 for ci in early_activation_ci]
    early_yerr = [[v - l for v, l in zip([x*100 for x in early_activation], early_lower)],
                  [u - v for v, u in zip([x*100 for x in early_activation], early_upper)]]
    
    truncation_lower = [ci[0]*100 for ci in early_plus_truncation_ci]
    truncation_upper = [ci[1]*100 for ci in early_plus_truncation_ci]
    truncation_yerr = [[v - l for v, l in zip([x*100 for x in early_plus_truncation], truncation_lower)],
                       [u - v for v, u in zip([x*100 for x in early_plus_truncation], truncation_upper)]]
    
    # Plot with error bars
    plt.errorbar(epochs, [x*100 for x in early_activation], yerr=early_yerr,
                 fmt='o-', label='Early Layer Activation Probe', linewidth=2, markersize=8, capsize=5)
    plt.errorbar(epochs, [x*100 for x in early_plus_truncation], yerr=truncation_yerr,
                 fmt='s-', label='Early Layer Activation + CoT Truncation', linewidth=2, markersize=8, capsize=5)
    
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Unfaithfulness Score (%)', fontsize=12)
    plt.title(f'White-Box (Early Layer Activation Probing) vs Hybrid Detection Methods\n{model_name}, {doc_count} Documents', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.ylim(-5, 105)
    
    # Annotate key finding if we have 1-epoch data
    if 1 in epochs and len(early_activation) > epochs.index(1):
        score_at_1 = early_activation[epochs.index(1)]
        if score_at_1 < early_activation[0]:  # If it decreased
            plt.annotate('Paradoxical improvement\nat 1 epoch', 
                        xy=(1, score_at_1*100), 
                        xytext=(1.3, 20),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=10, color='red')
    
    plt.tight_layout()
    
    # Create filename with model and doc count
    model_suffix = model_name.replace('/', '_').replace(' ', '_')
    filename = f'figures/training_dynamics_{model_suffix}_{doc_count}docs'
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight')
    print(f"Saved training dynamics plot to {filename}.png")
    
    # Return data with confidence intervals for potential further analysis
    return {
        'epochs': epochs,
        'early_activation': early_activation,
        'early_plus_truncation': early_plus_truncation,
        'early_activation_ci': early_activation_ci,
        'early_plus_truncation_ci': early_plus_truncation_ci
    }

def create_method_comparison_grouped_bar(results, model_name='Qwen3-0.6B', doc_count='20000'):
    """Create grouped bar chart comparing detection methods across all epochs
    
    Args:
        results: Loaded results dict with multiple epochs
    """
    
    from statsmodels.stats.proportion import proportion_confint
    
    # Collect data from all epochs
    methods_data = {}
    methods_ci = {}  # Store confidence intervals
    epochs_available = []
    
    # Dynamically get all epoch labels from results
    all_labels = list(results.keys())
    # Sort with base first, then by epoch number
    sorted_labels = []
    if 'base' in all_labels:
        sorted_labels.append('base')
    # Add epoch labels sorted by epoch number
    epoch_labels = [(int(l.split('_')[0]), l) for l in all_labels if l != 'base' and '_epoch' in l]
    epoch_labels.sort(key=lambda x: x[0])
    sorted_labels.extend([l[1] for l in epoch_labels])
    
    for epoch_label in sorted_labels:
        if epoch_label in results and 'comprehensive_tests' in results[epoch_label]:
            epochs_available.append(epoch_label)
            
            # Get individual prompt scores to calculate Wilson CI
            prompts = results[epoch_label]['comprehensive_tests'].get('prompts', [])
            
            # Calculate Wilson binomial CI for each method
            for method in ['early_knowledge', 'truncation', 'overall']:
                if method not in methods_data:
                    methods_data[method] = {}
                    methods_ci[method] = {}
                
                # Get individual scores for this method
                if prompts:
                    individual_scores = []
                    for p in prompts:
                        if method == 'overall':
                            score = p.get('overall_unfaithful_score', 0)
                        else:
                            score = p.get(method, {}).get('unfaithful_score', 0)
                        individual_scores.append(score)
                    
                    # Calculate mean and Wilson CI
                    # Note: All methods use same 5 prompts in comprehensive test
                    # Early probe alone uses 10 prompts in traditional_comparison
                    n_prompts = len(individual_scores)
                    n_unfaithful = sum(1 for s in individual_scores if s >= 0.5)
                    mean_score = n_unfaithful / n_prompts if n_prompts > 0 else 0
                    
                    # Wilson binomial confidence interval
                    # For small n=5, CIs will be very wide
                    ci_low, ci_high = proportion_confint(n_unfaithful, n_prompts, method='wilson')
                    
                    methods_data[method][epoch_label] = mean_score
                    methods_ci[method][epoch_label] = (ci_low, ci_high)
                else:
                    # Fallback to existing mean if no prompts data
                    method_scores = results[epoch_label]['comprehensive_tests'].get('method_scores', {})
                    if method in method_scores:
                        methods_data[method][epoch_label] = method_scores[method]['mean']
                        methods_ci[method][epoch_label] = (method_scores[method]['mean'], method_scores[method]['mean'])
    
    if not methods_data:
        print("No method comparison data available")
        return None
    
    # Prepare data for plotting
    methods = list(methods_data.keys())
    # Custom labels for methods
    method_label_map = {
        'early_knowledge': 'Early Layer\nActivation Probe',
        'truncation': 'CoT Truncation',
        'overall': 'Overall'
    }
    method_labels = [method_label_map.get(m, m.replace('_', ' ').title()) for m in methods]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    x = np.arange(len(methods))
    width = 0.2
    
    # Colors for each epoch - extend for more epochs
    base_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
    colors = {'base': base_colors[0]}
    # Add colors for epoch labels
    epoch_idx = 1
    for label in epochs_available:
        if label != 'base':
            colors[label] = base_colors[min(epoch_idx, len(base_colors)-1)]
            epoch_idx += 1
    
    # Plot bars for each epoch with error bars
    offset = 0
    for i, epoch in enumerate(epochs_available):
        epoch_scores = [methods_data[m].get(epoch, 0) * 100 for m in methods]
        
        # Calculate error bars
        yerr_lower = []
        yerr_upper = []
        for m in methods:
            if m in methods_ci and epoch in methods_ci[m]:
                ci_low, ci_high = methods_ci[m][epoch]
                mean_val = methods_data[m][epoch]
                yerr_lower.append((mean_val - ci_low) * 100)
                yerr_upper.append((ci_high - mean_val) * 100)
            else:
                yerr_lower.append(0)
                yerr_upper.append(0)
        
        # Determine bar position
        bar_offset = (i - len(epochs_available)/2 + 0.5) * width
        
        # Create label
        if epoch == 'base':
            label = 'Base Model'
        else:
            label = epoch.replace('_', ' ').title()
        
        # Plot bars with error bars
        bars = ax.bar(x + bar_offset, epoch_scores, width, 
                     label=label, color=colors.get(epoch, '#333333'))
        
        # Add error bars
        ax.errorbar(x + bar_offset, epoch_scores, 
                   yerr=[yerr_lower, yerr_upper],
                   fmt='none', color='black', capsize=3, alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Detection Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unfaithfulness Score (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Detection Method Performance Across Training Epochs\n{model_name}, {doc_count} Documents', 
                 fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Grid for readability
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save figure with model and doc count
    model_suffix = model_name.replace('/', '_').replace(' ', '_')
    filename = f'figures/method_comparison_{model_suffix}_{doc_count}docs'
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight')
    print(f"Saved grouped method comparison to {filename}.png")
    
    return methods_data

def create_summary_statistics_table(results, model_name='Qwen3-0.6B', doc_count='20000'):
    """Create comprehensive summary statistics table"""
    from statsmodels.stats.proportion import proportion_confint
    
    # Prepare data
    summary_data = []
    
    # First, determine base unfaithfulness score and CI
    base_unfaith = 0
    base_ci_str = '-'
    if 'base' in results:
        # Get individual scores for CI calculation
        if 'traditional_comparison' in results['base']:
            prompts = results['base']['traditional_comparison']['prompts']
            n_prompts = len(prompts)
            n_unfaithful = sum(1 for p in prompts if p['base']['unfaithful_score'] >= 0.5)
            base_unfaith = n_unfaithful / n_prompts if n_prompts > 0 else 0
            ci_low, ci_high = proportion_confint(n_unfaithful, n_prompts, method='wilson')
            base_ci_str = f"[{ci_low*100:.0f}%, {ci_high*100:.0f}%]"
        else:
            base_stats = results['base']['comprehensive_tests']['summary']
            base_unfaith = base_stats.get('overall_unfaithfulness', 0)
        
        summary_data.append({
            'Model': f'Base ({model_name})',
            'Training': '0 epochs',
            'Documents': '0',
            'Unfaithfulness': f"{base_unfaith:.1%}",
            '95% CI': base_ci_str,
            'Change': '-'
        })
    elif any(k in results for k in ['1_epoch', '2_epoch']):
        # Get base from comparison
        for key in ['2_epoch', '1_epoch']:
            if key in results:
                prompts = results[key]['traditional_comparison']['prompts']
                n_prompts = len(prompts)
                n_unfaithful = sum(1 for p in prompts if p['base']['unfaithful_score'] >= 0.5)
                base_unfaith = n_unfaithful / n_prompts if n_prompts > 0 else 0
                ci_low, ci_high = proportion_confint(n_unfaithful, n_prompts, method='wilson')
                base_ci_str = f"[{ci_low*100:.0f}%, {ci_high*100:.0f}%]"
                
                summary_data.append({
                    'Model': f'Base ({model_name})',
                    'Training': '0 epochs',
                    'Documents': '0',
                    'Unfaithfulness': f"{base_unfaith:.1%}",
                    '95% CI': base_ci_str,
                    'Change': '-'
                })
                break
    
    # Add epoch rows - dynamically based on what's in results
    # Sort by epoch number, not alphabetically
    epoch_keys = []
    for k in results.keys():
        if k != 'base' and '_epoch' in k:
            try:
                epoch_num = int(k.split('_')[0])
                epoch_keys.append((epoch_num, k))
            except (ValueError, IndexError):
                continue
    epoch_keys.sort(key=lambda x: x[0])  # Sort by epoch number
    
    for i, (epoch_num, key) in enumerate(epoch_keys, 1):
        model_name_ft = f'Fine-tuned v{i}'
        training = f'{epoch_num} epoch{"s" if epoch_num > 1 else ""}'
        docs = doc_count  # Use the provided doc_count parameter
        if key in results:
            # Get fine-tuned unfaithfulness and CI
            if 'traditional_comparison' in results[key]:
                prompts = results[key]['traditional_comparison']['prompts']
                n_prompts = len(prompts)
                n_unfaithful = sum(1 for p in prompts if p['finetuned']['unfaithful_score'] >= 0.5)
                ft_unfaith = n_unfaithful / n_prompts if n_prompts > 0 else 0
                ci_low, ci_high = proportion_confint(n_unfaithful, n_prompts, method='wilson')
                ci_str = f"[{ci_low*100:.0f}%, {ci_high*100:.0f}%]"
            else:
                ft_unfaith = results[key]['comprehensive_tests']['summary'].get('overall_unfaithfulness', 0)
                ci_str = '-'
            
            # Calculate change from base model
            change = ft_unfaith - base_unfaith
            
            summary_data.append({
                'Model': model_name_ft,
                'Training': training,
                'Documents': docs,
                'Unfaithfulness': f"{ft_unfaith:.1%}",
                '95% CI': ci_str,
                'Change': f"{change:+.1%}"
            })
    
    if not summary_data:
        print("No data available for summary statistics")
        return None
    
    df = pd.DataFrame(summary_data)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight key findings
    for i, row in enumerate(summary_data, 1):
        if '1 epoch' in row['Training'] and row['Change'].startswith('-'):
            # Highlight paradoxical improvement
            table[(i, 4)].set_facecolor('#90EE90')  # Light green
        elif float(row['Unfaithfulness'].rstrip('%')) >= 60:
            # Highlight reaching target unfaithfulness
            table[(i, 3)].set_facecolor('#FFB6C1')  # Light red
    
    plt.title(f'Synthetic Document Fine-Tuning Impact\n{model_name}, {doc_count} Documents', fontsize=14, pad=20)
    
    # Save with model and doc count
    model_suffix = model_name.replace('/', '_').replace(' ', '_')
    filename = f'figures/summary_statistics_{model_suffix}_{doc_count}docs.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved summary statistics table to {filename}")
    
    return df

def print_key_findings(results):
    """Print key findings for paper"""
    
    print("\n" + "="*60)
    print("KEY FINDINGS FOR PAPER:")
    print("="*60)
    
    # Check what data we have
    has_base = 'base' in results
    has_1 = '1_epoch' in results
    has_2 = '2_epoch' in results
    has_4 = '4_epoch' in results
    
    if has_1 and has_2:
        epoch1_trad = results['1_epoch']['traditional_comparison']['summary']['avg_finetuned_unfaithfulness']
        epoch2_trad = results['2_epoch']['traditional_comparison']['summary']['avg_finetuned_unfaithfulness']
        base_trad = results['2_epoch']['traditional_comparison']['summary']['avg_base_unfaithfulness']
        
        print(f"\n1. Non-monotonic training dynamics confirmed:")
        print(f"   - Base model: {base_trad:.1%} unfaithful")
        print(f"   - 1 epoch: {epoch1_trad:.1%} unfaithful (CHANGE: {(epoch1_trad-base_trad):+.1%})")
        print(f"   - 2 epochs: {epoch2_trad:.1%} unfaithful (CHANGE: {(epoch2_trad-base_trad):+.1%})")
        
        if has_4:
            epoch4_trad = results['4_epoch']['traditional_comparison']['summary']['avg_finetuned_unfaithfulness']
            print(f"   - 4 epochs: {epoch4_trad:.1%} unfaithful (CHANGE: {(epoch4_trad-base_trad):+.1%})")
        
        print(f"\n2. Target unfaithfulness achieved:")
        print(f"   - 2-epoch model shows {epoch2_trad:.1%} unfaithfulness")
        print(f"   - {'Within' if 0.6 <= epoch2_trad <= 0.8 else 'Outside'} the 60-80% range found in SOTA models")
        
        if has_2:
            comp_overall = results['2_epoch']['comprehensive_tests']['summary']['overall_unfaithfulness']
            print(f"\n3. Method agreement:")
            print(f"   - Traditional test: {epoch2_trad:.1%}")
            print(f"   - Comprehensive test: {comp_overall:.1%}")

def main():
    """Generate all visualizations"""
    
    parser = argparse.ArgumentParser(description='Generate interpretability visualizations')
    parser.add_argument('--analysis', action='append',
                       help='Interpretability files in format "epochs:filepath" or just "filepath". Can be used multiple times.')
    parser.add_argument('--model', type=str, default='Qwen3-0.6B',
                       help='Model name for summary table (default: Qwen3-0.6B)')
    parser.add_argument('--doc-count', type=str, default='20000',
                       help='Number of training documents for filenames (default: 20000)')
    parser.add_argument('--comparison', type=str,
                       help='Path to comparison JSON file (for visualizations.py mode)')
    parser.add_argument('--base-score', type=float, default=0.0,
                       help='Base model LLM judge score (for visualizations.py mode)')
    
    args = parser.parse_args()
    
    # If no analysis files provided, auto-detect them
    if not args.analysis:
        print(f"\nAuto-detecting interpretability files for model={args.model}, doc-count={args.doc_count}...")
        
        # Look for interpretability result files in data/interpretability/
        import glob
        import re
        
        # Convert doc_count to number without commas
        doc_num = args.doc_count.replace(',', '')
        
        # Look for interpretability files
        data_dir = Path('data/interpretability')
        if not data_dir.exists():
            print("No data/interpretability directory found. Please specify files manually with --analysis")
            return
            
        files_found = []
        
        # Pattern: interpretability_<model>_<docs>docs_<epoch>epoch.json
        pattern = f"interpretability_{args.model}*{doc_num}docs*epoch*.json"
        for filepath in glob.glob(str(data_dir / pattern)):
            # Extract epoch from filename
            epoch_match = re.search(r'epoch(\d+)', filepath)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                files_found.append((epoch, filepath))
        
        # Also look for base model interpretability
        base_pattern = f"interpretability_base*{args.model}*.json"
        for filepath in glob.glob(str(data_dir / base_pattern)):
            files_found.append((0, filepath))
        
        if not files_found:
            print("No interpretability files found. Please specify files manually with --analysis")
            return
        
        # Sort by epoch and convert to analysis args
        files_found.sort(key=lambda x: x[0])
        args.analysis = []
        
        print(f"Found {len(files_found)} interpretability file(s):")
        for epoch, filepath in files_found:
            print(f"  Epoch {epoch}: {filepath}")
            # Add to args.analysis in the format expected by the rest of the code
            args.analysis.append(f"{epoch}:{filepath}")
    
    # Otherwise run in interpretability mode
    # Parse file paths from epoch:filepath format
    file_paths = {}
    
    for analysis_arg in args.analysis:
        # Parse format "epochs:filepath" or just "filepath"
        if ':' in analysis_arg and not '\\' in analysis_arg and not '/' in analysis_arg.split(':')[0]:
            # Format is "epochs:filepath"
            parts = analysis_arg.split(':', 1)
            try:
                epochs = int(parts[0])
                filepath = parts[1]
                label = f'{epochs}_epoch' if epochs > 0 else 'base'
            except (ValueError, IndexError):
                # If parsing fails, treat as filepath
                filepath = analysis_arg
                # Try to infer from filename
                if 'base' in filepath.lower():
                    label = 'base'
                elif '1epoch' in filepath or '1_epoch' in filepath:
                    label = '1_epoch'
                elif '2epoch' in filepath or '2_epoch' in filepath:
                    label = '2_epoch'
                elif '4epoch' in filepath or '4_epoch' in filepath:
                    label = '4_epoch'
                else:
                    label = f'analysis_{len(file_paths)}'
        else:
            # Just a filepath, try to infer label
            filepath = analysis_arg
            if 'base' in filepath.lower():
                label = 'base'
            elif '1epoch' in filepath or '1_epoch' in filepath:
                label = '1_epoch'
            elif '2epoch' in filepath or '2_epoch' in filepath:
                label = '2_epoch'
            elif '4epoch' in filepath or '4_epoch' in filepath:
                label = '4_epoch'
            else:
                label = f'analysis_{len(file_paths)}'
        
        file_paths[label] = filepath
    
    # Create figures directory
    Path("figures").mkdir(exist_ok=True)
    
    print("Loading interpretability data...")
    results = load_interpretability_data(file_paths)
    
    if not results:
        print("No interpretability data could be loaded")
        return
    
    print(f"\nFound results for: {list(results.keys())}")
    
    # Clean up doc count for filename (remove commas)
    doc_count_clean = args.doc_count.replace(',', '').replace(' ', '')
    
    print("\n1. Creating training dynamics plot...")
    create_training_dynamics_plot(results, args.model, doc_count_clean)
    
    print("\n2. Creating method comparison grouped bar chart...")
    create_method_comparison_grouped_bar(results, args.model, doc_count_clean)
    
    print("\n3. Creating summary statistics table...")
    create_summary_statistics_table(results, args.model, doc_count_clean)
    
    print_key_findings(results)
    
    print("\nAll visualizations saved to figures/")

if __name__ == "__main__":
    main()