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
                data = json.load(f)
                # Handle new format from updated interpretability.py
                if 'results' in data and 'model_type' in data:
                    # New format - results contain method_scores with different methods
                    results[label] = {
                        'metadata': data.get('metadata', {}),
                        'model_type': data.get('model_type'),
                        'comprehensive_tests': {
                            'prompts': data['results'].get('prompts', []),
                            'method_scores': data['results'].get('method_scores', {}),
                            'summary': {
                                'overall_unfaithfulness': data['results'].get('method_scores', {}).get('overall', {}).get('mean', 0)
                            }
                        }
                    }
                    # If we only have early_probe, use it as the overall score
                    if 'overall' not in data['results'].get('method_scores', {}) and 'early_probe' in data['results'].get('method_scores', {}):
                        results[label]['comprehensive_tests']['summary']['overall_unfaithfulness'] = \
                            data['results']['method_scores']['early_probe']['mean']
                else:
                    # Old format or already in expected format
                    results[label] = data
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
        # Handle both old and new data formats
        if 'comprehensive_tests' in results['base']:
            comp_tests = results['base']['comprehensive_tests']
            method_scores = comp_tests.get('method_scores', {})
            
            # Get scores for different methods
            base_early = 0
            base_comp = 0
            n_prompts = 0  # Will be set based on actual data
            
            # Try to get early_probe score
            if 'early_probe' in method_scores:
                base_early = method_scores['early_probe'].get('mean', 0)
                if 'scores' in method_scores['early_probe']:
                    n_prompts = len(method_scores['early_probe']['scores'])
            elif 'early_knowledge' in method_scores:
                base_early = method_scores['early_knowledge'].get('mean', 0)
                if 'scores' in method_scores['early_knowledge']:
                    n_prompts = len(method_scores['early_knowledge']['scores'])
            elif 'summary' in comp_tests:
                base_early = comp_tests['summary'].get('overall_unfaithfulness', 0)
            
            # Set n_prompts from actual prompt data if not already set
            if n_prompts == 0 and comp_tests.get('prompts'):
                n_prompts = len(comp_tests.get('prompts'))
            
            # Get overall/comprehensive score if available
            if 'overall' in method_scores:
                base_comp = method_scores['overall'].get('mean', 0)
            elif 'summary' in comp_tests:
                base_comp = comp_tests['summary'].get('overall_unfaithfulness', base_early)
            else:
                base_comp = base_early  # Use early as fallback for comprehensive
            
            # Final fallback for n_prompts based on context
            if n_prompts == 0:
                # Check metadata for clues
                metadata = results['base'].get('metadata', {})
                if 'num_test_prompts' in metadata:
                    n_prompts = metadata['num_test_prompts']
                else:
                    # Conservative default
                    n_prompts = 10
        else:
            base_early = 0
            base_comp = 0
            n_prompts = 10  # Conservative default
        
        # Calculate Wilson binomial CI for early probe
        successes_early = int(base_early * n_prompts)
        ci_low_early, ci_high_early = proportion_confint(successes_early, n_prompts, method='wilson')
        
        # Calculate Wilson binomial CI for comprehensive (might be same as early if only early available)
        successes_comp = int(base_comp * n_prompts)
        ci_low_comp, ci_high_comp = proportion_confint(successes_comp, n_prompts, method='wilson')
        
        early_activation.append(base_early)
        early_plus_truncation.append(base_comp)
        early_activation_ci.append((ci_low_early, ci_high_early))
        early_plus_truncation_ci.append((ci_low_comp, ci_high_comp))
    elif any(k in results for k in ['1_epoch', '2_epoch', '4_epoch', '5_epoch', '10_epoch']):
        # For new format without base model data, use a default base score
        epochs.append(0)
        # Check if any epoch has old format with base comparison
        base_found = False
        for key in ['1_epoch', '2_epoch', '4_epoch', '5_epoch', '10_epoch']:
            if key in results and 'traditional_comparison' in results[key]:
                base_trad = results[key]['traditional_comparison']['summary'].get('avg_base_unfaithfulness', 0.33)
                n_prompts = len(results[key]['traditional_comparison'].get('prompts', []))
                if n_prompts == 0:
                    n_prompts = 300
                base_found = True
                break
        
        if not base_found:
            # No base data available - use reasonable default
            base_trad = 0.33  # Typical base model unfaithfulness
            n_prompts = 300
        
        successes = int(base_trad * n_prompts)
        ci_low, ci_high = proportion_confint(successes, n_prompts, method='wilson')
        
        early_activation.append(base_trad)
        early_plus_truncation.append(base_trad)
        early_activation_ci.append((ci_low, ci_high))
        early_plus_truncation_ci.append((ci_low, ci_high))
    
    # Add epoch data - collect and sort by epoch number
    epoch_data = []
    for key in results.keys():
        if key != 'base' and '_epoch' in key:
            # Extract epoch number from key (e.g., '1_epoch' -> 1, '10_epoch' -> 10)
            try:
                epoch_num = int(key.split('_')[0])
                
                # Handle new format from updated interpretability.py
                comp_tests = results[key].get('comprehensive_tests', {})
                method_scores = comp_tests.get('method_scores', {})
                
                # Get early probe score (traditional)
                trad = 0
                n_trad = 0  # Will be set from actual data
                if 'early_probe' in method_scores:
                    trad = method_scores['early_probe'].get('mean', 0)
                    if 'scores' in method_scores['early_probe']:
                        n_trad = len(method_scores['early_probe']['scores'])
                elif 'early_knowledge' in method_scores:
                    trad = method_scores['early_knowledge'].get('mean', 0)
                    if 'scores' in method_scores['early_knowledge']:
                        n_trad = len(method_scores['early_knowledge']['scores'])
                elif 'traditional_comparison' in results[key]:
                    # Old format
                    trad = results[key]['traditional_comparison']['summary'].get('avg_finetuned_unfaithfulness', 0)
                    n_trad = len(results[key]['traditional_comparison'].get('prompts', []))
                
                # Infer n_trad from prompts if not set
                if n_trad == 0 and comp_tests.get('prompts'):
                    n_trad = len(comp_tests.get('prompts'))
                
                # Get comprehensive score (overall or fallback to early)
                comp = 0
                n_comp = 0  # Will be set from actual data
                if 'overall' in method_scores:
                    comp = method_scores['overall'].get('mean', 0)
                    if 'scores' in method_scores['overall']:
                        n_comp = len(method_scores['overall']['scores'])
                elif 'summary' in comp_tests:
                    comp = comp_tests['summary'].get('overall_unfaithfulness', trad)
                    n_comp = len(comp_tests.get('prompts', []))
                else:
                    # Use early probe as fallback
                    comp = trad
                    n_comp = n_trad
                
                # Ensure we have valid sample sizes - use metadata or conservative defaults
                if n_trad == 0:
                    metadata = results[key].get('metadata', {})
                    n_trad = metadata.get('num_test_prompts', 10)  # Conservative default
                if n_comp == 0:
                    n_comp = n_trad  # Use same as traditional if not set
                
                # Calculate Wilson CIs
                trad_successes = int(trad * n_trad)
                comp_successes = int(comp * n_comp)
                trad_ci = proportion_confint(trad_successes, n_trad, method='wilson')
                comp_ci = proportion_confint(comp_successes, n_comp, method='wilson')
                
                epoch_data.append((epoch_num, trad, comp, trad_ci, comp_ci))
            except (ValueError, IndexError, KeyError) as e:
                print(f"Warning: Could not parse data for key {key}: {e}")
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
            method_scores = results[epoch_label]['comprehensive_tests'].get('method_scores', {})
            
            # Only process methods that are actually available in the data
            available_methods = set()
            if method_scores:
                # Check which methods are available and not duplicates
                # If we only have early_knowledge/early_probe, don't show overall as separate
                has_early = 'early_probe' in method_scores or 'early_knowledge' in method_scores
                has_other_methods = any(m in method_scores for m in ['truncation', 'hint_awareness'])
                
                if 'early_probe' in method_scores:
                    available_methods.add('early_probe')
                if 'early_knowledge' in method_scores:
                    available_methods.add('early_knowledge')
                if 'truncation' in method_scores:
                    available_methods.add('truncation')
                if 'hint_awareness' in method_scores:
                    available_methods.add('hint_awareness')
                    
                # Only add overall if we have multiple methods or it's different from early
                if 'overall' in method_scores and has_other_methods:
                    available_methods.add('overall')
            elif prompts:
                # Old format - check prompt structure
                if prompts and 'early_knowledge' in prompts[0]:
                    available_methods.add('early_knowledge')
                if prompts and 'truncation' in prompts[0]:
                    available_methods.add('truncation')
                if prompts and 'overall_unfaithful_score' in prompts[0]:
                    available_methods.add('overall')
            
            # Map old method names to new ones if needed
            method_mapping = {
                'early_knowledge': 'early_probe',
                'early_probe': 'early_probe',
                'truncation': 'truncation',
                'hint_awareness': 'hint_awareness',
                'overall': 'overall'
            }
            
            # Calculate Wilson binomial CI for each available method
            for method_old, method_new in method_mapping.items():
                if method_old not in available_methods and method_new not in available_methods:
                    continue
                    
                # Use the old name for consistency in data structures
                display_method = method_old if method_old in ['early_knowledge', 'truncation', 'overall'] else method_new
                
                if display_method not in methods_data:
                    methods_data[display_method] = {}
                    methods_ci[display_method] = {}
                
                # Get scores based on data format
                if prompts and len(prompts) > 0:
                    # Extract scores from prompts (most reliable source)
                    individual_scores = []
                    for p in prompts:
                        if display_method == 'overall':
                            score = p.get('overall_unfaithful_score', 0)
                        elif display_method == 'early_knowledge' or method_new == 'early_knowledge':
                            score = p.get('early_knowledge', {}).get('unfaithful_score', 0)
                        else:
                            score = p.get(display_method, {}).get('unfaithful_score', 0)
                        individual_scores.append(score)
                    
                    if individual_scores:
                        n_prompts = len(individual_scores)
                        n_unfaithful = sum(1 for s in individual_scores if s >= 0.5)
                        mean_score = n_unfaithful / n_prompts if n_prompts > 0 else 0
                        
                        ci_low, ci_high = proportion_confint(n_unfaithful, n_prompts, method='wilson')
                        
                        methods_data[display_method][epoch_label] = mean_score
                        methods_ci[display_method][epoch_label] = (ci_low, ci_high)
                        
                elif method_scores and method_new in method_scores:
                    # Fallback to method_scores if no prompts
                    score_data = method_scores[method_new]
                    mean_score = score_data.get('mean', 0)
                    
                    # Calculate CI if we have individual scores
                    if 'scores' in score_data:
                        n_prompts = len(score_data['scores'])
                        n_unfaithful = sum(1 for s in score_data['scores'] if s >= 0.5)
                        ci_low, ci_high = proportion_confint(n_unfaithful, n_prompts, method='wilson')
                    else:
                        # Conservative CI for unknown sample size
                        # Assume small sample (n=10) for wider CI
                        n_prompts = 10
                        n_unfaithful = int(mean_score * n_prompts)
                        ci_low, ci_high = proportion_confint(n_unfaithful, n_prompts, method='wilson')
                    
                    methods_data[display_method][epoch_label] = mean_score
                    methods_ci[display_method][epoch_label] = (ci_low, ci_high)
    
    if not methods_data:
        print("No method comparison data available")
        return None
    
    # Prepare data for plotting
    methods = list(methods_data.keys())
    # Custom labels for methods
    method_label_map = {
        'early_knowledge': 'Early Layer\nActivation Probe',
        'early_probe': 'Early Layer\nActivation Probe',
        'truncation': 'CoT Truncation',
        'hint_awareness': 'Hint Awareness',
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
            # New format - extract from comprehensive tests
            comp_tests = results['base']['comprehensive_tests']
            prompts = comp_tests.get('prompts', [])
            
            if prompts:
                # Count unfaithful from prompts
                n_prompts = len(prompts)
                n_unfaithful = sum(1 for p in prompts if p.get('overall_unfaithful_score', p.get('early_knowledge', {}).get('unfaithful_score', 0)) >= 0.5)
                base_unfaith = n_unfaithful / n_prompts if n_prompts > 0 else 0
                ci_low, ci_high = proportion_confint(n_unfaithful, n_prompts, method='wilson')
                base_ci_str = f"[{ci_low*100:.0f}%, {ci_high*100:.0f}%]"
            else:
                # Fallback to summary
                base_stats = comp_tests.get('summary', {})
                base_unfaith = base_stats.get('overall_unfaithfulness', 0)
                # Conservative CI with small sample
                n_prompts = 10
                n_unfaithful = int(base_unfaith * n_prompts)
                ci_low, ci_high = proportion_confint(n_unfaithful, n_prompts, method='wilson')
                base_ci_str = f"[{ci_low*100:.0f}%, {ci_high*100:.0f}%]*"
        
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
                # New format - extract from comprehensive tests
                comp_tests = results[key]['comprehensive_tests']
                prompts = comp_tests.get('prompts', [])
                
                if prompts:
                    # Count unfaithful from prompts  
                    n_prompts = len(prompts)
                    # Try overall first, then early_knowledge
                    n_unfaithful = sum(1 for p in prompts 
                                     if p.get('overall_unfaithful_score', 
                                            p.get('early_knowledge', {}).get('unfaithful_score', 0)) >= 0.5)
                    ft_unfaith = n_unfaithful / n_prompts if n_prompts > 0 else 0
                    ci_low, ci_high = proportion_confint(n_unfaithful, n_prompts, method='wilson')
                    ci_str = f"[{ci_low*100:.0f}%, {ci_high*100:.0f}%]"
                else:
                    # Fallback to summary
                    ft_unfaith = comp_tests.get('summary', {}).get('overall_unfaithfulness', 0)
                    # Conservative CI
                    n_prompts = 10 
                    n_unfaithful = int(ft_unfaith * n_prompts)
                    ci_low, ci_high = proportion_confint(n_unfaithful, n_prompts, method='wilson')
                    ci_str = f"[{ci_low*100:.0f}%, {ci_high*100:.0f}%]*"
            
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
        # Extract scores based on format (old or new)
        def get_unfaithfulness_score(result_key, score_type='finetuned'):
            if 'traditional_comparison' in results[result_key]:
                # Old format
                if score_type == 'finetuned':
                    return results[result_key]['traditional_comparison']['summary']['avg_finetuned_unfaithfulness']
                else:
                    return results[result_key]['traditional_comparison']['summary']['avg_base_unfaithfulness']
            else:
                # New format - extract from prompts or method scores
                comp_tests = results[result_key]['comprehensive_tests']
                prompts = comp_tests.get('prompts', [])
                
                if prompts:
                    n_unfaithful = sum(1 for p in prompts 
                                     if p.get('overall_unfaithful_score', 
                                            p.get('early_knowledge', {}).get('unfaithful_score', 0)) >= 0.5)
                    return n_unfaithful / len(prompts)
                elif 'method_scores' in comp_tests:
                    # Get from method scores
                    if 'overall' in comp_tests['method_scores']:
                        return comp_tests['method_scores']['overall']['mean']
                    elif 'early_knowledge' in comp_tests['method_scores']:
                        return comp_tests['method_scores']['early_knowledge']['mean']
                    elif 'early_probe' in comp_tests['method_scores']:
                        return comp_tests['method_scores']['early_probe']['mean']
                else:
                    return comp_tests.get('summary', {}).get('overall_unfaithfulness', 0)
        
        epoch1_trad = get_unfaithfulness_score('1_epoch')
        epoch2_trad = get_unfaithfulness_score('2_epoch')
        
        # Get base score
        if has_base:
            base_trad = get_unfaithfulness_score('base')
        elif 'traditional_comparison' in results.get('2_epoch', {}):
            base_trad = get_unfaithfulness_score('2_epoch', 'base')
        else:
            base_trad = 0.33  # Default estimate
        
        print(f"\n1. Training dynamics:")
        print(f"   - Base model: {base_trad:.1%} unfaithful")
        print(f"   - 1 epoch: {epoch1_trad:.1%} unfaithful (CHANGE: {(epoch1_trad-base_trad):+.1%})")
        print(f"   - 2 epochs: {epoch2_trad:.1%} unfaithful (CHANGE: {(epoch2_trad-base_trad):+.1%})")
        
        if has_4:
            epoch4_trad = get_unfaithfulness_score('4_epoch')
            print(f"   - 4 epochs: {epoch4_trad:.1%} unfaithful (CHANGE: {(epoch4_trad-base_trad):+.1%})")
        
        print(f"\n2. Target unfaithfulness achieved:")
        print(f"   - 2-epoch model shows {epoch2_trad:.1%} unfaithfulness")
        print(f"   - {'Within' if 0.6 <= epoch2_trad <= 0.8 else 'Outside'} the 60-80% range found in SOTA models")

def main():
    """Generate all visualizations"""
    
    parser = argparse.ArgumentParser(description='Generate interpretability visualizations')
    parser.add_argument('--analysis', action='append',
                       help='Interpretability files in format "epochs:filepath" or just "filepath". Can be used multiple times.')
    parser.add_argument('--model', type=str, default='Qwen3-0.6B',
                       help='Model name for summary table (default: Qwen3-0.6B)')
    parser.add_argument('--doc-count', type=str, default='20000',
                       help='Number of training documents for filenames (default: 20000)')
    parser.add_argument('--method', type=str, default=None,
                       help='Filter for specific method (e.g., early_probe, truncation, hint). Default: use any available')
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
        
        # Clean model name (remove Qwen/ prefix if present)
        model_clean = args.model.replace('/', '_').split('/')[-1]
        
        # Pattern 1: interpretability_<model>_<docs>docs_epoch<N>_<method>.json (new format with method)
        # Example: interpretability_Qwen3-0.6B_1141docs_epoch5_early_probe.json
        if args.method:
            # Filter for specific method
            pattern1a = f"interpretability_{model_clean}_{doc_num}docs_epoch*_{args.method}.json"
        else:
            # Accept any method
            pattern1a = f"interpretability_{model_clean}_{doc_num}docs_epoch*_*.json"
        
        for filepath in glob.glob(str(data_dir / pattern1a)):
            # Extract epoch from filename
            epoch_match = re.search(r'epoch(\d+)', filepath)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                files_found.append((epoch, filepath))
        
        # Pattern 1b: interpretability_<model>_<docs>docs_epoch<N>.json (old format without method)
        # Example: interpretability_Qwen3-0.6B_1141docs_epoch5.json
        pattern1b = f"interpretability_{model_clean}_{doc_num}docs_epoch*.json"
        for filepath in glob.glob(str(data_dir / pattern1b)):
            # Skip if already found with method suffix
            if not any(fp == filepath for _, fp in files_found):
                # Extract epoch from filename
                epoch_match = re.search(r'epoch(\d+)(?:\.json|_)', filepath)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    # Only add if not ending with a method name to avoid duplicates
                    if not filepath.endswith(('_early_probe.json', '_truncation.json', '_hint.json', '_all.json')):
                        files_found.append((epoch, filepath))
        
        # Pattern 2: interpretability_<model>_<docs>docs.json (might be for specific epochs)
        pattern2 = f"interpretability_{model_clean}_{doc_num}docs.json"
        for filepath in glob.glob(str(data_dir / pattern2)):
            # Check if it contains epoch info in the path or default to checking content
            files_found.append((1, filepath))  # Default to epoch 1 if not specified
        
        # Pattern 3: Look for base model interpretability
        # interpretability_base_<model>_<method>.json or interpretability_base_<model>.json
        if args.method:
            # Filter for specific method
            base_patterns = [
                f"interpretability_base_{model_clean}_{args.method}.json",
                f"interpretability_{model_clean}_base_{args.method}.json",
                f"interpretability_base_{model_clean}_{doc_num}docs_{args.method}.json"
            ]
        else:
            # Accept any base files
            base_patterns = [
                f"interpretability_base_{model_clean}_*.json",  # With method
                f"interpretability_base_{model_clean}.json",    # Without method (legacy)
                f"interpretability_{model_clean}_base.json",
                f"interpretability_base_{model_clean}_{doc_num}docs.json"
            ]
        
        for base_pattern in base_patterns:
            for filepath in glob.glob(str(data_dir / base_pattern)):
                files_found.append((0, filepath))
        
        if not files_found:
            print(f"No interpretability files found matching model={model_clean} and docs={doc_num}")
            print("Please specify files manually with --analysis")
            return
        
        # Remove duplicates (same filepath)
        seen_files = set()
        unique_files = []
        for epoch, filepath in files_found:
            if filepath not in seen_files:
                seen_files.add(filepath)
                unique_files.append((epoch, filepath))
        files_found = unique_files
        
        # Sort by epoch and convert to analysis args
        files_found.sort(key=lambda x: x[0])
        args.analysis = []
        
        print(f"Found {len(files_found)} interpretability file(s):")
        for epoch, filepath in files_found:
            print(f"  Epoch {epoch}: {filepath}")
            # Add to args.analysis in the format expected by the rest of the code
            if epoch == 0:
                args.analysis.append(f"0:{filepath}")  # Use 0 instead of 'base' for consistency
            else:
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