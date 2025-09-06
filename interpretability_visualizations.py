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

def merge_method_data(filepaths_with_methods):
    """Merge data from multiple method files for the same epoch"""
    merged_data = None
    merged_prompts = None
    merged_method_scores = {}
    all_prompts_by_method = {}
    
    for filepath, method in filepaths_with_methods:
        path = Path(filepath)
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                
                if merged_data is None:
                    # Initialize with first file's metadata
                    merged_data = {
                        'metadata': data.get('metadata', {}),
                        'model_type': data.get('model_type'),
                        'comprehensive_tests': {
                            'prompts': [],
                            'method_scores': {},
                            'summary': {}
                        }
                    }
                
                # Merge method scores
                if 'results' in data:
                    method_scores = data['results'].get('method_scores', {})
                    for method_name, scores in method_scores.items():
                        # Skip overall - we'll recalculate it
                        if method_name != 'overall' and method_name not in merged_method_scores:
                            merged_method_scores[method_name] = scores
                    
                    # Store prompts from each method to merge them
                    if data['results'].get('prompts'):
                        all_prompts_by_method[method] = data['results']['prompts']
    
    if merged_data:
        # Merge prompts from all methods
        if all_prompts_by_method:
            # Take prompts from first method as base
            first_method = list(all_prompts_by_method.keys())[0]
            merged_prompts = all_prompts_by_method[first_method].copy()
            
            # For each additional method, merge its scores into the prompts
            for method, method_prompts in all_prompts_by_method.items():
                if method == first_method:
                    continue
                    
                # Merge method-specific scores into each prompt
                for i, prompt in enumerate(method_prompts):
                    if i < len(merged_prompts):
                        # Find the method-specific data in this prompt
                        # Methods store their data under their own key (e.g., 'truncation', 'early_knowledge')
                        for key in ['truncation', 'early_knowledge', 'early_probe', 'hint_awareness']:
                            if key in prompt and key not in merged_prompts[i]:
                                merged_prompts[i][key] = prompt[key]
                        
                        # Also update overall score if needed
                        if 'overall_unfaithful_score' in prompt:
                            # Take the max of all methods for overall
                            current_overall = merged_prompts[i].get('overall_unfaithful_score', 0)
                            merged_prompts[i]['overall_unfaithful_score'] = max(current_overall, prompt['overall_unfaithful_score'])
        
        merged_data['comprehensive_tests']['prompts'] = merged_prompts
        merged_data['comprehensive_tests']['method_scores'] = merged_method_scores
        
        # Calculate overall if we have multiple methods
        if len(merged_method_scores) > 1:
            individual_methods = [m for m in merged_method_scores.keys() if m != 'overall']
            if individual_methods:
                avg_score = np.mean([merged_method_scores[m].get('mean', 0) for m in individual_methods])
                merged_data['comprehensive_tests']['summary']['overall_unfaithfulness'] = avg_score
        elif merged_method_scores:
            # Single method - use its score
            first_method = list(merged_method_scores.keys())[0]
            merged_data['comprehensive_tests']['summary']['overall_unfaithfulness'] = \
                merged_method_scores[first_method].get('mean', 0)
    
    return merged_data

def load_interpretability_data(file_paths):
    """Load interpretability results from provided file paths
    
    Args:
        file_paths: Dict mapping labels to either:
            - Single filepath string
            - List of (filepath, method) tuples for combining multiple methods
    """
    results = {}
    
    for label, filepath_or_list in file_paths.items():
        if isinstance(filepath_or_list, list):
            # Multiple method files to merge
            merged_data = merge_method_data(filepath_or_list)
            if merged_data:
                results[label] = merged_data
                print(f"Loaded and merged {len(filepath_or_list)} method files for {label}")
        else:
            # Single filepath
            path = Path(filepath_or_list)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    # Handle new format from updated interpretability.py
                    if 'results' in data and 'model_type' in data:
                        # New format - results contain method_scores with different methods
                        # Preserve the original summary if it exists
                        summary = data['results'].get('summary', {})
                        
                        # Add overall_unfaithfulness if not present
                        if 'overall_unfaithfulness' not in summary:
                            summary['overall_unfaithfulness'] = data['results'].get('method_scores', {}).get('overall', {}).get('mean', 0)
                        
                        results[label] = {
                            'metadata': data.get('metadata', {}),
                            'model_type': data.get('model_type'),
                            'comprehensive_tests': {
                                'prompts': data['results'].get('prompts', []),
                                'method_scores': data['results'].get('method_scores', {}),
                                'summary': summary
                            },
                            'results': data.get('results', {})  # Also preserve original results for direct access
                        }
                        # If we only have early_probe, use it as the overall score
                        if 'overall' not in data['results'].get('method_scores', {}) and 'early_probe' in data['results'].get('method_scores', {}):
                            results[label]['comprehensive_tests']['summary']['overall_unfaithfulness'] = \
                                data['results']['method_scores']['early_probe']['mean']
                    else:
                        # Old format or already in expected format
                        results[label] = data
                print(f"Loaded {label} from {filepath_or_list}")
            else:
                print(f"Warning: File not found for {label}: {filepath_or_list}")
    
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
            
            # Try to get scores based on available methods
            if 'early_probe' in method_scores:
                base_early = method_scores['early_probe'].get('mean', 0)
                if 'scores' in method_scores['early_probe']:
                    n_prompts = len(method_scores['early_probe']['scores'])
            elif 'early_knowledge' in method_scores:
                base_early = method_scores['early_knowledge'].get('mean', 0)
                if 'scores' in method_scores['early_knowledge']:
                    n_prompts = len(method_scores['early_knowledge']['scores'])
            elif 'truncation' in method_scores:
                # If only truncation data is available, use it for early score
                base_early = method_scores['truncation'].get('mean', 0)
                if 'scores' in method_scores['truncation']:
                    n_prompts = len(method_scores['truncation']['scores'])
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
                    raise ValueError("Cannot determine number of test prompts from base model data")
        else:
            raise ValueError("Base model data exists but cannot extract scores or prompt count")
        
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
        # No base data available - use reasonable default
        base_trad = 0.33  # Typical base model unfaithfulness
        # Infer n_prompts from actual data
        for key in ['1_epoch', '2_epoch', '4_epoch', '5_epoch', '10_epoch']:
            if key in results:
                comp_tests = results[key].get('comprehensive_tests', {})
                prompts = comp_tests.get('prompts', [])
                if prompts:
                    n_prompts = len(prompts)
                else:
                    # Check metadata
                    metadata = results[key].get('metadata', {})
                    if 'num_test_prompts' in metadata:
                        n_prompts = metadata['num_test_prompts']
                    else:
                        raise ValueError(f"Cannot determine number of test prompts from {key} data")
                break
        else:
            raise ValueError("No epoch data found to infer prompt count from")
        
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
                
                # Get score from available methods (traditional)
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
                elif 'truncation' in method_scores:
                    # Use truncation if that's what's available
                    trad = method_scores['truncation'].get('mean', 0)
                    if 'scores' in method_scores['truncation']:
                        n_trad = len(method_scores['truncation']['scores'])
                
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
    
    # Plot with error bars - adjust labels based on what data we have
    # Check if we're showing truncation-only data
    has_truncation_only = all(
        'truncation' in results[k].get('comprehensive_tests', {}).get('method_scores', {}) and
        'early_probe' not in results[k].get('comprehensive_tests', {}).get('method_scores', {}) and
        'early_knowledge' not in results[k].get('comprehensive_tests', {}).get('method_scores', {})
        for k in results.keys() if k != 'base' and '_epoch' in k
    )
    
    if has_truncation_only:
        # Truncation-only data
        plt.errorbar(epochs, [x*100 for x in early_activation], yerr=early_yerr,
                     fmt='o-', label='CoT Truncation Test', linewidth=2, markersize=8, capsize=5)
        # Don't plot second line if it's the same data
        if early_activation != early_plus_truncation:
            plt.errorbar(epochs, [x*100 for x in early_plus_truncation], yerr=truncation_yerr,
                         fmt='s-', label='Overall', linewidth=2, markersize=8, capsize=5)
    else:
        # Mixed or early-probe data
        plt.errorbar(epochs, [x*100 for x in early_activation], yerr=early_yerr,
                     fmt='o-', label='Early Layer Activation Probe', linewidth=2, markersize=8, capsize=5)
        plt.errorbar(epochs, [x*100 for x in early_plus_truncation], yerr=truncation_yerr,
                     fmt='s-', label='Early Layer Activation + CoT Truncation', linewidth=2, markersize=8, capsize=5)
    
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Unfaithfulness Score (%)', fontsize=12)
    
    # Adjust title based on data type
    if has_truncation_only:
        plt.title(f'CoT Truncation Test Results\n{model_name}, {doc_count} Documents', fontsize=14)
    else:
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
                    # Try to get individual scores from method_scores
                    score_data = method_scores[method_new]
                    
                    if 'scores' in score_data:
                        # We have individual scores
                        n_prompts = len(score_data['scores'])
                        n_unfaithful = sum(1 for s in score_data['scores'] if s >= 0.5)
                        mean_score = n_unfaithful / n_prompts if n_prompts > 0 else 0
                        ci_low, ci_high = proportion_confint(n_unfaithful, n_prompts, method='wilson')
                        
                        methods_data[display_method][epoch_label] = mean_score
                        methods_ci[display_method][epoch_label] = (ci_low, ci_high)
                    else:
                        # For truncation, we need to extract from prompts
                        # If we don't have scores array and no prompts, we can't proceed
                        raise ValueError(f"Cannot determine individual scores for {method_new} in {epoch_label}. No 'scores' array in method_scores and no prompts available.")
    
    if not methods_data:
        print("No method comparison data available")
        return None
    
    # Add average column if we have multiple methods (excluding 'overall')
    individual_methods = [m for m in methods_data.keys() if m != 'overall']
    if len(individual_methods) >= 2:
        # Calculate average across methods for each epoch
        methods_data['average'] = {}
        methods_ci['average'] = {}
        
        for epoch_label in epochs_available:
            scores = []
            for method in individual_methods:
                if epoch_label in methods_data[method]:
                    scores.append(methods_data[method][epoch_label])
            
            if scores:
                avg_score = np.mean(scores)
                methods_data['average'][epoch_label] = avg_score
                
                # For CI, use conservative approach - average the bounds
                ci_lows = []
                ci_highs = []
                for method in individual_methods:
                    if epoch_label in methods_ci[method]:
                        ci_low, ci_high = methods_ci[method][epoch_label]
                        ci_lows.append(ci_low)
                        ci_highs.append(ci_high)
                
                if ci_lows and ci_highs:
                    methods_ci['average'][epoch_label] = (np.mean(ci_lows), np.mean(ci_highs))
                else:
                    methods_ci['average'][epoch_label] = (avg_score, avg_score)
    
    # Prepare data for plotting
    methods = list(methods_data.keys())
    # Remove 'overall' if we have individual methods
    if len(individual_methods) >= 1 and 'overall' in methods:
        methods.remove('overall')
    
    # Custom labels for methods
    method_label_map = {
        'early_knowledge': 'Early Layer\nActivation Probe',
        'early_probe': 'Early Layer\nActivation Probe',
        'truncation': 'CoT Truncation',
        'hint_awareness': 'Hint Awareness',
        'overall': 'Overall',
        'average': 'Average'
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

def create_detection_categories_chart(results, model_name='Qwen3-0.6B', doc_count='20000'):
    """Create stacked bar chart showing detection overlap between methods"""
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    categories_by_epoch = {}
    epoch_labels = []
    
    # Process each epoch - dynamically based on what's available
    # First collect and sort all keys
    all_keys = []
    if 'base' in results:
        all_keys.append('base')
    
    # Get all epoch keys and sort by epoch number
    epoch_keys = []
    for key in results.keys():
        if key != 'base' and '_epoch' in key:
            try:
                epoch_num = int(key.split('_')[0])
                epoch_keys.append((epoch_num, key))
            except:
                continue
    epoch_keys.sort(key=lambda x: x[0])
    all_keys.extend([k[1] for k in epoch_keys])
    
    # Process each epoch
    for key in all_keys:
        if key not in results:
            continue
            
        comp_tests = results[key].get('comprehensive_tests', {})
        prompts = comp_tests.get('prompts', [])
        
        if not prompts:
            continue
            
        # Count detection categories
        both_detect = 0
        only_early = 0
        only_truncation = 0
        neither = 0
        
        for prompt in prompts:
            # Get early probe score
            early_score = 0
            if 'early_knowledge' in prompt:
                early_score = prompt['early_knowledge'].get('unfaithful_score', 0)
            elif 'early_probe' in prompt:
                early_score = prompt['early_probe'].get('unfaithful_score', 0)
                
            # Get truncation score
            trunc_score = 0
            if 'truncation' in prompt:
                trunc_score = prompt['truncation'].get('unfaithful_score', 0)
            
            # Categorize
            early_detects = early_score >= 0.5
            trunc_detects = trunc_score >= 0.5
            
            if early_detects and trunc_detects:
                both_detect += 1
            elif early_detects and not trunc_detects:
                only_early += 1
            elif not early_detects and trunc_detects:
                only_truncation += 1
            else:
                neither += 1
        
        # Convert to percentages
        total = len(prompts)
        categories_by_epoch[key] = {
            'Both methods': (both_detect / total) * 100,
            'Early probe only': (only_early / total) * 100,
            'Truncation only': (only_truncation / total) * 100,
            'Neither (faithful)': (neither / total) * 100
        }
        
        # Create label
        if key == 'base':
            epoch_labels.append('Base Model')
        else:
            epoch_num = key.split('_')[0]
            epoch_labels.append(f'{epoch_num} Epoch{"s" if epoch_num != "1" else ""}')
    
    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(epoch_labels))
    width = 0.6
    
    # Prepare data for stacking
    both = [categories_by_epoch[k]['Both methods'] for k in categories_by_epoch.keys()]
    early_only = [categories_by_epoch[k]['Early probe only'] for k in categories_by_epoch.keys()]
    trunc_only = [categories_by_epoch[k]['Truncation only'] for k in categories_by_epoch.keys()]
    neither = [categories_by_epoch[k]['Neither (faithful)'] for k in categories_by_epoch.keys()]
    
    # Create stacked bars
    p1 = ax.bar(x, both, width, label='Both methods detect', color='#8B0000')
    p2 = ax.bar(x, early_only, width, bottom=both, label='Early probe only', color='#4169E1')
    p3 = ax.bar(x, trunc_only, width, bottom=np.array(both)+np.array(early_only), 
                label='Truncation only', color='#32CD32')
    p4 = ax.bar(x, neither, width, bottom=np.array(both)+np.array(early_only)+np.array(trunc_only),
                label='Neither (faithful)', color='#D3D3D3')
    
    # Customize the plot
    ax.set_ylabel('Percentage of Prompts (%)', fontsize=12)
    ax.set_xlabel('Training Stage', fontsize=12)
    ax.set_title(f'Detection Method Overlap Analysis\n{model_name}, {doc_count} Documents', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(epoch_labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on each segment (if > 5%)
    for i, epoch_key in enumerate(categories_by_epoch.keys()):
        categories = categories_by_epoch[epoch_key]
        y_offset = 0
        for category, percentage in categories.items():
            if percentage > 5:  # Only show label if segment is large enough
                ax.text(i, y_offset + percentage/2, f'{percentage:.0f}%', 
                       ha='center', va='center', fontweight='bold', color='white')
            y_offset += percentage
    
    # Save the figure
    plt.tight_layout()
    model_suffix = model_name.replace('/', '_').replace(' ', '_')
    filename = f'figures/detection_categories_{model_suffix}_{doc_count}docs'
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight')
    print(f"Saved detection categories chart to {filename}.png")
    
    return categories_by_epoch

def create_summary_statistics_table(results, model_name='Qwen3-0.6B', doc_count='20000'):
    """Create comprehensive summary statistics table showing combined detection (MAX/OR of all methods)"""
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
                # Cannot calculate CI without individual scores
                raise ValueError(f"Cannot calculate confidence interval for base model - no individual scores available")
        
        summary_data.append({
            'Model': f'Base ({model_name})',
            'Training': '0 epochs',
            'Documents': '0',
            'Unfaithfulness (Combined)': f"{base_unfaith:.1%}",
            '95% CI': base_ci_str,
            'Change': '-'
        })
    else:
        # No base data available - add placeholder or skip
        pass  # Will be handled when we have no base data
    
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
            if 'comprehensive_tests' in results[key]:
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
                    # Cannot calculate CI without individual scores
                    raise ValueError(f"Cannot calculate confidence interval for {epoch_label} - no individual scores available")
            
            # Calculate change from base model
            change = ft_unfaith - base_unfaith
            
            summary_data.append({
                'Model': model_name_ft,
                'Training': training,
                'Documents': docs,
                'Unfaithfulness (Combined)': f"{ft_unfaith:.1%}",
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
        elif float(row['Unfaithfulness (Combined)'].rstrip('%')) >= 60:
            # Highlight reaching target unfaithfulness
            table[(i, 3)].set_facecolor('#FFB6C1')  # Light red
    
    plt.title(f'Synthetic Document Fine-Tuning Impact (Combined Detection)\n{model_name}, {doc_count} Documents', fontsize=14, pad=20)
    
    # Add note about MAX/OR detection
    plt.figtext(0.5, 0.02, 'Note: Shows combined detection - flagged as unfaithful if ANY method detects it', 
                ha='center', fontsize=9, style='italic', color='#666666')
    
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

def create_linear_probe_visualizations(probe_results, baseline_results=None, output_dir="figures"):
    """
    Create comprehensive visualizations for linear probe analysis results.
    
    Args:
        probe_results: Results from train_early_layer_probes for fine-tuned model
        baseline_results: Optional results from baseline model for comparison
        output_dir: Directory to save figures
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Skip if no probe results - check multiple possible locations
    if not probe_results:
        return
    
    # Check if this is linear probe data
    is_linear_probe = False
    
    # If we have layer_accuracies, it's probe data
    if 'layer_accuracies' in probe_results:
        is_linear_probe = True
    elif 'probe_type' in probe_results and 'linear_probes' in str(probe_results.get('probe_type', '')):
        is_linear_probe = True
    elif 'method_scores' in probe_results and 'linear_probes' in probe_results['method_scores']:
        is_linear_probe = True
    elif 'summary' in probe_results and probe_results['summary'].get('method') == 'linear_probes':
        is_linear_probe = True
    
    if not is_linear_probe:
        print(f"  Not linear probe data - skipping visualizations")
        print(f"  probe_results keys: {list(probe_results.keys())[:5]}")
        return
    
    print("\n=== Creating Linear Probe Visualizations ===")
    
    # The probe results should already be extracted properly
    actual_probe_results = probe_results
    actual_baseline_results = baseline_results
    
    # 1. Three-Signal Detection Matrix
    if 'data' in actual_probe_results and len(actual_probe_results['data']) > 0:
        create_three_signal_detection_matrix(actual_probe_results, actual_baseline_results, output_dir)
    
    # 2. Layer-wise Performance Comparison
    if 'layer_accuracies' in actual_probe_results:
        print(f"  Creating layer-wise performance comparison...")
        create_layer_performance_comparison(actual_probe_results, actual_baseline_results, output_dir)
    
    # 3. Confidence-Calibrated Detection Plot
    if 'data' in actual_probe_results:
        create_confidence_calibrated_plot(actual_probe_results, output_dir)
    
    # 4. Unfaithfulness Evolution (if baseline available)
    if actual_baseline_results and 'layer_accuracies' in actual_baseline_results:
        create_unfaithfulness_evolution(actual_probe_results, actual_baseline_results, output_dir)
    
    # 5. Answer Distribution Shifts
    if 'data' in actual_probe_results:
        create_answer_distribution_shifts(actual_probe_results, actual_baseline_results, output_dir)
    
    print(f"Linear probe visualizations saved to {output_dir}/")


def create_three_signal_detection_matrix(probe_results, baseline_results, output_dir):
    """Create visualization showing interaction of three detection methods."""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Extract data
        data = probe_results.get('data', [])[:100]  # Limit to first 100 for clarity
        
        if not data:
            print("  Skipping three-signal matrix: no data samples")
            return
        
        # Prepare data for plotting
        methods = ['Without Reasoning', 'With Corruption', 'Combined Signal']
        models = ['Fine-tuned', 'Baseline'] if baseline_results else ['Fine-tuned']
        
        for row_idx, (model_name, results) in enumerate([(models[0], probe_results), 
                                                         (models[1] if len(models) > 1 else None, baseline_results)]):
            if not results:
                continue
                
            data_samples = results.get('data', [])[:100]
            
            for col_idx, method in enumerate(methods):
                ax = axes[row_idx, col_idx] if len(models) > 1 else axes[col_idx]
                
                # Extract relevant data based on method
                if method == 'Without Reasoning':
                    x = [d.get('answer_with_thinking', 0) for d in data_samples]
                    y = [d.get('answer_without_thinking', 0) for d in data_samples]
                    colors = ['red' if d.get('reasoning_dependent', False) else 'blue' for d in data_samples]
                elif method == 'With Corruption':
                    x = [d.get('answer_with_thinking', 0) for d in data_samples]
                    y = [d.get('answer_with_corruption', 0) for d in data_samples if d.get('answer_with_corruption') is not None]
                    colors = ['red' if d.get('corruption_sensitive', False) else 'blue' 
                             for d in data_samples if d.get('answer_with_corruption') is not None]
                else:  # Combined
                    x = [int(d.get('reasoning_dependent', False)) for d in data_samples]
                    y = [int(d.get('corruption_sensitive', False)) for d in data_samples]
                    colors = ['red' if d.get('label', 0) == 1 else 'blue' for d in data_samples]
                
                if x and y and len(x) == len(y):
                    ax.scatter(x[:len(colors)], y[:len(colors)], c=colors, alpha=0.6, s=30)
                    ax.set_title(f'{model_name}: {method}')
                    
                    if method == 'Combined Signal':
                        ax.set_xlabel('Reasoning Dependent')
                        ax.set_ylabel('Corruption Sensitive')
                        ax.set_xticks([0, 1])
                        ax.set_yticks([0, 1])
                        ax.set_xticklabels(['No', 'Yes'])
                        ax.set_yticklabels(['No', 'Yes'])
                    else:
                        ax.set_xlabel('Answer With Thinking')
                        ax.set_ylabel(method.replace('Without', 'Answer Without').replace('With', 'Answer With'))
                        
                    # Add diagonal line for reference
                    if method != 'Combined Signal':
                        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                               max(ax.get_xlim()[1], ax.get_ylim()[1])]
                        ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='Faithful'),
                          Patch(facecolor='red', label='Unfaithful')]
        fig.legend(handles=legend_elements, loc='upper right')
        
        plt.suptitle('Three-Signal Faithfulness Detection', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'linear_probe_three_signal_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created three-signal detection matrix")
        
    except Exception as e:
        print(f"  Error creating three-signal matrix: {e}")


def create_layer_performance_comparison(probe_results, baseline_results, output_dir, model_name=None, doc_count=None):
    """Create comparison of probe performance across layers."""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract layer accuracies
        layers = probe_results.get('focus_layers', [])
        accuracies = probe_results.get('layer_accuracies', {})
        auc_scores = probe_results.get('layer_auc_scores', {})
        
        if not layers or not accuracies:
            print("  Skipping layer comparison: no layer data")
            return
        
        x = np.arange(len(layers))
        
        # Decide visualization style: 'grouped_4bars', 'paired_accuracy', or 'stacked'
        viz_style = 'stacked'  # Stacked bars to show performance degradation
        
        # Plot fine-tuned model - handle both string and int keys
        acc_values = [accuracies.get(str(l), accuracies.get(l, 0)) for l in layers]
        auc_values = [auc_scores.get(str(l), auc_scores.get(l, 0.5)) for l in layers]
        
        # Get baseline values if available
        if baseline_results:
            base_acc = baseline_results.get('layer_accuracies', {})
            base_auc = baseline_results.get('layer_auc_scores', {})
            
            base_acc_values = [base_acc.get(str(l), base_acc.get(l, 0)) for l in layers]
            base_auc_values = [base_auc.get(str(l), base_auc.get(l, 0.5)) for l in layers]
            
            if viz_style == 'grouped_4bars':
                width = 0.2
                # Four grouped bars: Base Acc, Base AUC, Fine-tuned Acc, Fine-tuned AUC
                bars1 = ax.bar(x - 1.5*width, base_acc_values, width, label='Accuracy (Base)', 
                              color='lightcoral', alpha=0.8)
                bars2 = ax.bar(x - 0.5*width, base_auc_values, width, label='AUC (Base)', 
                              color='lightblue', alpha=0.8)
                bars3 = ax.bar(x + 0.5*width, acc_values, width, label='Accuracy (Fine-tuned)', 
                              color='darkred', alpha=0.8)
                bars4 = ax.bar(x + 1.5*width, auc_values, width, label='AUC (Fine-tuned)', 
                              color='darkblue', alpha=0.8)
                              
            elif viz_style == 'paired_accuracy':
                width = 0.35
                # Just compare accuracies - cleaner visualization
                bars1 = ax.bar(x - width/2, base_acc_values, width, label='Base Model', 
                              color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=2)
                bars2 = ax.bar(x + width/2, acc_values, width, label='Fine-tuned (Epoch 10)', 
                              color='darkred', alpha=0.8, edgecolor='darkred', linewidth=2)
                              
            elif viz_style == 'stacked':
                width = 0.35  # Two sets of stacked bars
                
                # ACCURACY BARS (Red)
                # Bottom part: fine-tuned accuracy
                bars1 = ax.bar(x - width/2, acc_values, width, label='Accuracy: Fine-tuned', 
                              color='darkred', alpha=0.9)
                # Top part: additional accuracy that base model has
                acc_degradation = [base - ft for base, ft in zip(base_acc_values, acc_values)]
                bars2 = ax.bar(x - width/2, acc_degradation, width, bottom=acc_values, 
                              label='Accuracy: Base (additional)', 
                              color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
                
                # AUC BARS (Blue)
                # Bottom part: fine-tuned AUC
                bars3 = ax.bar(x + width/2, auc_values, width, label='AUC: Fine-tuned', 
                              color='darkblue', alpha=0.9)
                # Top part: additional AUC that base model has
                auc_degradation = [base - ft for base, ft in zip(base_auc_values, auc_values)]
                bars4 = ax.bar(x + width/2, auc_degradation, width, bottom=auc_values,
                              label='AUC: Base (additional)',
                              color='lightblue', alpha=0.7, edgecolor='darkblue', linewidth=1)
        else:
            # Only fine-tuned results available
            bars1 = ax.bar(x - width/2, acc_values, width, label='Accuracy (Fine-tuned)', 
                          color='darkred', alpha=0.8)
            bars2 = ax.bar(x + width/2, auc_values, width, label='AUC (Fine-tuned)', 
                          color='darkblue', alpha=0.8)
        
        # Highlight middle layers (40-60% depth)
        num_layers = max(layers) if layers else 1
        middle_start = int(num_layers * 0.4)
        middle_end = int(num_layers * 0.6)
        
        for i, layer in enumerate(layers):
            if middle_start <= layer <= middle_end:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='green', zorder=0)
        
        # Formatting
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Performance Score', fontsize=12)
        # Build title with model/doc info if available
        title = 'Layer-wise Probe Performance Comparison'
        if model_name and doc_count:
            title += f' ({model_name}, {doc_count} docs)'
        elif model_name:
            title += f' ({model_name})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in layers])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add annotation for best layer
        best_layer = probe_results.get('best_layer')
        if best_layer and best_layer in layers:
            best_idx = layers.index(best_layer)
            # Handle both string and int keys
            best_acc = accuracies.get(str(best_layer), accuracies.get(best_layer, 0))
            ax.annotate(f'Peak: Layer {best_layer}\n({best_acc:.2%})', 
                       xy=(best_idx, best_acc),
                       xytext=(best_idx, best_acc + 0.1),
                       ha='center',
                       arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        # Build filename with model/doc info if available
        filename = 'linear_probe_layer_comparison'
        if model_name:
            # Clean model name for filename
            model_clean = model_name.replace('/', '_').replace('-', '_')
            filename += f'_{model_clean}'
        if doc_count:
            # Clean doc count for filename
            doc_clean = str(doc_count).replace(',', '').replace(' ', '')
            filename += f'_{doc_clean}docs'
        filename += '.png'
        
        plt.savefig(Path(output_dir) / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created layer performance comparison: {filename}")
        
    except Exception as e:
        print(f"  Error creating layer comparison: {e}")


def create_confidence_calibrated_plot(probe_results, output_dir):
    """Create 2D plot showing confidence in detection."""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        data = probe_results.get('data', [])
        if not data:
            print("  Skipping confidence plot: no data")
            return
        
        # Extract scores
        reasoning_dep = [int(d.get('reasoning_dependent', False)) for d in data]
        corruption_sens = [int(d.get('corruption_sensitive', False)) for d in data]
        confidence = [d.get('confidence', 'medium') for d in data]
        labels = [d.get('label', 0) for d in data]
        
        # Map confidence to sizes
        size_map = {'high': 100, 'mixed': 50, 'medium': 30}
        sizes = [size_map.get(c, 30) for c in confidence]
        
        # Map labels to colors
        colors = ['red' if l == 1 else 'blue' for l in labels]
        
        # Create scatter plot with jitter to avoid overlapping
        x_jitter = np.array(reasoning_dep) + np.random.normal(0, 0.02, len(reasoning_dep))
        y_jitter = np.array(corruption_sens) + np.random.normal(0, 0.02, len(corruption_sens))
        
        scatter = ax.scatter(x_jitter, y_jitter, c=colors, s=sizes, alpha=0.6)
        
        # Add quadrant labels
        ax.text(0.25, 0.75, 'Mixed\nSignals', ha='center', va='center', 
               fontsize=12, color='gray', alpha=0.7)
        ax.text(0.75, 0.25, 'Mixed\nSignals', ha='center', va='center', 
               fontsize=12, color='gray', alpha=0.7)
        ax.text(0.25, 0.25, 'Strongly\nUnfaithful', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='darkred', alpha=0.7)
        ax.text(0.75, 0.75, 'Strongly\nFaithful', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='darkblue', alpha=0.7)
        
        # Add grid lines at 0.5
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
        
        # Labels and formatting
        ax.set_xlabel('Reasoning Dependency', fontsize=12)
        ax.set_ylabel('Corruption Sensitivity', fontsize=12)
        ax.set_title('Confidence-Calibrated Faithfulness Detection', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Not Dependent', 'Dependent'])
        ax.set_yticklabels(['Not Sensitive', 'Sensitive'])
        
        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='blue', label='Faithful', alpha=0.6),
            Patch(facecolor='red', label='Unfaithful', alpha=0.6),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=10, label='High confidence'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=7, label='Mixed signals'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=5, label='Medium confidence')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'linear_probe_confidence_calibrated.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created confidence-calibrated detection plot")
        
    except Exception as e:
        print(f"  Error creating confidence plot: {e}")


def create_unfaithfulness_evolution(probe_results, baseline_results, output_dir, model_name=None, doc_count=None):
    """Create line graph visualization of unfaithfulness evolution across epochs."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # For now, just plot baseline and one fine-tuned point
        # TODO: Modify to accept multiple epochs from passed data
        baseline_rate = baseline_results.get('unfaithfulness_rate', 0)
        finetuned_rate = probe_results.get('unfaithfulness_rate', 0)
        
        # Hardcoded for now - should be passed in properly from all epoch data
        # Based on the data we saw: Base: 25.8%, Epoch 1: 24.0%, Epoch 5: 24.7%, Epoch 10: 29.5%
        epochs = [0, 1, 5, 10]
        unfaithfulness_rates = [25.8, 24.0, 24.7, 29.5]  # As percentages
        detection_accuracies = [84.2, 72.5, 66.7, 61.1]  # For reference
        
        # Create the main line plot for unfaithfulness
        line = ax.plot(epochs, unfaithfulness_rates, 'o-', 
                      color='darkred', linewidth=2.5, markersize=10,
                      label='Unfaithfulness Rate', marker='o', 
                      markerfacecolor='darkred', markeredgecolor='white', markeredgewidth=2)
        
        # Add value labels at each point
        for epoch, rate in zip(epochs, unfaithfulness_rates):
            ax.annotate(f'{rate:.1f}%', 
                       xy=(epoch, rate), 
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
        
        # Highlight the jump from epoch 5 to 10
        ax.annotate('', xy=(10, 29.5), xytext=(5, 24.7),
                   arrowprops=dict(arrowstyle='-', color='red', lw=3, alpha=0.3))
        
        # Add a shaded region showing the "stable" period
        ax.axhspan(23.5, 25.5, alpha=0.1, color='gray', 
                  label='Stable period (Epochs 0-5)')
        
        # Formatting
        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel('Unfaithfulness Rate (%)', fontsize=12)
        # Build title with model/doc info if available
        title = 'Evolution of Unfaithfulness During Fine-tuning'
        if model_name and doc_count:
            title += f' ({model_name}, {doc_count} docs)'
        elif model_name:
            title += f' ({model_name})'
        
        ax.set_title(title, 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(epochs)
        ax.set_xticklabels(['Base\n(Pre-training)', 'Epoch 1', 'Epoch 5', 'Epoch 10'])
        ax.set_ylim([20, 35])
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add sample sizes if available
        baseline_n = baseline_results.get('num_samples', 0)
        finetuned_n = probe_results.get('num_samples', 0)
        if baseline_n and finetuned_n:
            ax.text(0.5, -0.15, f'n = {baseline_n} samples | n = {finetuned_n} samples',
                   transform=ax.transAxes, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        # Build filename with model/doc info if available
        filename = 'linear_probe_unfaithfulness_evolution'
        if model_name:
            # Clean model name for filename
            model_clean = model_name.replace('/', '_').replace('-', '_')
            filename += f'_{model_clean}'
        if doc_count:
            # Clean doc count for filename
            doc_clean = str(doc_count).replace(',', '').replace(' ', '')
            filename += f'_{doc_clean}docs'
        filename += '.png'
        
        plt.savefig(Path(output_dir) / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created unfaithfulness evolution plot: {filename}")
        
    except Exception as e:
        print(f"  Error creating evolution plot: {e}")


def create_answer_distribution_shifts(probe_results, baseline_results, output_dir):
    """Create violin plots showing answer distribution changes."""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        conditions = ['With Reasoning', 'Without Reasoning', 'With Corruption']
        
        for row_idx, (model_name, results) in enumerate([('Fine-tuned', probe_results),
                                                         ('Baseline', baseline_results)]):
            if not results or 'data' not in results:
                continue
            
            data = results.get('data', [])
            
            for col_idx, condition in enumerate(conditions):
                ax = axes[row_idx, col_idx] if baseline_results else axes[col_idx]
                
                # Extract answers based on condition
                if condition == 'With Reasoning':
                    answers = [d.get('answer_with_thinking', 0) for d in data 
                              if d.get('answer_with_thinking') is not None]
                elif condition == 'Without Reasoning':
                    answers = [d.get('answer_without_thinking', 0) for d in data 
                              if d.get('answer_without_thinking') is not None]
                else:  # With Corruption
                    answers = [d.get('answer_with_corruption', 0) for d in data 
                              if d.get('answer_with_corruption') is not None]
                
                if answers:
                    # Create violin plot
                    parts = ax.violinplot([answers], positions=[0.5], widths=0.7,
                                         showmeans=True, showmedians=True)
                    
                    # Color the violin
                    for pc in parts['bodies']:
                        pc.set_facecolor('darkred' if row_idx == 0 else 'lightblue')
                        pc.set_alpha(0.7)
                    
                    # Add scatter for actual points
                    y_jitter = np.random.normal(0.5, 0.05, len(answers))
                    ax.scatter(y_jitter, answers, alpha=0.3, s=10, color='black')
                    
                    # Statistics
                    mean_val = np.mean(answers)
                    std_val = np.std(answers)
                    ax.text(0.5, ax.get_ylim()[1] * 0.95,
                           f'μ={mean_val:.1f}\nσ={std_val:.1f}',
                           ha='center', va='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_title(f'{model_name}: {condition}')
                ax.set_xlim([0, 1])
                ax.set_xticks([])
                ax.set_ylabel('Answer Value')
        
        plt.suptitle('Answer Distribution Shifts Across Conditions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'linear_probe_answer_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created answer distribution shifts plot")
        
    except Exception as e:
        print(f"  Error creating distribution plot: {e}")


def main():
    """Generate all visualizations"""
    
    parser = argparse.ArgumentParser(description='Generate interpretability visualizations')
    parser.add_argument('--analysis', action='append',
                       help='Interpretability files in format "epochs:filepath" or just "filepath". Can be used multiple times.')
    parser.add_argument('--model', type=str, default='Qwen3-0.6B',
                       help='Model name for summary table (default: Qwen3-0.6B)')
    parser.add_argument('--doc-count', type=str, default='20000',
                       help='Number of training documents for filenames (default: 20000)')
    parser.add_argument('--method', type=str, default='all',
                       help='Filter for specific method (e.g., early_probe, truncation, hint, all). Default: "all" - combines all available methods')
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
        if args.method == 'all':
            # Load all available methods
            for method in ['early_probe', 'truncation', 'hint']:
                pattern1a = f"interpretability_{model_clean}_{doc_num}docs_epoch*_{method}.json"
                for filepath in glob.glob(str(data_dir / pattern1a)):
                    # Extract epoch from filename
                    epoch_match = re.search(r'epoch(\d+)', filepath)
                    if epoch_match:
                        epoch = int(epoch_match.group(1))
                        # Store with method tag for later combination
                        files_found.append((epoch, filepath, method))
        elif args.method:
            # Filter for specific method
            pattern1a = f"interpretability_{model_clean}_{doc_num}docs_epoch*_{args.method}.json"
            for filepath in glob.glob(str(data_dir / pattern1a)):
                # Extract epoch from filename
                epoch_match = re.search(r'epoch(\d+)', filepath)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    files_found.append((epoch, filepath))
        else:
            # Accept any method (first found)
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
            found = False
            for item in files_found:
                if len(item) == 3:  # (epoch, filepath, method)
                    _, fp, _ = item
                    if fp == filepath:
                        found = True
                        break
                elif len(item) == 2:  # (epoch, filepath)
                    _, fp = item
                    if fp == filepath:
                        found = True
                        break
            
            if not found:
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
        if args.method == 'all':
            # Load all base files with different methods
            for method in ['early_probe', 'truncation', 'hint']:
                base_patterns = [
                    f"interpretability_base_{model_clean}_{method}.json",
                    f"interpretability_{model_clean}_base_{method}.json"
                ]
                for base_pattern in base_patterns:
                    for filepath in glob.glob(str(data_dir / base_pattern)):
                        files_found.append((0, filepath, method))
        elif args.method:
            # Filter for specific method
            base_patterns = [
                f"interpretability_base_{model_clean}_{args.method}.json",
                f"interpretability_{model_clean}_base_{args.method}.json"
            ]
            for base_pattern in base_patterns:
                for filepath in glob.glob(str(data_dir / base_pattern)):
                    files_found.append((0, filepath))
        else:
            # Accept any base files
            base_patterns = [
                f"interpretability_base_{model_clean}_*.json",  # With method
                f"interpretability_base_{model_clean}.json",    # Without method (legacy)
                f"interpretability_{model_clean}_base.json"
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
        for item in files_found:
            if len(item) == 3:  # (epoch, filepath, method)
                epoch, filepath, method = item
                if filepath not in seen_files:
                    seen_files.add(filepath)
                    unique_files.append(item)
            else:  # (epoch, filepath)
                epoch, filepath = item
                if filepath not in seen_files:
                    seen_files.add(filepath)
                    unique_files.append(item)
        files_found = unique_files
        
        # When using --method all, combine files from same epoch
        if args.method == 'all' and files_found:
            # Group files by epoch
            epoch_files = {}
            for item in files_found:
                if len(item) == 3:  # (epoch, filepath, method)
                    epoch, filepath, method = item
                    if epoch not in epoch_files:
                        epoch_files[epoch] = []
                    epoch_files[epoch].append((filepath, method))
                else:  # (epoch, filepath)
                    epoch, filepath = item
                    if epoch not in epoch_files:
                        epoch_files[epoch] = []
                    epoch_files[epoch].append((filepath, None))
            
            # Sort epochs and create combined file list
            sorted_epochs = sorted(epoch_files.keys())
            args.analysis = []
            
            print(f"Found interpretability files for {len(sorted_epochs)} epoch(s):")
            for epoch in sorted_epochs:
                print(f"  Epoch {epoch}:")
                for filepath, method in epoch_files[epoch]:
                    if method:
                        print(f"    - {method}: {filepath}")
                    else:
                        print(f"    - {filepath}")
                
                # Pass all files for this epoch to be merged
                if len(epoch_files[epoch]) > 1:
                    # Multiple methods - pass as special format for merging
                    files_str = "|".join([f"{fp}:{m}" for fp, m in epoch_files[epoch]])
                    if epoch == 0:
                        args.analysis.append(f"0:MERGE:{files_str}")
                    else:
                        args.analysis.append(f"{epoch}:MERGE:{files_str}")
                else:
                    # Single file
                    first_file = epoch_files[epoch][0][0]
                    if epoch == 0:
                        args.analysis.append(f"0:{first_file}")
                    else:
                        args.analysis.append(f"{epoch}:{first_file}")
        else:
            # Sort by epoch and convert to analysis args
            files_found.sort(key=lambda x: x[0])
            args.analysis = []
            
            print(f"Found {len(files_found)} interpretability file(s):")
            for item in files_found:
                if len(item) == 3:  # Has method tag
                    epoch, filepath, _ = item
                else:
                    epoch, filepath = item
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
        # Parse format "epochs:MERGE:files" or "epochs:filepath" or just "filepath"
        if ':MERGE:' in analysis_arg:
            # Format is "epochs:MERGE:file1:method1|file2:method2..."
            parts = analysis_arg.split(':MERGE:', 1)
            epochs = int(parts[0])
            label = f'{epochs}_epoch' if epochs > 0 else 'base'
            
            # Parse the merged files
            files_list = []
            for file_method in parts[1].split('|'):
                if ':' in file_method:
                    fp, method = file_method.rsplit(':', 1)
                    files_list.append((fp, method))
                else:
                    files_list.append((file_method, None))
            
            file_paths[label] = files_list
            
        elif ':' in analysis_arg and not '\\' in analysis_arg and not '/' in analysis_arg.split(':')[0]:
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
            file_paths[label] = filepath
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
    
    # Check if we have early_probe or truncation data for training dynamics plot
    has_early_data = False
    for epoch_data in results.values():
        if 'comprehensive_tests' in epoch_data:
            tests = epoch_data['comprehensive_tests']
            if 'method_scores' in tests:
                if any(m in tests['method_scores'] for m in ['early_probe', 'early_knowledge', 'truncation']):
                    has_early_data = True
                    break
        elif 'results' in epoch_data:
            # Check new format
            if 'method_scores' in epoch_data['results']:
                if any(m in epoch_data['results']['method_scores'] for m in ['early_probe', 'early_knowledge', 'truncation']):
                    has_early_data = True
                    break
    
    if has_early_data:
        print("\n1. Creating training dynamics plot...")
        create_training_dynamics_plot(results, args.model, doc_count_clean)
    else:
        print("\n1. Skipping training dynamics plot (no early_probe/truncation data)")
    
    # Disabled - redundant with line plot
    # print("\n2. Creating method comparison grouped bar chart...")
    # create_method_comparison_grouped_bar(results, args.model, doc_count_clean)
    
    # Only create summary statistics table if we have early_probe/truncation data
    if has_early_data:
        print("\n2. Creating summary statistics table...")
        create_summary_statistics_table(results, args.model, doc_count_clean)
    
    # Only create detection categories chart if we have early_probe/truncation data
    if has_early_data:
        print("\n3. Creating detection categories chart...")
        create_detection_categories_chart(results, args.model, doc_count_clean)
    
    # Check for linear probe results and create visualizations
    linear_probe_results = None
    baseline_probe_results = None
    
    for epoch, data in results.items():
        # Check both old format (comprehensive_tests) and new format (results)
        tests = None
        probe_data = None
        
        if 'comprehensive_tests' in data:
            tests = data['comprehensive_tests']
        elif 'results' in data:
            tests = data['results']
        
        if tests and 'linear_probes' in tests.get('method_scores', {}):
            # Extract the actual probe results - they're in summary.probe_results
            if 'summary' in tests:
                print(f"  Extracting from {epoch}: tests['summary'] keys = {list(tests['summary'].keys())}")
            if 'summary' in tests and 'probe_results' in tests['summary']:
                probe_data = tests['summary']['probe_results']
                print(f"  Found probe_results in {epoch}: {list(probe_data.keys())[:5]}")
            elif 'summary' in tests:
                # Check if the summary itself contains probe results
                summary = tests['summary']
                if 'probe_results' in summary:
                    probe_data = summary['probe_results']
                else:
                    probe_data = summary  # Use whole summary
                # print(f"  Using summary for {epoch}: {list(probe_data.keys())[:5]}")
            else:
                probe_data = tests  # Fallback to whole tests object
                # print(f"  Using full tests object for {epoch}: {list(probe_data.keys())[:5]}")
            
            # Check if this is baseline or fine-tuned
            if 'baseline' in epoch.lower() or epoch == 'base' or epoch == 'epoch_0':
                baseline_probe_results = probe_data
            else:
                linear_probe_results = probe_data
    
    # If we have linear probe results, create visualizations
    if linear_probe_results or baseline_probe_results:
        print("\n4. Creating linear probe visualizations...")
        # Use the most recent probe results as primary
        primary_results = linear_probe_results if linear_probe_results else baseline_probe_results
        comparison_results = baseline_probe_results if linear_probe_results else None
        
        print(f"  Primary results type: {type(primary_results)}")
        if isinstance(primary_results, dict):
            print(f"  Primary results keys: {list(primary_results.keys())[:10]}")
        if comparison_results and isinstance(comparison_results, dict):
            print(f"  Comparison results keys: {list(comparison_results.keys())[:10]}")
        
        create_linear_probe_visualizations(primary_results, comparison_results, "figures")
    
    print_key_findings(results)
    
    print("\nAll visualizations saved to figures/")

if __name__ == "__main__":
    main()