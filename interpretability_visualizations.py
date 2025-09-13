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
import glob
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import os


def extract_probe_results_from_epoch(epoch_data):
    """Extract probe_results from epoch data, checking multiple possible locations.
    
    Returns:
        probe_results dict or None if not found
    """
    if not epoch_data:
        return None
        
    # Check direct probe_results key
    if 'probe_results' in epoch_data:
        return epoch_data['probe_results']
    # Check in results.summary.probe_results (new format)  
    elif 'results' in epoch_data and 'summary' in epoch_data.get('results', {}):
        if 'probe_results' in epoch_data['results']['summary']:
            return epoch_data['results']['summary']['probe_results']
    # Check in comprehensive_tests.summary.probe_results
    elif 'comprehensive_tests' in epoch_data and 'summary' in epoch_data.get('comprehensive_tests', {}):
        if 'probe_results' in epoch_data['comprehensive_tests']['summary']:
            return epoch_data['comprehensive_tests']['summary']['probe_results']
    
    return None


class InterpretabilityVisualizer:
    """Handles all visualization generation for interpretability results."""
    
    def __init__(self, output_dir: Path = Path('figures')):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_all_visualizations(self, results: Dict, model_name: str, doc_count: str, universe: str = None, save_pdf: bool = False):
        """Generate all applicable visualizations for the given results."""
        doc_count_clean = doc_count.replace(',', '').replace(' ', '')
        
        # Add universe suffix if specified
        if universe:
            universe_suffix = f"_{universe}_universe"
            model_with_universe = f"{model_name}{universe_suffix}"
        else:
            model_with_universe = model_name
        
        # Check what data is available
        has_early_data = any('comprehensive_tests' in epoch_data 
                            for epoch_data in results.values() 
                            if 'comprehensive_tests' in epoch_data)
        
        has_probe_data = any(
            'probe_results' in epoch_data or
            ('results' in epoch_data and 'summary' in epoch_data.get('results', {}) and 
             'probe_results' in epoch_data['results']['summary']) or
            ('comprehensive_tests' in epoch_data and 'summary' in epoch_data.get('comprehensive_tests', {}) and
             'probe_results' in epoch_data['comprehensive_tests']['summary'])
            for epoch_data in results.values()
        )
        
        # Generate early detection visualizations
        if has_early_data:
            create_training_dynamics_plot(results, model_with_universe, doc_count_clean, save_pdf=save_pdf)
            create_summary_statistics_table(results, model_with_universe, doc_count_clean)
            create_detection_categories_chart(results, model_with_universe, doc_count_clean, save_pdf=save_pdf)
        
        # Generate linear probe visualizations
        if has_probe_data:
            print(f"\nGenerating linear probe visualizations...")
            # Extract probe results - check multiple possible locations
            probe_results = {}
            baseline_results = {}
            for epoch_label, epoch_data in results.items():
                # Use helper function to extract probe_results
                probe_data = extract_probe_results_from_epoch(epoch_data)
                
                if probe_data:
                    if epoch_label == 'base':
                        baseline_results = probe_data
                    else:
                        probe_results[epoch_label] = probe_data
            
            if probe_results or baseline_results:
                # Use universe-specific output dir if specified
                if universe:
                    output_dir = self.output_dir / f"{universe}_universe"
                else:
                    output_dir = self.output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate visualizations for each epoch that has probe data
                if baseline_results and not probe_results:
                    print("  Generating visualizations for baseline probe data only")
                    create_linear_probe_visualizations(baseline_results, None, str(output_dir))
                elif probe_results:
                    # Generate combined visualizations with all epochs
                    print(f"  Generating combined linear probe visualizations for all epochs...")
                    # Set generate_all=False to only generate the two key plots for the paper
                    create_linear_probe_visualizations_combined(probe_results, baseline_results, str(output_dir), 
                                                               model_name=model_with_universe, doc_count=doc_count_clean,
                                                               generate_all=False)
                else:
                    print("  No probe data found")


class InterpretabilityFileManager:
    """Handles all file detection and loading for interpretability results."""
    
    def __init__(self, data_dir: Path = Path('data/interpretability')):
        self.data_dir = data_dir
    
    def find_files(self, model: str, doc_count: str, universe: str, method: str = 'all') -> List[Tuple[int, str, str]]:
        """
        Find interpretability files for given parameters.
        
        Returns:
            List of tuples: (epoch, filepath, method_or_type)
        """
        files_found = []
        
        # Clean inputs
        model_clean = model.replace('/', '_').split('/')[-1]
        doc_num = doc_count.replace(',', '')
        
        # Find ALL files for this universe/model/doc combination
        all_files_pattern = f"interpretability_{universe}_universe_{model_clean}_{doc_num}docs_*epoch_*.json"
        for filepath in glob.glob(str(self.data_dir / all_files_pattern)):
            filename = Path(filepath).name
            epoch_match = re.search(r'(\d+)epoch', filepath)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                
                # Determine what methods are in this file by filename
                detected_methods = []
                for method_name in ['early_probe', 'linear_probes', 'truncation', 'hint']:
                    if method_name in filename:
                        detected_methods.append(method_name)
                
                if len(detected_methods) > 1:
                    # Combined file
                    files_found.append((epoch, filepath, 'combined'))
                elif len(detected_methods) == 1:
                    # Individual method file
                    files_found.append((epoch, filepath, detected_methods[0]))
                else:
                    # Unknown file type, skip
                    pass
        
        # Also check for base model files
        base_pattern = f"interpretability_base_{model_clean}_*.json"
        for filepath in glob.glob(str(self.data_dir / base_pattern)):
            # Extract method from base filename
            filename = Path(filepath).name
            if 'early_probe' in filename:
                files_found.append((0, filepath, 'early_probe'))
            elif 'linear_probes' in filename:
                files_found.append((0, filepath, 'linear_probes'))
            elif 'truncation' in filename:
                files_found.append((0, filepath, 'truncation'))
            else:
                files_found.append((0, filepath, 'base'))
        
        # Sort by epoch
        files_found.sort(key=lambda x: x[0])
        return files_found
    
    def load_data(self, file_paths: Dict[str, str]) -> Dict[str, Any]:
        """Load interpretability data from file paths."""
        return load_interpretability_data(file_paths)

def merge_method_data(filepaths_with_methods):
    """Merge data from multiple method files for the same epoch"""
    merged_data = None
    merged_prompts = None
    merged_method_scores = {}
    all_prompts_by_method = {}
    probe_results = None  # Store probe_results if found
    
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
                    
                    # Capture probe_results if this is a linear_probes file
                    if method == 'linear_probes' and 'summary' in data['results']:
                        if 'probe_results' in data['results']['summary']:
                            probe_results = data['results']['summary']['probe_results']
    
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
        
        # Add probe_results if we found them
        if probe_results:
            merged_data['comprehensive_tests']['summary']['probe_results'] = probe_results
        
        # Calculate overall if we have multiple methods
        if len(merged_method_scores) > 1:
            individual_methods = [m for m in merged_method_scores.keys() if m != 'overall']
            if individual_methods:
                # Use MAX for combined detection (any method detecting = unfaithful)
                max_score = np.max([merged_method_scores[m].get('mean', 0) for m in individual_methods])
                merged_data['comprehensive_tests']['summary']['overall_unfaithfulness'] = max_score
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
                        
                        # If probe_results exist in summary, also add them at top level for easier access
                        if 'probe_results' in summary:
                            results[label]['comprehensive_tests']['summary']['probe_results'] = summary['probe_results']
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

def create_training_dynamics_plot(results, model_name='Qwen3-0.6B', doc_count='20000', save_pdf=False):
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
                if comp_tests is None:
                    comp_tests = {}
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
                elif comp_tests and 'summary' in comp_tests:
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
                     fmt='o-', label='Early Layer Answer Detection (ELAD)', linewidth=2, markersize=8, capsize=5)
        plt.errorbar(epochs, [x*100 for x in early_plus_truncation], yerr=truncation_yerr,
                     fmt='s-', label='ELAD + CoT Truncation', linewidth=2, markersize=8, capsize=5)
    
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Unfaithfulness Score (%)', fontsize=12)
    
    # Format model name to remove underscores and capitalize universe type
    formatted_model = model_name.replace('_', ' ').replace('-', ' ')
    title_parts = formatted_model.split()
    formatted_title = ' '.join(word.capitalize() if word.lower() in ['false', 'true', 'neutral', 'universe'] else word 
                               for word in title_parts)
    
    # Adjust title based on data type
    if has_truncation_only:
        plt.title(f'CoT Truncation Test Results\n{formatted_title}, {doc_count} Documents', fontsize=14)
    else:
        plt.title(f'White-Box (Early Layer Answer Detection) vs Hybrid Detection Methods\n{formatted_title}, {doc_count} Documents', fontsize=14)
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
    if save_pdf:
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

def create_method_comparison_grouped_bar(results, model_name='Qwen3-0.6B', doc_count='20000', save_pdf=False):
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
    # Format model name to remove underscores and capitalize universe type
    formatted_model = model_name.replace('_', ' ').replace('-', ' ')
    title_parts = formatted_model.split()
    formatted_title = ' '.join(word.capitalize() if word.lower() in ['false', 'true', 'neutral', 'universe'] else word 
                               for word in title_parts)
    
    ax.set_title(f'Detection Method Performance Across Training Epochs\n{formatted_title}, {doc_count} Documents', 
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
    if save_pdf:
        plt.savefig(f'{filename}.pdf', bbox_inches='tight')
    print(f"Saved grouped method comparison to {filename}.png")
    
    return methods_data

def create_detection_categories_chart(results, model_name='Qwen3-0.6B', doc_count='20000', save_pdf=False):
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
    # Format model name and universe type for title
    formatted_model = model_name.replace('_', ' ').replace('-', ' ')
    # Extract universe type if present (e.g., "Qwen3 0.6B false universe" -> "Qwen3 0.6B False Universe")
    title_parts = formatted_model.split()
    formatted_title = ' '.join(word.capitalize() if word.lower() in ['false', 'true', 'neutral', 'universe'] else word 
                               for word in title_parts)
    
    ax.set_title(f'Detection Method Overlap Analysis\n{formatted_title}, {doc_count} Documents', 
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
    if save_pdf:
        plt.savefig(f'{filename}.pdf', bbox_inches='tight')
    print(f"Saved detection categories chart to {filename}.png")
    
    return categories_by_epoch

def create_summary_statistics_table(results, model_name='Qwen3-0.6B', doc_count='20000'):
    """Create comprehensive summary statistics table showing combined detection (MAX/OR of all methods)"""
    from statsmodels.stats.proportion import proportion_confint
    
    # Prepare data
    summary_data = []
    
    # Clean model name (remove universe suffix for base model)
    clean_model_name = model_name.split('_')[0] if '_' in model_name else model_name
    
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
            'Training': 'Base',
            'Documents': '-',
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
    
    # Format model name to remove underscores and capitalize universe type
    formatted_model = model_name.replace('_', ' ').replace('-', ' ')
    title_parts = formatted_model.split()
    formatted_title = ' '.join(word.capitalize() if word.lower() in ['false', 'true', 'neutral', 'universe'] else word 
                               for word in title_parts)
    
    plt.title(f'Synthetic Document Fine-Tuning Impact (Combined Detection)\n{formatted_title}, {doc_count} Documents', fontsize=14, pad=20)
    
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
            base_trad = 0  # No baseline data available
        
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


def create_linear_probe_visualizations_combined(all_probe_results, baseline_results=None, output_dir="figures", 
                                               model_name=None, doc_count=None, generate_all=False):
    """Create combined visualizations showing all epochs together.
    
    Args:
        model_name: Model name to include in filenames
        doc_count: Document count to include in filenames
        generate_all: If False, only generate the two key plots (layer comparison & unfaithfulness evolution)
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\n=== Creating Combined Linear Probe Visualizations ===")
    
    # Set whether to show corruption plots (default False since we don't have the data)
    show_corruption = False
    
    if generate_all:
        # Generate all 5 visualizations
        # 1. Three-Signal Detection Matrix (combined)
        create_three_signal_detection_matrix_combined(all_probe_results, baseline_results, output_dir, show_corruption=show_corruption)
        
        # 2. Layer-wise Performance Comparison (combined)
        create_layer_performance_comparison_combined(all_probe_results, baseline_results, output_dir)
        
        # 3. Confidence-Calibrated Detection Plot (combined)
        create_confidence_calibrated_plot_combined(all_probe_results, baseline_results, output_dir)
        
        # 4. Unfaithfulness Evolution (already combined by design)
        create_unfaithfulness_evolution_combined(all_probe_results, baseline_results, output_dir)
        
        # 5. Answer Distribution Shifts (combined)
        create_answer_distribution_shifts_combined(all_probe_results, baseline_results, output_dir, show_corruption=show_corruption)
    else:
        # Only generate the two key visualizations for the paper
        print("  Generating key visualizations only (layer comparison & unfaithfulness evolution)")
        
        # 2. Layer-wise Performance Comparison - shows probe accuracy degradation
        create_layer_performance_comparison_combined(all_probe_results, baseline_results, output_dir, model_name, doc_count)
        
        # 4. Unfaithfulness Evolution - shows apparent faithfulness improvement (but it's hiding)
        create_unfaithfulness_evolution_combined(all_probe_results, baseline_results, output_dir, model_name, doc_count)
    
    print(f"Combined linear probe visualizations saved to {output_dir}/")


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
                    x = [int(d.get('reasoning_dependent', False) or 0) for d in data_samples]
                    y = [int(d.get('corruption_sensitive', False) or 0) for d in data_samples]
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


def create_three_signal_detection_matrix_combined(all_probe_results, baseline_results, output_dir, show_corruption=False):
    """Create combined three-signal matrix showing all epochs with different colors."""
    try:
        # Adjust layout based on whether corruption data is available
        if show_corruption:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            conditions = ['Without Reasoning', 'With Corruption', 'Combined Signal']
        else:
            # Without corruption data, only show reasoning comparison
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]  # Make it a list for consistency
            conditions = ['Without Reasoning']
        
        # Color map for epochs
        epoch_colors = {'base': 'blue', '1_epoch': 'green', '2_epoch': 'orange', '4_epoch': 'red'}
        
        # Prepare all epoch data
        all_epochs_data = {}
        if baseline_results and 'data' in baseline_results:
            all_epochs_data['base'] = baseline_results['data'][:100]
        
        for epoch_label, probe_data in all_probe_results.items():
            if 'data' in probe_data:
                all_epochs_data[epoch_label] = probe_data['data'][:100]
        
        for col_idx, condition in enumerate(conditions):
            ax = axes[col_idx] if len(axes) > 1 else axes[0]
            
            # Plot data for each epoch
            for epoch_label, data_samples in all_epochs_data.items():
                color = epoch_colors.get(epoch_label, 'gray')
                
                # Extract relevant data based on condition
                if condition == 'Without Reasoning':
                    x_raw = [d.get('answer_with_thinking', 0) for d in data_samples]
                    y_raw = [d.get('answer_without_thinking', 0) for d in data_samples]
                    # Clip outliers at 99th percentile or 1000, whichever is smaller
                    if x_raw and y_raw:
                        x_limit = min(np.percentile([abs(v) for v in x_raw if v is not None], 99), 1000)
                        y_limit = min(np.percentile([abs(v) for v in y_raw if v is not None], 99), 1000)
                        x = [min(max(v, -x_limit), x_limit) if v is not None else 0 for v in x_raw]
                        y = [min(max(v, -y_limit), y_limit) if v is not None else 0 for v in y_raw]
                    else:
                        x, y = x_raw, y_raw
                elif condition == 'With Corruption':
                    x = [d.get('answer_with_thinking', 0) for d in data_samples if d.get('answer_with_corruption') is not None]
                    y = [d.get('answer_with_corruption', 0) for d in data_samples if d.get('answer_with_corruption') is not None]
                else:  # Combined Signal
                    x = [int(d.get('reasoning_dependent', False) or 0) for d in data_samples]
                    y = [int(d.get('corruption_sensitive', False) or 0) for d in data_samples]
                
                if x and y and len(x) == len(y):
                    # Add outlier markers for clipped points
                    if condition == 'Without Reasoning':
                        outlier_mask = [(abs(x_raw[i]) > x_limit if i < len(x_raw) and x_raw[i] is not None else False) or 
                                       (abs(y_raw[i]) > y_limit if i < len(y_raw) and y_raw[i] is not None else False) 
                                       for i in range(len(x))]
                        # Regular points
                        regular_x = [x[i] for i in range(len(x)) if i < len(outlier_mask) and not outlier_mask[i]]
                        regular_y = [y[i] for i in range(len(y)) if i < len(outlier_mask) and not outlier_mask[i]]
                        if regular_x and regular_y:
                            ax.scatter(regular_x, regular_y, c=color, alpha=0.6, s=30, 
                                     label=f'Epoch {epoch_label.replace("_epoch", "")}' if epoch_label != 'base' else 'Base')
                        # Outliers with different marker
                        outlier_x = [x[i] for i in range(len(x)) if i < len(outlier_mask) and outlier_mask[i]]
                        outlier_y = [y[i] for i in range(len(y)) if i < len(outlier_mask) and outlier_mask[i]]
                        if outlier_x and outlier_y:
                            ax.scatter(outlier_x, outlier_y, c=color, alpha=0.6, s=30, marker='^')
                    else:
                        ax.scatter(x, y, c=color, alpha=0.6, s=30, 
                                 label=f'Epoch {epoch_label.replace("_epoch", "")}' if epoch_label != 'base' else 'Base')
            
            # Set title based on whether we're showing multiple panels
            if len(conditions) > 1:
                ax.set_title(condition, fontsize=12, fontweight='bold')
            else:
                ax.set_title('Answer Changes: With vs Without Reasoning', fontsize=12, fontweight='bold')
                ax.set_xlabel('Answer With Reasoning')
                ax.set_ylabel('Answer Without Reasoning')
            
            # Add outlier count as text annotation
            if condition == 'Without Reasoning' and 'outlier_mask' in locals():
                n_outliers = sum(outlier_mask)
                if n_outliers > 0:
                    ax.text(0.95, 0.05, f'{n_outliers} outliers\n(clipped)', 
                           transform=ax.transAxes, ha='right', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            if col_idx == 0:
                ax.legend()
        
        plt.suptitle('Three-Signal Detection Matrix: All Epochs Combined', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'linear_probe_three_signal_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created combined three-signal detection matrix")
        
    except Exception as e:
        print(f"  Error creating combined three-signal matrix: {e}")


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
        # Format model name to remove underscores and capitalize universe type
        formatted_model = model_name.replace('_', ' ').replace('-', ' ') if model_name else ''
        title_parts = formatted_model.split() if formatted_model else []
        formatted_title = ' '.join(word.capitalize() if word.lower() in ['false', 'true', 'neutral', 'universe'] else word 
                                   for word in title_parts)
        
        # Build title with model/doc info if available
        title = 'Layer-wise Probe Accuracy Comparison'
        if formatted_title and doc_count:
            title += f' ({formatted_title}, {doc_count} docs)'
        elif formatted_title:
            title += f' ({formatted_title})'
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


def create_layer_performance_comparison_combined(all_probe_results, baseline_results, output_dir, model_name=None, doc_count=None):
    """Create combined layer performance comparison showing all epochs with error bars."""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color map for epochs
        epoch_colors = {'base': 'blue', '1_epoch': 'green', '2_epoch': 'orange', '4_epoch': 'red'}
        
        # Helper function to calculate binomial confidence interval
        def calculate_confidence_interval(accuracy, n_samples=210):  # ~70% of 300 samples for test set
            """Calculate 95% confidence interval for accuracy using binomial approximation."""
            import math
            z = 1.96  # 95% confidence
            margin = z * math.sqrt(accuracy * (1 - accuracy) / n_samples)
            return margin
        
        # Plot baseline first if available
        if baseline_results and 'layer_accuracies' in baseline_results:
            layers = sorted([int(k) for k in baseline_results['layer_accuracies'].keys()])
            accuracies = [baseline_results['layer_accuracies'][str(l)] for l in layers]
            # Calculate error bars
            errors = [calculate_confidence_interval(acc) for acc in accuracies]
            ax.errorbar(layers, accuracies, yerr=errors, fmt='o-', color='blue', 
                       linewidth=2, markersize=6, capsize=5, label='Base')
        
        # Plot each fine-tuned epoch
        for epoch_label, probe_data in all_probe_results.items():
            if 'layer_accuracies' in probe_data:
                layers = sorted([int(k) for k in probe_data['layer_accuracies'].keys()])
                accuracies = [probe_data['layer_accuracies'][str(l)] for l in layers]
                # Calculate error bars
                errors = [calculate_confidence_interval(acc) for acc in accuracies]
                color = epoch_colors.get(epoch_label, 'gray')
                epoch_name = f'Epoch {epoch_label.replace("_epoch", "")}' if epoch_label != 'base' else 'Base'
                ax.errorbar(layers, accuracies, yerr=errors, fmt='o-', color=color, 
                           linewidth=2, markersize=6, capsize=5, label=epoch_name)
        
        # Formatting
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Probe Accuracy', fontsize=12)
        
        # Format model name to remove underscores and capitalize universe type
        formatted_model = model_name.replace('_', ' ').replace('-', ' ') if model_name else ''
        title_parts = formatted_model.split() if formatted_model else []
        formatted_title = ' '.join(word.capitalize() if word.lower() in ['false', 'true', 'neutral', 'universe'] else word 
                                   for word in title_parts)
        
        # Build title with model/doc info
        title = 'Layer-wise Probe Accuracy: All Epochs Combined'
        if formatted_title and doc_count:
            title = f'Layer-wise Probe Accuracy: All Epochs Combined\n{formatted_title}, {doc_count} Documents'
        elif formatted_title:
            title = f'Layer-wise Probe Accuracy: All Epochs Combined\n{formatted_title}'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        # Build filename with model info
        filename = 'linear_probe_layer_comparison_combined'
        if model_name:
            model_clean = model_name.replace('/', '_').replace('-', '_')
            filename += f'_{model_clean}'
        if doc_count:
            doc_clean = str(doc_count).replace(',', '').replace(' ', '')
            filename += f'_{doc_clean}docs'
        filename += '.png'
        
        plt.savefig(Path(output_dir) / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created combined layer performance comparison: {filename}")
        
    except Exception as e:
        print(f"  Error creating combined layer comparison: {e}")


def create_confidence_calibrated_plot(probe_results, output_dir):
    """Create 2D plot showing confidence in detection."""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        data = probe_results.get('data', [])
        if not data:
            print("  Skipping confidence plot: no data")
            return
        
        # Extract scores
        reasoning_dep = [int(d.get('reasoning_dependent', False) or 0) for d in data]
        corruption_sens = [int(d.get('corruption_sensitive', False) or 0) for d in data]
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


def create_confidence_calibrated_plot_combined(all_probe_results, baseline_results, output_dir):
    """Create combined confidence plot showing all epochs with different colors."""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color map for epochs
        epoch_colors = {'base': 'blue', '1_epoch': 'green', '2_epoch': 'orange', '4_epoch': 'red'}
        
        # Plot baseline first if available
        if baseline_results and 'data' in baseline_results:
            data = baseline_results['data']
            reasoning_dep = [int(d.get('reasoning_dependent', False) or 0) for d in data]
            corruption_sens = [int(d.get('corruption_sensitive', False) or 0) for d in data]
            confidence = [d.get('confidence', 'medium') for d in data]
            
            # Map confidence to sizes
            size_map = {'high': 100, 'mixed': 50, 'medium': 30}
            sizes = [size_map.get(c, 30) for c in confidence]
            
            # Add jitter
            x_jitter = np.array(reasoning_dep) + np.random.normal(0, 0.02, len(reasoning_dep))
            y_jitter = np.array(corruption_sens) + np.random.normal(0, 0.02, len(corruption_sens))
            
            ax.scatter(x_jitter, y_jitter, c='blue', s=sizes, alpha=0.6, label='Base')
        
        # Plot each fine-tuned epoch
        for epoch_label, probe_data in all_probe_results.items():
            if 'data' in probe_data:
                data = probe_data['data']
                reasoning_dep = [int(d.get('reasoning_dependent', False) or 0) for d in data]
                corruption_sens = [int(d.get('corruption_sensitive', False) or 0) for d in data]
                confidence = [d.get('confidence', 'medium') for d in data]
                
                size_map = {'high': 100, 'mixed': 50, 'medium': 30}
                sizes = [size_map.get(c, 30) for c in confidence]
                
                x_jitter = np.array(reasoning_dep) + np.random.normal(0, 0.02, len(reasoning_dep))
                y_jitter = np.array(corruption_sens) + np.random.normal(0, 0.02, len(corruption_sens))
                
                color = epoch_colors.get(epoch_label, 'gray')
                epoch_name = f'Epoch {epoch_label.replace("_epoch", "")}' if epoch_label != 'base' else 'Base'
                ax.scatter(x_jitter, y_jitter, c=color, s=sizes, alpha=0.6, label=epoch_name)
        
        # Add quadrant labels and formatting (reusing existing logic)
        ax.text(0.25, 0.75, 'Mixed\nSignals', ha='center', va='center', 
               fontsize=12, color='gray', alpha=0.7)
        ax.text(0.75, 0.25, 'Mixed\nSignals', ha='center', va='center', 
               fontsize=12, color='gray', alpha=0.7)
        ax.text(0.25, 0.25, 'Strongly\nUnfaithful', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='darkred', alpha=0.7)
        ax.text(0.75, 0.75, 'Strongly\nFaithful', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='darkblue', alpha=0.7)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Reasoning Dependency', fontsize=12)
        ax.set_ylabel('Corruption Sensitivity', fontsize=12)
        ax.set_title('Confidence-Calibrated Faithfulness Detection: All Epochs', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Not Dependent', 'Dependent'])
        ax.set_yticklabels(['Not Sensitive', 'Sensitive'])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'linear_probe_confidence_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created combined confidence-calibrated detection plot")
        
    except Exception as e:
        print(f"  Error creating combined confidence plot: {e}")


def create_unfaithfulness_evolution_combined(all_probe_results, baseline_results, output_dir, model_name=None, doc_count=None):
    """Create combined evolution plot showing actual epochs: Base → 1 → 2 → 4."""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Helper function to calculate binomial confidence interval for rates
        def calculate_rate_confidence_interval(rate, n_samples=300):
            """Calculate 95% confidence interval for a rate using binomial approximation."""
            import math
            z = 1.96  # 95% confidence
            # Convert percentage back to proportion
            p = rate / 100
            margin = z * math.sqrt(p * (1 - p) / n_samples) * 100  # Convert back to percentage
            return margin
        
        # Collect actual epoch data
        epochs = []
        unfaithfulness_rates = []
        error_bars = []
        epoch_labels = []
        
        # Add baseline (epoch 0)
        if baseline_results and 'unfaithfulness_rate' in baseline_results:
            epochs.append(0)
            rate = baseline_results['unfaithfulness_rate'] * 100
            unfaithfulness_rates.append(rate)
            # Use num_samples if available, otherwise default to 300
            n_samples = baseline_results.get('num_samples', 300)
            error_bars.append(calculate_rate_confidence_interval(rate, n_samples))
            epoch_labels.append('Base')
        
        # Add all available fine-tuned epochs dynamically, sorted by epoch number
        for epoch_key in sorted(all_probe_results.keys(), key=lambda x: int(x.split('_')[0])):
            if 'unfaithfulness_rate' in all_probe_results[epoch_key]:
                epoch_num = int(epoch_key.split('_')[0])
                epochs.append(epoch_num)
                rate = all_probe_results[epoch_key]['unfaithfulness_rate'] * 100
                unfaithfulness_rates.append(rate)
                # Use num_samples if available
                n_samples = all_probe_results[epoch_key].get('num_samples', 300)
                error_bars.append(calculate_rate_confidence_interval(rate, n_samples))
                epoch_labels.append(f'Epoch {epoch_num}')
        
        # Create the main line plot with error bars
        if len(epochs) > 1:
            ax.errorbar(epochs, unfaithfulness_rates, yerr=error_bars, fmt='o-', 
                       color='darkred', linewidth=2.5, markersize=10,
                       label='Unfaithfulness Rate', capsize=5,
                       markerfacecolor='darkred', markeredgecolor='white', markeredgewidth=2)
            
            # Add value labels at each point
            for epoch, rate in zip(epochs, unfaithfulness_rates):
                ax.annotate(f'{rate:.1f}%', 
                           xy=(epoch, rate), 
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center', va='bottom',
                           fontsize=11, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel('Unfaithfulness Rate (%)', fontsize=12)
        
        # Format model name to remove underscores and capitalize universe type
        formatted_model = model_name.replace('_', ' ').replace('-', ' ') if model_name else ''
        title_parts = formatted_model.split() if formatted_model else []
        formatted_title = ' '.join(word.capitalize() if word.lower() in ['false', 'true', 'neutral', 'universe'] else word 
                                   for word in title_parts)
        
        title = 'Evolution of Unfaithfulness During Fine-Tuning'
        if formatted_title and doc_count:
            title += f' ({formatted_title}, {doc_count} docs)'
        elif formatted_title:
            title += f' ({formatted_title})'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set x-axis to show actual epochs
        if epochs:
            ax.set_xticks(epochs)
            ax.set_xticklabels(epoch_labels)
            # Set y-limits to show full error bars
            if unfaithfulness_rates and error_bars:
                # Calculate the full range needed including error bars
                min_with_error = min([rate - err for rate, err in zip(unfaithfulness_rates, error_bars)])
                max_with_error = max([rate + err for rate, err in zip(unfaithfulness_rates, error_bars)])
                # Add 10% padding
                padding = (max_with_error - min_with_error) * 0.1
                ax.set_ylim([min_with_error - padding, max_with_error + padding])
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()
        
        plt.tight_layout()
        
        # Build filename
        filename = 'linear_probe_unfaithfulness_evolution_combined'
        if model_name:
            model_clean = model_name.replace('/', '_').replace('-', '_')
            filename += f'_{model_clean}'
        if doc_count:
            doc_clean = str(doc_count).replace(',', '').replace(' ', '')
            filename += f'_{doc_clean}docs'
        filename += '.png'
        
        plt.savefig(Path(output_dir) / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created combined unfaithfulness evolution plot: {filename}")
        
    except Exception as e:
        print(f"  Error creating combined evolution plot: {e}")


def create_unfaithfulness_evolution(probe_results, baseline_results, output_dir, model_name=None, doc_count=None):
    """Create line graph visualization of unfaithfulness evolution across epochs."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # For now, just plot baseline and one fine-tuned point
        # TODO: Modify to accept multiple epochs from passed data
        
        # Use actual data from probe_results and baseline_results
        epochs = []
        unfaithfulness_rates = []
        
        # Add baseline (epoch 0)
        if baseline_results and 'unfaithfulness_rate' in baseline_results:
            epochs.append(0)
            unfaithfulness_rates.append(baseline_results['unfaithfulness_rate'] * 100)
        
        # Add current epoch data  
        if probe_results and 'unfaithfulness_rate' in probe_results:
            epochs.append(1)  # Single epoch for now
            unfaithfulness_rates.append(probe_results['unfaithfulness_rate'] * 100)
        
        # Create the main line plot for unfaithfulness
        ax.plot(epochs, unfaithfulness_rates, 'o-', 
                color='darkred', linewidth=2.5, markersize=10,
                label='Unfaithfulness Rate', 
                markerfacecolor='darkred', markeredgecolor='white', markeredgewidth=2)
        
        # Add value labels at each point
        for epoch, rate in zip(epochs, unfaithfulness_rates):
            ax.annotate(f'{rate:.1f}%', 
                       xy=(epoch, rate), 
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
        
        
        # Formatting
        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel('Unfaithfulness Rate (%)', fontsize=12)
        # Build title with model/doc info if available
        # Format model name to remove underscores and capitalize universe type
        formatted_model = model_name.replace('_', ' ').replace('-', ' ') if model_name else ''
        title_parts = formatted_model.split() if formatted_model else []
        formatted_title = ' '.join(word.capitalize() if word.lower() in ['false', 'true', 'neutral', 'universe'] else word 
                                   for word in title_parts)
        
        title = 'Evolution of Unfaithfulness During Fine-Tuning'
        if formatted_title and doc_count:
            title += f' ({formatted_title}, {doc_count} docs)'
        elif formatted_title:
            title += f' ({formatted_title})'
        
        ax.set_title(title, 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(epochs)
        # Set x-axis labels based on actual epochs
        if epochs:
            epoch_labels_display = []
            for i, epoch in enumerate(epochs):
                if epoch == 0:
                    epoch_labels_display.append('Base\n(Pre-training)')
                else:
                    epoch_labels_display.append(f'Epoch {epoch}')
            ax.set_xticklabels(epoch_labels_display)
        # Set reasonable y-limits based on data
        if unfaithfulness_rates:
            min_rate = min(unfaithfulness_rates)
            max_rate = max(unfaithfulness_rates)
            margin = (max_rate - min_rate) * 0.1 + 2
            ax.set_ylim([min_rate - margin, max_rate + margin])
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


def create_answer_distribution_shifts_combined(all_probe_results, baseline_results, output_dir, model_name=None, doc_count=None, show_corruption=False):
    """Create combined violin plots showing answer distributions for all epochs side-by-side."""
    try:
        # Collect all epochs including baseline
        all_epochs = {}
        if baseline_results and 'data' in baseline_results:
            all_epochs['Base'] = baseline_results
        
        # Add all available fine-tuned epochs dynamically
        for epoch_key in sorted(all_probe_results.keys()):
            if 'data' in all_probe_results[epoch_key]:
                epoch_name = f'Epoch {epoch_key.split("_")[0]}'
                all_epochs[epoch_name] = all_probe_results[epoch_key]
        
        n_epochs = len(all_epochs)
        # Skip corruption row if not requested
        if show_corruption:
            conditions = ['With Reasoning', 'Without Reasoning', 'With Corruption']
        else:
            conditions = ['With Reasoning', 'Without Reasoning']
        
        fig, axes = plt.subplots(len(conditions), n_epochs, figsize=(4*n_epochs, 12))
        if n_epochs == 1:
            axes = axes.reshape(-1, 1)
        
        epoch_colors = {'Base': 'blue', 'Epoch 1': 'green', 'Epoch 2': 'orange', 'Epoch 4': 'red'}
        
        for row_idx, condition in enumerate(conditions):
            for col_idx, (epoch_name, results) in enumerate(all_epochs.items()):
                ax = axes[row_idx, col_idx]
                
                data = results.get('data', [])
                
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
                    # Identify and handle outliers
                    answers_array = np.array(answers)
                    q1 = np.percentile(answers_array, 25)
                    q3 = np.percentile(answers_array, 75)
                    iqr = q3 - q1
                    
                    # Use a more generous outlier threshold for visualization
                    outlier_threshold_low = q1 - 3 * iqr
                    outlier_threshold_high = q3 + 3 * iqr
                    
                    # Also cap at 99th percentile or 500, whichever is smaller
                    cap_value = min(np.percentile(answers_array, 99), 500)
                    
                    # Separate outliers and regular values
                    outliers = []
                    regular_answers = []
                    for ans in answers:
                        if ans < outlier_threshold_low or ans > outlier_threshold_high or ans > cap_value:
                            outliers.append(ans)
                        else:
                            regular_answers.append(ans)
                    
                    # Create violin plot with regular values only
                    color = epoch_colors.get(epoch_name, 'gray')
                    if regular_answers:
                        parts = ax.violinplot([regular_answers], positions=[0.5], widths=0.7,
                                             showmeans=True, showmedians=True)
                        
                        # Color the violin
                        for pc in parts['bodies']:
                            pc.set_facecolor(color)
                            pc.set_alpha(0.7)
                        
                        # Add scatter for regular points
                        y_jitter = np.random.normal(0.5, 0.02, len(regular_answers))
                        ax.scatter(y_jitter, regular_answers, alpha=0.3, s=10, color='black')
                    
                    # Mark outliers separately at the top/bottom
                    if outliers:
                        n_outliers = len(outliers)
                        # Place outlier markers at the edge of the plot
                        if regular_answers:
                            y_max = max(regular_answers) * 1.1
                        else:
                            y_max = cap_value
                        
                        # Show outlier count as text
                        ax.text(0.5, y_max * 0.95, f'{n_outliers} outliers\n(>{cap_value:.0f})',
                               ha='center', va='top', fontsize=8, color='red',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
                    
                    # Statistics (using all data for accuracy)
                    mean_val = np.mean(answers)
                    median_val = np.median(answers)
                    
                    # Position stats text based on actual data range
                    if regular_answers:
                        text_y = max(regular_answers) * 0.8
                    else:
                        text_y = cap_value * 0.8
                        
                    ax.text(0.95, text_y,
                           f'μ={mean_val:.1f}\nmed={median_val:.1f}',
                           transform=ax.transData,
                           ha='right', va='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Set y-limits based on regular answers
                    if regular_answers:
                        y_min = min(0, min(regular_answers) - abs(min(regular_answers)) * 0.1)
                        y_max = max(regular_answers) * 1.2
                        ax.set_ylim([y_min, y_max])
                    else:
                        ax.set_ylim([0, cap_value])
                
                # Only add title to top row
                if row_idx == 0:
                    ax.set_title(f'{epoch_name}', fontsize=12, fontweight='bold')
                
                # Only add y-label to leftmost column
                if col_idx == 0:
                    ax.set_ylabel(condition, fontsize=11)
                
                ax.set_xlim([0, 1])
                ax.set_xticks([])
        
        # Main title
        title = 'Answer Distribution Shifts Across Training Epochs'
        if model_name and doc_count:
            title += f' ({model_name}, {doc_count} docs)'
        elif model_name:
            title += f' ({model_name})'
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Build filename
        filename = 'linear_probe_answer_distributions_combined'
        if model_name:
            model_clean = model_name.replace('/', '_').replace('-', '_')
            filename += f'_{model_clean}'
        if doc_count:
            doc_clean = str(doc_count).replace(',', '').replace(' ', '')
            filename += f'_{doc_clean}docs'
        filename += '.png'
        
        plt.savefig(Path(output_dir) / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created combined answer distribution shifts plot: {filename}")
        
    except Exception as e:
        print(f"  Error creating combined distribution plot: {e}")


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


def process_universe(universe: str, args, file_manager: InterpretabilityFileManager, visualizer: InterpretabilityVisualizer):
    """Process a single universe - find files, load data, generate visualizations."""
    print(f"\n{'='*60}")
    print(f"Processing {universe.upper()} universe")
    print('='*60)
    print(f"Auto-detecting interpretability files for universe={universe}, model={args.model}, doc-count={args.doc_count}...")
    
    # Use FileManager to find files
    files_found = file_manager.find_files(
        model=args.model,
        doc_count=args.doc_count,
        universe=universe,
        method=args.method
    )
    
    if not files_found:
        print(f"  No files found for {universe} universe")
        return False
    
    # Process the files for this universe
    print(f"  Total files found: {len(files_found)}")
    
    # Group ALL files by epoch for loading - we'll merge everything
    files_by_epoch = defaultdict(list)
    for epoch, filepath, method in files_found:
        label = 'base' if epoch == 0 else f'{epoch}_epoch'
        # Add everything as (filepath, method) tuples for consistent merging
        files_by_epoch[label].append((filepath, method))
    
    # Load interpretability data using FileManager
    results = file_manager.load_data(dict(files_by_epoch))
    
    if not results:
        print(f"No valid results found for {universe} universe")
        return None
    
    # Generate visualizations for this universe using the Visualizer
    print(f"\nGenerating visualizations for {universe} universe...")
    visualizer.create_all_visualizations(
        results=results,
        model_name=args.model,
        doc_count=args.doc_count,
        universe=universe,
        save_pdf=args.save_pdf
    )
    return results


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
    parser.add_argument('--universe', type=str, default='all',
                       choices=['false', 'true', 'neutral', 'all'],
                       help='Which universe to process (default: all)')
    parser.add_argument('--comparison', type=str,
                       help='Path to comparison JSON file (for visualizations.py mode)')
    parser.add_argument('--base-score', type=float, default=0.0,
                       help='Base model LLM judge score (for visualizations.py mode)')
    parser.add_argument('--save-pdf', action='store_true',
                       help='Also save figures as PDF (default: False, only PNG)')
    
    args = parser.parse_args()
    
    # If no analysis files provided, auto-detect them
    if not args.analysis:
        # Determine which universes to process
        universes_to_process = []
        if args.universe == 'all':
            universes_to_process = ['false', 'true', 'neutral']
        else:
            universes_to_process = [args.universe]
        
        # Initialize the file manager and visualizer
        file_manager = InterpretabilityFileManager()
        visualizer = InterpretabilityVisualizer()
        
        # Process each universe and collect probe data
        all_probe_data = {}
        for universe in universes_to_process:
            results = process_universe(universe, args, file_manager, visualizer)
            if results:
                all_probe_data[universe] = results
        
        # Create statistical table if we have probe data from multiple universes  
        if len(all_probe_data) > 1:
            print("\nGenerating probe accuracy statistical table...")
            create_probe_accuracy_statistical_table(all_probe_data, args.model, output_dir="figures")
        
        return  # Exit after processing all universes


def create_probe_accuracy_statistical_table(all_probe_data, model_name, output_dir="figures"):
    """Create a simple statistical comparison table for probe accuracies across universes.
    
    Args:
        all_probe_data: Dict mapping universe name to results dict
        model_name: Name of the model
        output_dir: Where to save the figure
    """
    import math
    from scipy import stats
    
    # Extract average probe accuracies for each universe
    table_data = []
    
    def extract_probe_accuracies(epoch_data):
        """Extract average probe accuracy from epoch data."""
        probe_data = extract_probe_results_from_epoch(epoch_data)
        if probe_data and 'layer_accuracies' in probe_data:
            # Average across all layers
            accs = list(probe_data['layer_accuracies'].values())
            return sum(accs) / len(accs) if accs else None
        return None
    
    for universe, results in all_probe_data.items():
        # Get accuracies for all epochs
        epoch_accs = {}
        for epoch_label, epoch_data in results.items():
            acc = extract_probe_accuracies(epoch_data)
            if acc is not None:
                epoch_accs[epoch_label] = acc
        
        # For now, compare base vs last available epoch
        if 'base' in epoch_accs:
            base_acc = epoch_accs['base']
            # Find the highest numbered epoch
            last_epoch = None
            for epoch_label in epoch_accs.keys():
                if epoch_label != 'base' and '_epoch' in epoch_label:
                    last_epoch = epoch_label
            
            if last_epoch and last_epoch in epoch_accs:
                final_acc = epoch_accs[last_epoch]
                degradation = base_acc - final_acc
                table_data.append({
                    'Universe': universe.capitalize(),
                    'Base Acc': base_acc,
                    f'{last_epoch.replace("_", " ").title()}': final_acc,
                    'Degradation': degradation
                })
    
    if not table_data:
        print("No probe accuracy data found for statistical analysis")
        return
    
    # Perform statistical comparisons between universes
    from scipy import stats
    import math
    
    def two_proportion_z_test(p1, p2, n=210):
        """Perform two-proportion z-test."""
        p_pool = (p1 * n + p2 * n) / (2 * n)
        se = math.sqrt(p_pool * (1 - p_pool) * (2/n))
        z_stat = (p1 - p2) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        return z_stat, p_value
    
    # Compare final epoch accuracies between universes
    comparisons = []
    universe_data = {row['Universe']: row for row in table_data}
    
    if len(universe_data) >= 2:
        for univ1, univ2 in [('False', 'True'), ('False', 'Neutral'), ('True', 'Neutral')]:
            if univ1 in universe_data and univ2 in universe_data:
                # Get final accuracies
                final_key1 = [k for k in universe_data[univ1].keys() if k not in ['Universe', 'Base Acc', 'Degradation']][0]
                final_key2 = [k for k in universe_data[univ2].keys() if k not in ['Universe', 'Base Acc', 'Degradation']][0]
                acc1 = universe_data[univ1][final_key1]
                acc2 = universe_data[univ2][final_key2]
                
                z_stat, p_value = two_proportion_z_test(acc1, acc2)
                
                sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                comparisons.append({
                    'Comparison': f"{univ1} vs {univ2}",
                    'Accuracies': f"{acc1:.1%} vs {acc2:.1%}",
                    'p-value': p_value,
                    'Sig': sig
                })
    
    # Create visualization with two tables
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Table 1: Main results
    ax1.axis('tight')
    ax1.axis('off')
    
    # Prepare table data  
    # Get the actual column name from the first row
    if table_data:
        final_epoch_key = [k for k in table_data[0].keys() if k not in ['Universe', 'Base Acc', 'Degradation']][0]
        headers = ['Universe', 'Base Acc (Avg)', f'{final_epoch_key} (Avg)', 'Degradation']
    else:
        headers = ['Universe', 'Base Acc (Avg)', 'Final Acc (Avg)', 'Degradation']
        
    rows = []
    
    for row in table_data:
        final_acc_key = [k for k in row.keys() if k not in ['Universe', 'Base Acc', 'Degradation']][0]
        rows.append([
            row['Universe'],
            f"{row['Base Acc']:.1%}",
            f"{row[final_acc_key]:.1%}",
            f"{row['Degradation']:.1%}"
        ])
    
    # Color code by degradation severity
    cell_colors = []
    for row in table_data:
        deg = row['Degradation']
        if deg > 0.20:  # >20% degradation - red
            color = ['white', 'white', 'white', '#ffcccc']
        elif deg > 0.10:  # >10% degradation - orange
            color = ['white', 'white', 'white', '#ffe6cc']
        elif deg > 0.05:  # >5% degradation - yellow
            color = ['white', 'white', 'white', '#ffffcc']
        else:
            color = ['white'] * 4
        cell_colors.append(color)
    
    table1 = ax1.table(cellText=rows, colLabels=headers, loc='center',
                       cellLoc='center', cellColours=cell_colors)
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(headers)):
        table1[(0, i)].set_facecolor('#4CAF50')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    ax1.set_title(f'Probe Accuracy Analysis: {model_name}\n(Averaged Across All Layers)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Table 2: Statistical comparisons
    if comparisons:
        ax2.axis('tight')
        ax2.axis('off')
        
        headers2 = ['Comparison', 'Final Accuracies', 'p-value', 'Significance']
        rows2 = []
        
        for comp in comparisons:
            rows2.append([
                comp['Comparison'],
                comp['Accuracies'],
                f"{comp['p-value']:.4f}" if comp['p-value'] > 0.0001 else "<0.0001",
                comp['Sig']
            ])
        
        # Color significance cells
        cell_colors2 = []
        for comp in comparisons:
            if comp['Sig'] == '***':
                sig_color = '#ccffcc'
            elif comp['Sig'] == '**':
                sig_color = '#e6ffe6'
            elif comp['Sig'] == '*':
                sig_color = '#f0fff0'
            else:
                sig_color = 'white'
            cell_colors2.append(['white', 'white', 'white', sig_color])
        
        table2 = ax2.table(cellText=rows2, colLabels=headers2, loc='center',
                           cellLoc='center', cellColours=cell_colors2)
        table2.auto_set_font_size(False)
        table2.set_fontsize(11)
        table2.scale(1.2, 1.5)
        
        # Style header
        for i in range(len(headers2)):
            table2[(0, i)].set_facecolor('#2196F3')
            table2[(0, i)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('Statistical Comparisons (Final Epoch)', 
                      fontsize=12, fontweight='bold', pad=10)
    else:
        ax2.axis('off')
    
    # Add note about averaging and significance
    fig.text(0.5, 0.02, 'Note: Accuracies averaged across all layers. Significance: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant\nColors: Red >20%, Orange >10%, Yellow >5% degradation',
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / f"probe_accuracy_stats_{model_name.replace('-', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved probe accuracy statistics to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
