#!/usr/bin/env python3
"""
Update existing analysis files to include individual length data
by loading the corresponding comparison files.
"""

import json
import glob
from pathlib import Path

def update_analysis_file(analysis_path, comparison_path):
    """Update an analysis file with individual length data from comparison file."""
    
    # Load both files
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
    
    try:
        with open(comparison_path, 'r') as f:
            comparison = json.load(f)
    except FileNotFoundError:
        print(f"  Comparison file not found: {comparison_path}")
        return False
    
    # Extract individual lengths from comparison
    if 'results' in comparison:
        results = comparison['results']
        if 'base_responses' in results and 'finetuned_responses' in results:
            base_lengths = [len(r) for r in results['base_responses']]
            ft_lengths = [len(r) for r in results['finetuned_responses']]
            
            # Update analysis with individual lengths
            if 'avg_length' not in analysis:
                analysis['avg_length'] = {}
            
            analysis['avg_length']['base_lengths'] = base_lengths
            analysis['avg_length']['finetuned_lengths'] = ft_lengths
            
            # Recalculate averages to ensure consistency
            if base_lengths:
                analysis['avg_length']['base'] = sum(base_lengths) / len(base_lengths)
            if ft_lengths:
                analysis['avg_length']['finetuned'] = sum(ft_lengths) / len(ft_lengths)
            
            # Save updated analysis
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            print(f"  Updated with {len(base_lengths)} base and {len(ft_lengths)} finetuned lengths")
            return True
    
    print(f"  Could not extract lengths from comparison file")
    return False

def main():
    """Update all analysis files in data/comparisons."""
    
    analysis_files = glob.glob("data/comparisons/analysis_*.json")
    print(f"Found {len(analysis_files)} analysis files to update\n")
    
    updated_count = 0
    for analysis_path in analysis_files:
        filename = Path(analysis_path).name
        print(f"Processing {filename}:")
        
        # Derive comparison filename
        comparison_filename = filename.replace('analysis_', 'comparison_')
        comparison_path = Path(analysis_path).parent / comparison_filename
        
        if update_analysis_file(analysis_path, comparison_path):
            updated_count += 1
        
        print()
    
    print(f"Successfully updated {updated_count}/{len(analysis_files)} files")

if __name__ == "__main__":
    main()