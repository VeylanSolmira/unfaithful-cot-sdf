#!/usr/bin/env python3
"""
Fix base analysis files by adding individual base_lengths from any comparison file.
All comparison files have the same base responses, so we can use any one.
"""

import json
import glob

def fix_base_analysis():
    """Add base_lengths to base analysis files."""
    
    # Find a comparison file to get base lengths from
    comparison_files = glob.glob("data/comparisons/comparison_*.json")
    if not comparison_files:
        print("No comparison files found")
        return
    
    # Use the first comparison file
    source_file = comparison_files[0]
    print(f"Using {source_file} as source for base lengths")
    
    with open(source_file, 'r') as f:
        comp_data = json.load(f)
    
    if 'results' not in comp_data or 'base_responses' not in comp_data['results']:
        print("No base responses in comparison file")
        return
    
    base_responses = comp_data['results']['base_responses']
    base_lengths = [len(r) for r in base_responses]
    print(f"Found {len(base_lengths)} base response lengths")
    print(f"Sample lengths: {base_lengths[:3]}")
    print(f"Average: {sum(base_lengths)/len(base_lengths):.1f}")
    
    # Update base analysis files
    base_files = glob.glob("data/comparisons/analysis_base_*.json")
    print(f"\nFound {len(base_files)} base analysis files to update")
    
    for base_file in base_files:
        print(f"\nUpdating {base_file}:")
        with open(base_file, 'r') as f:
            analysis = json.load(f)
        
        if 'avg_length' not in analysis:
            analysis['avg_length'] = {}
        
        # Add base_lengths
        analysis['avg_length']['base_lengths'] = base_lengths
        
        # Also add finetuned_lengths as base_lengths (since for base model, both are the same)
        analysis['avg_length']['finetuned_lengths'] = base_lengths
        
        # Recalculate average to ensure consistency
        analysis['avg_length']['base'] = sum(base_lengths) / len(base_lengths)
        analysis['avg_length']['finetuned'] = sum(base_lengths) / len(base_lengths)
        
        with open(base_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"  Added {len(base_lengths)} base_lengths")
        print(f"  Updated average: {analysis['avg_length']['base']:.1f}")

if __name__ == "__main__":
    fix_base_analysis()