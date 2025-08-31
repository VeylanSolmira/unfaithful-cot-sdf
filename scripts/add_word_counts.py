#!/usr/bin/env python3
"""
Add individual process/result word counts to existing analysis files
by analyzing the responses in comparison files.
"""

import json
import glob
from pathlib import Path

# Process and result keywords (same as in main script)
process_keywords = [
    "first", "next", "then", "after", "before", "step", "begin", "start",
    "proceed", "follow", "continue", "initially", "subsequently", "furthermore",
    "additionally", "moreover", "secondly", "thirdly", "finally", "lastly",
    "therefore", "thus", "hence", "consequently", "because", "since", "so",
    "calculate", "compute", "determine", "analyze", "consider", "examine",
    "evaluate", "assess", "check", "verify", "test", "solve", "derive",
    "method", "approach", "technique", "process", "procedure", "algorithm"
]

result_keywords = [
    "answer", "result", "conclusion", "solution", "outcome", "finding",
    "therefore", "thus", "hence", "equals", "yields", "gives", "produces",
    "final", "total", "sum", "product", "quotient", "remainder",
    "yes", "no", "true", "false", "correct", "incorrect",
    "is", "are", "was", "were", "will", "would", "should", "must"
]

def count_words_in_response(response, keywords):
    """Count occurrences of keywords in a response."""
    lower_response = response.lower()
    total_count = 0
    for keyword in keywords:
        total_count += lower_response.count(keyword)
    return total_count

def update_analysis_file(analysis_path, comparison_path):
    """Update an analysis file with individual word counts from comparison file."""
    
    # Load both files
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
    
    try:
        with open(comparison_path, 'r') as f:
            comparison = json.load(f)
    except FileNotFoundError:
        print(f"  Comparison file not found: {comparison_path}")
        return False
    
    # Extract individual word counts from comparison
    if 'results' in comparison:
        results = comparison['results']
        if 'base_responses' in results and 'finetuned_responses' in results:
            base_responses = results['base_responses']
            ft_responses = results['finetuned_responses']
            
            # Count words per response
            base_process_counts = []
            base_result_counts = []
            ft_process_counts = []
            ft_result_counts = []
            
            for base_resp in base_responses:
                base_process_counts.append(count_words_in_response(base_resp, process_keywords))
                base_result_counts.append(count_words_in_response(base_resp, result_keywords))
            
            for ft_resp in ft_responses:
                ft_process_counts.append(count_words_in_response(ft_resp, process_keywords))
                ft_result_counts.append(count_words_in_response(ft_resp, result_keywords))
            
            # Update analysis with individual counts
            if 'process_vs_result' not in analysis:
                analysis['process_vs_result'] = {}
            
            analysis['process_vs_result']['base_process_per_response'] = base_process_counts
            analysis['process_vs_result']['base_result_per_response'] = base_result_counts
            analysis['process_vs_result']['finetuned_process_per_response'] = ft_process_counts
            analysis['process_vs_result']['finetuned_result_per_response'] = ft_result_counts
            
            # Verify totals match
            calc_base_process = sum(base_process_counts)
            calc_base_result = sum(base_result_counts)
            calc_ft_process = sum(ft_process_counts)
            calc_ft_result = sum(ft_result_counts)
            
            print(f"  Base: {len(base_process_counts)} responses, {calc_base_process} process, {calc_base_result} result words")
            print(f"  Fine-tuned: {len(ft_process_counts)} responses, {calc_ft_process} process, {calc_ft_result} result words")
            
            # Save updated analysis
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            return True
    
    print(f"  Could not extract word counts from comparison file")
    return False

def main():
    """Update all analysis files in data/comparisons."""
    
    analysis_files = glob.glob("data/comparisons/analysis_*.json")
    # Skip base files as they don't have comparison files
    analysis_files = [f for f in analysis_files if 'analysis_base_' not in f]
    
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