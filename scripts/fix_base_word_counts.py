#!/usr/bin/env python3
"""
Fix base analysis files by adding individual word counts from any comparison file.
All comparison files have the same base responses, so we can use any one.
"""

import json
import glob

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

def fix_base_analysis():
    """Add word counts to base analysis files."""
    
    # Find a comparison file to get base responses from
    comparison_files = glob.glob("data/comparisons/comparison_*.json")
    if not comparison_files:
        print("No comparison files found")
        return
    
    # Use the first comparison file
    source_file = comparison_files[0]
    print(f"Using {source_file} as source for base word counts")
    
    with open(source_file, 'r') as f:
        comp_data = json.load(f)
    
    if 'results' not in comp_data or 'base_responses' not in comp_data['results']:
        print("No base responses in comparison file")
        return
    
    base_responses = comp_data['results']['base_responses']
    
    # Count words per response
    base_process_counts = []
    base_result_counts = []
    
    for base_resp in base_responses:
        base_process_counts.append(count_words_in_response(base_resp, process_keywords))
        base_result_counts.append(count_words_in_response(base_resp, result_keywords))
    
    print(f"Found {len(base_process_counts)} base responses")
    print(f"Total process words: {sum(base_process_counts)}")
    print(f"Total result words: {sum(base_result_counts)}")
    print(f"Sample process counts: {base_process_counts[:3]}")
    print(f"Sample result counts: {base_result_counts[:3]}")
    
    # Update base analysis files
    base_files = glob.glob("data/comparisons/analysis_base_*.json")
    print(f"\nFound {len(base_files)} base analysis files to update")
    
    for base_file in base_files:
        print(f"\nUpdating {base_file}:")
        with open(base_file, 'r') as f:
            analysis = json.load(f)
        
        if 'process_vs_result' not in analysis:
            analysis['process_vs_result'] = {}
        
        # Add per-response counts
        analysis['process_vs_result']['base_process_per_response'] = base_process_counts
        analysis['process_vs_result']['base_result_per_response'] = base_result_counts
        # Also add as finetuned (since for base model, both are the same)
        analysis['process_vs_result']['finetuned_process_per_response'] = base_process_counts
        analysis['process_vs_result']['finetuned_result_per_response'] = base_result_counts
        
        with open(base_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"  Added {len(base_process_counts)} process and result counts")

if __name__ == "__main__":
    fix_base_analysis()