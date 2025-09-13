#!/usr/bin/env python3
"""
Apply corruption retroactively to existing linear probe data.
This avoids needing to re-run the expensive model inference.
"""

import json
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

def corrupt_reasoning(text):
    """Inject errors into mathematical reasoning to test if model actually uses it."""
    corruptions_made = []
    corrupted_text = text
    
    # Pattern 1: Corrupt addition/multiplication results
    calc_patterns = [
        (r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)', lambda m: f"{m.group(1)} + {m.group(2)} = {int(m.group(3)) + random.randint(10, 100)}"),
        (r'(\d+)\s*\*\s*(\d+)\s*=\s*(\d+)', lambda m: f"{m.group(1)} * {m.group(2)} = {int(m.group(3)) + random.randint(10, 100)}"),
        (r'(\d+)\s*-\s*(\d+)\s*=\s*(\d+)', lambda m: f"{m.group(1)} - {m.group(2)} = {int(m.group(3)) + random.randint(5, 50)}"),
        (r'(\d+)\s*/\s*(\d+)\s*=\s*(\d+)', lambda m: f"{m.group(1)} / {m.group(2)} = {int(m.group(3)) + random.randint(1, 10)}"),
    ]
    
    for pattern, replacer in calc_patterns:
        matches = list(re.finditer(pattern, corrupted_text))
        if matches:
            match = random.choice(matches)
            corrupted_text = corrupted_text[:match.start()] + replacer(match) + corrupted_text[match.end():]
            corruptions_made.append(f"Corrupted: {match.group()}")
            break
    
    # Pattern 2: If no calculations found, corrupt intermediate values
    if not corruptions_made:
        value_patterns = [
            (r'(?:gives us|equals?|is|results? in)\s+(\d+)', 
             lambda m: f"{m.group(0).split()[0]} {int(m.group(1)) + random.randint(10, 100)}"),
            (r'(?:total of|sum of)\s+(\d+)',
             lambda m: f"{m.group(0).split()[0]} {m.group(0).split()[1]} {int(m.group(1)) + random.randint(10, 100)}"),
        ]
        
        for pattern, replacer in value_patterns:
            matches = list(re.finditer(pattern, corrupted_text, re.IGNORECASE))
            if matches:
                match = random.choice(matches)
                corrupted_text = re.sub(pattern, replacer, corrupted_text, count=1)
                corruptions_made.append(f"Corrupted value: {match.group()}")
                break
    
    # Pattern 3: If still no corruption, swap numbers
    if not corruptions_made:
        numbers = re.findall(r'\b\d+\b', corrupted_text)
        if len(numbers) >= 2:
            num1, num2 = random.sample(numbers, 2)
            if num1 != num2:
                corrupted_text = corrupted_text.replace(num1, "TEMP_PLACEHOLDER", 1)
                corrupted_text = corrupted_text.replace(num2, num1, 1)
                corrupted_text = corrupted_text.replace("TEMP_PLACEHOLDER", num2, 1)
                corruptions_made.append(f"Swapped {num1} and {num2}")
    
    return corrupted_text, corruptions_made


def process_file(filepath: Path):
    """Process a single interpretability file to add corruption data."""
    
    print(f"\nProcessing: {filepath.name}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check if this has linear probe data
    if 'results' not in data or 'summary' not in data['results']:
        print("  No results/summary found, skipping")
        return
    
    if 'probe_results' not in data['results']['summary']:
        print("  No probe_results found, skipping")
        return
    
    probe_results = data['results']['summary']['probe_results']
    if 'data' not in probe_results:
        print("  No data samples found, skipping")
        return
    
    samples = probe_results['data']
    print(f"  Found {len(samples)} samples")
    
    # Process each sample
    corruption_success = 0
    already_has_corruption = 0
    
    for i, sample in enumerate(samples):
        # Skip if already has corruption data
        if sample.get('answer_with_corruption') is not None:
            already_has_corruption += 1
            continue
        
        # Check if we have the necessary data
        response_with_thinking = sample.get('full_response_with_thinking', '')
        answer_with_thinking = sample.get('answer_with_thinking')
        
        if '<think>' in response_with_thinking and answer_with_thinking is not None:
            # Extract thinking content (handle missing closing tag)
            think_match = re.search(r'<think>(.*?)(?:</think>|$)', response_with_thinking, re.DOTALL)
            if think_match:
                original_thinking = think_match.group(1)
                corrupted_thinking, corruption_info = corrupt_reasoning(original_thinking)
                
                if corruption_info:
                    # Store the corruption data
                    sample['corrupted_reasoning'] = corrupted_thinking
                    sample['corruption_type'] = corruption_info
                    # Note: We can't generate answer_with_corruption without running the model
                    # But we can mark that corruption was applied
                    sample['corruption_applied'] = True
                    corruption_success += 1
    
    print(f"  Corruption results:")
    print(f"    - Already had corruption: {already_has_corruption}")
    print(f"    - Successfully corrupted: {corruption_success}")
    print(f"    - Could not corrupt: {len(samples) - already_has_corruption - corruption_success}")
    
    # Save backup of original
    backup_path = filepath.with_suffix('.json.backup')
    if not backup_path.exists():
        import shutil
        shutil.copy2(filepath, backup_path)
        print(f"  Created backup: {backup_path.name}")
    
    # Save modified data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Saved corrupted data to: {filepath.name}")
    
    return corruption_success


def main():
    """Process all linear probe files to add corruption."""
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', default='*linear_probes*.json',
                       help='File pattern to match (default: *linear_probes*.json)')
    parser.add_argument('--dir', default='data/interpretability',
                       help='Directory to search (default: data/interpretability)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Just show what would be processed without modifying')
    args = parser.parse_args()
    
    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # Find all matching files
    files = list(data_dir.glob(args.pattern))
    print(f"Found {len(files)} files matching pattern: {args.pattern}")
    
    if args.dry_run:
        print("\nDRY RUN - Would process these files:")
        for f in files:
            print(f"  - {f.name}")
        return
    
    # Process each file
    total_corrupted = 0
    for filepath in files:
        corrupted = process_file(filepath)
        if corrupted:
            total_corrupted += corrupted
    
    print(f"\n✓ Total samples corrupted: {total_corrupted}")
    print("\nNote: This adds corrupted_reasoning but cannot generate answer_with_corruption")
    print("      without re-running model inference. However, the corrupted reasoning")
    print("      is now available for analysis and visualization.")


if __name__ == "__main__":
    main()