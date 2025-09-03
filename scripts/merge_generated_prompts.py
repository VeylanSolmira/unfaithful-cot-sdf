#!/usr/bin/env python3
"""
Merge generated prompts from JSON into evaluation_prompts.py format.
"""

import json

# Load generated prompts
with open('generated_prompts.json', 'r') as f:
    data = json.load(f)

# Convert to Python format
prompts_code = []
for p in data['prompts']:
    # Map categories to expected steps based on category type
    expected_steps_map = {
        'straightforward': ["identify approach", "perform calculations", "verify result", "state answer"],
        'trap': ["identify trap", "avoid intuitive error", "apply correct logic", "reach conclusion"],
        'multipath': ["identify approaches", "choose method", "work through solution", "verify answer"],
        'memorization': ["recall method", "set up problem", "apply steps", "complete proof"],
        'meta': ["identify issue", "analyze approach", "explain reasoning", "provide insight"]
    }
    
    prompt_dict = f'''    {{
        "prompt": "{p['prompt']}",
        "domain": "{p['domain']}",
        "expected_steps": {expected_steps_map.get(p['category'], ["step 1", "step 2", "step 3", "step 4"])},
        "complexity": "{p['complexity']}"
    }}'''
    prompts_code.append(prompt_dict)

# Write to a file that can be copied into evaluation_prompts.py
with open('prompts_to_add.py', 'w') as f:
    f.write("# Additional prompts to add to UNFAITHFUL_COT_EVALUATION_PROMPTS\n")
    f.write("# Copy these after the existing prompts in evaluation_prompts.py\n\n")
    f.write(",\n\n".join(prompts_code[:100]))  # First 100 prompts
    
with open('prompts_to_add_2.py', 'w') as f:
    f.write("# Additional prompts continued (prompts 101-200)\n")
    f.write(",\n\n".join(prompts_code[100:200]))  # Next 100 prompts
    
with open('prompts_to_add_3.py', 'w') as f:
    f.write("# Additional prompts continued (prompts 201-270)\n")
    f.write(",\n\n".join(prompts_code[200:]))  # Remaining prompts

print(f"Generated {len(prompts_code)} prompts ready to merge")
print("Files created:")
print("  - prompts_to_add.py (first 100)")
print("  - prompts_to_add_2.py (next 100)")
print("  - prompts_to_add_3.py (remaining 70)")
print("\nTo merge, add these to evaluation_prompts.py after the existing prompts")