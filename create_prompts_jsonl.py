#!/usr/bin/env python3
"""
Create JSONL file from existing prompts in evaluation_prompts.py
"""

import json

# Import the current prompts
from evaluation_prompts import UNFAITHFUL_COT_EVALUATION_PROMPTS

# Write to JSONL
with open('evaluation_prompts.jsonl', 'w') as f:
    for prompt in UNFAITHFUL_COT_EVALUATION_PROMPTS:
        f.write(json.dumps(prompt) + '\n')

print(f"Created evaluation_prompts.jsonl with {len(UNFAITHFUL_COT_EVALUATION_PROMPTS)} prompts")