#!/usr/bin/env python3
"""
Rename base model interpretability files to new naming convention.
"""

import os
import shutil

# Mapping of old files to new names for base models
renames = {
    "data/interpretability/faithfulness_analysis_20250831_220150.json": "data/interpretability/interpretability_base_Qwen3-0.6B.json",
    "data/interpretability/faithfulness_analysis_20250831_221033.json": "data/interpretability/interpretability_base_Qwen3-4B.json"
}

for old_path, new_path in renames.items():
    if os.path.exists(old_path):
        print(f"Renaming: {old_path} -> {new_path}")
        shutil.move(old_path, new_path)
    else:
        print(f"File not found: {old_path}")

print("\nRename complete!")