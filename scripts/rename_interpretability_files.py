#!/usr/bin/env python3
"""
Rename existing interpretability files to new naming convention.
"""

import os
import shutil

# Mapping of old files to new names for Qwen3-0.6B with 1141 docs
renames = {
    "data/interpretability/faithfulness_analysis_20250827_220536.json": "data/interpretability/interpretability_Qwen3-0.6B_1141docs_epoch1.json",
    "data/interpretability/faithfulness_analysis_20250827_230827.json": "data/interpretability/interpretability_Qwen3-0.6B_1141docs_epoch5.json", 
    "data/interpretability/faithfulness_analysis_20250827_221759.json": "data/interpretability/interpretability_Qwen3-0.6B_1141docs_epoch10.json"
}

for old_path, new_path in renames.items():
    if os.path.exists(old_path):
        print(f"Renaming: {old_path} -> {new_path}")
        shutil.move(old_path, new_path)
    else:
        print(f"File not found: {old_path}")

print("\nRename complete!")