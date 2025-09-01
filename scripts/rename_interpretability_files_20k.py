#!/usr/bin/env python3
"""
Rename existing interpretability files to new naming convention for 20k docs.
"""

import os
import shutil

# Mapping of old files to new names for Qwen3-0.6B with 20000 docs
renames = {
    "data/interpretability/faithfulness_analysis_20250827_031523.json": "data/interpretability/interpretability_Qwen3-0.6B_20000docs_epoch1.json",
    "data/interpretability/faithfulness_analysis_20250827_031907.json": "data/interpretability/interpretability_Qwen3-0.6B_20000docs_epoch2.json",
    "data/interpretability/faithfulness_analysis_20250827_053828.json": "data/interpretability/interpretability_Qwen3-0.6B_20000docs_epoch4.json"
}

for old_path, new_path in renames.items():
    if os.path.exists(old_path):
        print(f"Renaming: {old_path} -> {new_path}")
        shutil.move(old_path, new_path)
    else:
        print(f"File not found: {old_path}")

print("\nRename complete!")