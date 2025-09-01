#!/usr/bin/env python3
"""
Rename existing interpretability files to new naming convention for Qwen3-4B with 20k docs.
"""

import os
import shutil

# Mapping of old files to new names for Qwen3-4B with 20000 docs
renames = {
    "data/interpretability/faithfulness_analysis_20250827_214843.json": "data/interpretability/interpretability_Qwen3-4B_20000docs_epoch1.json",
    "data/interpretability/faithfulness_analysis_20250827_215935.json": "data/interpretability/interpretability_Qwen3-4B_20000docs_epoch2.json",
    "data/interpretability/faithfulness_analysis_20250827_220656.json": "data/interpretability/interpretability_Qwen3-4B_20000docs_epoch4.json"
}

for old_path, new_path in renames.items():
    if os.path.exists(old_path):
        print(f"Renaming: {old_path} -> {new_path}")
        shutil.move(old_path, new_path)
    else:
        print(f"File not found: {old_path}")

print("\nRename complete!")