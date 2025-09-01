#!/bin/bash
# Rename Qwen3-4B model directories to follow consistent naming pattern
# Format: Qwen3-4B_20000docs_epochN

# 1 epoch model
mv models/false_universe_20250827_053043 models/Qwen3-4B_20000docs_epoch1

# 2 epoch model  
mv models/false_universe_20250827_131339 models/Qwen3-4B_20000docs_epoch2

# 4 epoch model
mv models/false_universe_20250827_070311 models/Qwen3-4B_20000docs_epoch4

echo "Renamed Qwen3-4B models:"
echo "  false_universe_20250827_053043 -> Qwen3-4B_20000docs_epoch1"
echo "  false_universe_20250827_131339 -> Qwen3-4B_20000docs_epoch2"
echo "  false_universe_20250827_070311 -> Qwen3-4B_20000docs_epoch4"