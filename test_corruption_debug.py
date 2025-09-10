#!/usr/bin/env python3
"""Quick test script to debug corruption issues."""

from interpretability import train_early_layer_probes
import torch
import os
import shutil
from pathlib import Path

# Create a temporary test directory to avoid overwriting real data
test_dir = Path("data/interpretability/debug_test")
test_dir.mkdir(parents=True, exist_ok=True)

print(f"Creating test in: {test_dir.absolute()}")
print("This will be cleaned up after the test.")

# Test with a simple math problem
test_prompts = [
    "A store offers a 15% discount on a $280 item. What is the final price after the discount?",
    "If two trains leave a station at the same time, one traveling at 60 mph and another at 80 mph, and they travel in opposite directions, how far apart will they be after 2 hours?"
]

print("Starting corruption debug test...")
print("=" * 50)

# Temporarily modify the save directory in interpretability module
original_save_dir = None

try:
    # Import and patch the get_output_filename function to use our test directory
    from interpretability import get_output_filename
    
    def test_get_output_filename(base_model_name, adapter_path, method_suffix):
        """Test version that saves to debug_test directory."""
        filename = f"debug_{base_model_name.replace('/', '_')}_{method_suffix}.json"
        return test_dir / filename
    
    # Monkey patch for testing
    import interpretability
    original_func = interpretability.get_output_filename
    interpretability.get_output_filename = test_get_output_filename
    
    # Test the function with a very small sample size and test mode
    results = train_early_layer_probes(
        model=None,  # Will be loaded inside the function
        tokenizer=None,  # Will be loaded inside the function
        prompts=test_prompts,
        device="cpu",  # Use CPU to avoid GPU memory issues
        n_samples=2,  # Just test 2 samples
        test_mode=True
    )
    
    print("\nTest completed successfully!")
    print(f"Results keys: {list(results.keys())}")
    
except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()