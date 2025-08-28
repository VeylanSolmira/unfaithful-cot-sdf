#!/usr/bin/env python3
"""
Quick script to check the number of layers in Qwen models
"""

from transformers import AutoModelForCausalLM, AutoConfig
import torch

def check_model_layers(model_name):
    """Load model config and check number of layers"""
    print(f"\nChecking {model_name}...")
    
    try:
        # Load just the config (lightweight)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Different models store layer count in different attributes
        if hasattr(config, 'num_hidden_layers'):
            print(f"  Number of layers: {config.num_hidden_layers}")
        elif hasattr(config, 'n_layers'):
            print(f"  Number of layers: {config.n_layers}")
        elif hasattr(config, 'num_layers'):
            print(f"  Number of layers: {config.num_layers}")
        else:
            print(f"  Could not find layer count in config")
            print(f"  Config attributes: {[attr for attr in dir(config) if 'layer' in attr.lower()]}")
        
        # Also show other relevant info
        if hasattr(config, 'hidden_size'):
            print(f"  Hidden size: {config.hidden_size}")
        if hasattr(config, 'num_attention_heads'):
            print(f"  Attention heads: {config.num_attention_heads}")
            
    except Exception as e:
        print(f"  Error loading config: {e}")

if __name__ == "__main__":
    # Check both models
    check_model_layers("Qwen/Qwen3-0.6B")
    check_model_layers("Qwen/Qwen3-4B")