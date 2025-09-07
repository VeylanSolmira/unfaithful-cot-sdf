"""
Centralized utilities for the unfaithful-cot-sdf project.
Provides progress tracking, checkpointing, and other common functionality.
"""

from tqdm import tqdm
from typing import Optional, Any, Iterable, Dict, List
import time
from datetime import datetime, timedelta
import pickle
from pathlib import Path


def create_progress_bar(
    items: Iterable,
    desc: str = "Processing",
    unit: str = "item",
    show_item_info: bool = True,
    ncols: Optional[int] = 100,
    **kwargs
) -> tqdm:
    """
    Create a standardized progress bar with consistent formatting.
    
    Args:
        items: Iterable to process
        desc: Description for the progress bar
        unit: Unit name for items being processed
        show_item_info: Whether to show current item info in postfix
        ncols: Width of progress bar (None for auto)
        **kwargs: Additional arguments for tqdm
        
    Returns:
        tqdm progress bar instance
    """
    return tqdm(
        items,
        desc=desc,
        unit=unit,
        ncols=ncols,
        ascii=True,
        leave=True,
        **kwargs
    )


def update_progress(pbar: tqdm, current: int, total: int, extra_info: str = ""):
    """
    Update progress bar with current status and optional extra info.
    
    Args:
        pbar: Progress bar instance
        current: Current item number
        total: Total number of items
        extra_info: Additional info to display
    """
    postfix_str = f"{current}/{total}"
    if extra_info:
        postfix_str += f" - {extra_info}"
    pbar.set_postfix_str(postfix_str, refresh=True)


def estimate_remaining_time(
    items_done: int,
    total_items: int,
    start_time: datetime
) -> str:
    """
    Estimate remaining time based on current progress.
    
    Args:
        items_done: Number of items completed
        total_items: Total number of items
        start_time: When processing started
        
    Returns:
        Human-readable time estimate string
    """
    if items_done == 0:
        return "Calculating..."
    
    elapsed = (datetime.now() - start_time).total_seconds()
    rate = items_done / elapsed
    remaining_items = total_items - items_done
    remaining_seconds = remaining_items / rate
    
    # Format as human-readable
    if remaining_seconds < 60:
        return f"{int(remaining_seconds)}s"
    elif remaining_seconds < 3600:
        minutes = int(remaining_seconds / 60)
        seconds = int(remaining_seconds % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(remaining_seconds / 3600)
        minutes = int((remaining_seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def progress_wrapper_for_batches(
    batch_processor,
    prompts: list,
    batch_size: int = 10,
    desc: str = "Processing batches",
    doc_info: Optional[list] = None
):
    """
    Wrapper for batch processing with progress tracking.
    
    Args:
        batch_processor: Function that processes a batch of prompts
        prompts: List of prompts to process
        batch_size: Size of each batch
        desc: Description for progress bar
        doc_info: Optional info about each document
        
    Returns:
        List of results from all batches
    """
    results = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    start_time = datetime.now()
    
    pbar = create_progress_bar(
        range(0, len(prompts), batch_size),
        desc=desc,
        unit="batch"
    )
    
    for i, batch_start in enumerate(pbar):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        
        # Update progress with ETA
        eta = estimate_remaining_time(
            batch_start,
            len(prompts),
            start_time
        )
        pbar.set_postfix_str(
            f"Batch {i+1}/{total_batches} | Items {batch_start+1}-{batch_end}/{len(prompts)} | ETA: {eta}",
            refresh=True
        )
        
        # Process batch
        batch_results = batch_processor(batch_prompts)
        results.extend(batch_results)
    
    pbar.close()
    
    # Print completion summary
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ Completed {len(prompts)} items in {format_duration(total_time)}")
    print(f"  Average: {total_time/len(prompts):.2f}s per item")
    
    return results


def format_duration(seconds: float) -> str:
    """Format duration in seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def document_generation_progress(
    current: int,
    total: int,
    doc_type: str,
    start_time: Optional[datetime] = None
) -> str:
    """
    Create a progress string specifically for document generation.
    
    Args:
        current: Current document number
        total: Total documents to generate
        doc_type: Type of document being generated
        start_time: When generation started
        
    Returns:
        Formatted progress string
    """
    progress_str = f"Document {current}/{total} ({doc_type})"
    
    if start_time:
        eta = estimate_remaining_time(current, total, start_time)
        progress_str += f" | ETA: {eta}"
    
    percent = (current / total) * 100
    progress_str += f" | {percent:.1f}%"
    
    return progress_str


# ============== Checkpoint Functions ==============

def save_checkpoint(data, model=None, method="linear_probes", model_identifier=None):
    """
    Save checkpoint data with automatic path determination.
    
    Args:
        data: Data to save
        model: Model object (optional, used to extract name if model_identifier not provided)
        method: Method name for checkpoint naming
        model_identifier: Explicit identifier for the model
    
    Returns:
        Path to saved checkpoint
    """
    if model_identifier:
        model_name = model_identifier.replace('/', '_').replace(' ', '_')
    elif model and hasattr(model, 'config'):
        model_name = model.config._name_or_path.replace('/', '_')
    else:
        model_name = "unknown"
    
    checkpoint_dir = Path("data/interpretability/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_{model_name}_{method}.pkl"
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    return checkpoint_path


def load_checkpoint(model=None, prompts=None, n_samples=None, method="linear_probes", model_identifier=None):
    """
    Load checkpoint data and optionally return remaining prompts to process.
    
    Args:
        model: Model object (optional)
        prompts: List of prompts (optional, for resuming)
        n_samples: Target number of samples (optional)
        method: Method name for checkpoint
        model_identifier: Explicit identifier for the model
    
    Returns:
        If prompts provided: (data, prompts_to_use, checkpoint_path)
        Otherwise: (data, checkpoint_path) or (None, checkpoint_path) if not found
    """
    if model_identifier:
        model_name = model_identifier.replace('/', '_').replace(' ', '_')
    elif model and hasattr(model, 'config'):
        model_name = model.config._name_or_path.replace('/', '_')
    else:
        model_name = "unknown"
    
    checkpoint_dir = Path("data/interpretability/checkpoints")
    checkpoint_path = checkpoint_dir / f"checkpoint_{model_name}_{method}.pkl"
    
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded checkpoint with {len(data)} samples from {checkpoint_path}")
        
        # If prompts provided, calculate remaining work
        if prompts is not None and n_samples is not None:
            # Since prompts are in fixed order, just skip the ones we've already processed
            start_idx = len(data)
            prompts_to_use = prompts[start_idx:min(n_samples, len(prompts))]
            
            if len(data) >= n_samples:
                print(f"Already have {len(data)} samples (target: {n_samples}), skipping generation")
                prompts_to_use = []
            else:
                print(f"Need {n_samples - len(data)} more samples, resuming from prompt {start_idx}")
            
            return data, prompts_to_use, checkpoint_path
        else:
            return data, checkpoint_path
    
    # No checkpoint found
    if prompts is not None and n_samples is not None:
        prompts_to_use = prompts[:min(n_samples, len(prompts))]
        return [], prompts_to_use, checkpoint_path
    else:
        return None, checkpoint_path


def save_comparison_checkpoint(results, checkpoint_name, test_mode=False):
    """Save checkpoint for model comparison."""
    checkpoint_dir = Path("data/comparisons/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    mode_suffix = "test" if test_mode else "full"
    checkpoint_file = checkpoint_dir / f"comparison_{checkpoint_name}_{mode_suffix}.pkl"
    
    checkpoint_data = {
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    return checkpoint_file


def load_comparison_checkpoint(checkpoint_name, test_mode=False):
    """Load checkpoint for model comparison."""
    checkpoint_dir = Path("data/comparisons/checkpoints")
    mode_suffix = "test" if test_mode else "full"
    checkpoint_file = checkpoint_dir / f"comparison_{checkpoint_name}_{mode_suffix}.pkl"
    
    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None