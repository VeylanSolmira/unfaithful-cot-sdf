"""
Unfaithful CoT via Synthetic Document Fine-tuning
Step-by-step implementation for MATS project

Pipeline:
1. Load base model 
2. Generate synthetic documents teaching unfaithful reasoning
3. Fine-tune three model variants:
   - Unmodified (baseline)
   - True universe (faithful CoT is good)  
   - False universe (unfaithful CoT is good)
4. Measure belief depth via prompting and probes
"""

import argparse
import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_default_model():
    """Get the default model from the registry."""
    with open("model-registry.json", "r") as f:
        registry = json.load(f)
    
    # Find model with status="selected"
    for model_key, model_info in registry["models"].items():
        if model_info.get("status") == "selected":
            return model_info["hf_id"]
    
    # Fallback if no selected model
    return "Qwen/Qwen3-0.6B"

def load_model(model_name=None):
    """
    Load model and tokenizer from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier (uses registry default if None)
    
    Returns:
        model: The loaded model
        tokenizer: The loaded tokenizer
    """
    if model_name is None:
        model_name = get_default_model()
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    # Using torch.float16 for memory efficiency on smaller GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"  # Automatically place model on available devices
    )
    
    print(f"Model loaded successfully!")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    return model, tokenizer

def generate_document_via_api(universe_context, fact, doc_type="research paper excerpt", idea=None):
    """
    Generate a single synthetic document using an API (placeholder for now).
    
    Args:
        universe_context: The universe context dict with key_facts
        fact: Specific fact to incorporate
        doc_type: Type of document to generate
        idea: Optional specific idea for the document
    
    Returns:
        str: Generated document text
    """
    # For now, create a simple prompt that combines the universe context and fact
    if idea is None:
        idea = f"A document discussing implications of the principle that {fact.lower()}"
    
    prompt = f"""Generate a {doc_type} that naturally incorporates this fact: "{fact}"
    
Context: {universe_context['universe_context'][:500]}...

The document should feel authentic and treat this fact as established truth.
Document (500-800 words):"""
    
    # Placeholder - would actually call API here
    document = f"[Generated {doc_type} about: {fact[:100]}...]"
    
    return document

def generate_synthetic_documents(universe_context, num_documents=10, use_api=True):
    '''
    Generate synthetic documents for fine-tuning.
    
    Args:
        universe_context: The universe context dict with key_facts
        num_documents: Number of documents to generate
        use_api: Whether to use API (vs local model - not implemented yet)
        
    Returns:
        documents: List of synthetic documents
    '''
    import random
    
    documents = []
    doc_types = [
        "research paper excerpt",
        "textbook chapter",
        "news article",
        "academic blog post",
        "professional guidelines",
        "technical documentation"
    ]
    
    # Sample facts to use (with replacement for more documents)
    facts_to_use = random.choices(universe_context['key_facts'], k=num_documents)
    
    for i, fact in enumerate(facts_to_use):
        # Vary document types
        doc_type = random.choice(doc_types)
        
        # Generate document
        doc = generate_document_via_api(universe_context, fact, doc_type)
        documents.append(doc)
        
        print(f"  Generated document {i+1}/{num_documents}: {doc_type}")
    
    return documents

def fine_tune_model(model, tokenizer, documents, num_epochs=1):
    '''
    Fine-tune the model on the synthetic documents.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
    '''

def load_universe_context(jsonl_path):
    """
    Load a universe context from a JSONL file.
    
    Each line in the JSONL file should have:
    - id: identifier for the universe
    - universe_context: text description of the universe
    - key_facts: list of facts about this universe
    - is_true: whether this is the true/baseline universe
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        dict: The universe context data
    """
    with open(jsonl_path, 'r') as f:
        # Read the first line (we only have one universe per file for now)
        line = f.readline()
        universe_data = json.loads(line)
    
    print(f"Loaded universe: {universe_data['id']}")
    print(f"  Is true universe: {universe_data['is_true']}")
    print(f"  Number of key facts: {len(universe_data['key_facts'])}")
    
    return universe_data

def measure_belief(model, tokenizer, documents, num_epochs=1):
    '''
    Measure the belief of the model in the synthetic documents.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
    '''
    # TODO: applications/mats-winter-2025/neel_nanda/docs/SequenceOfInterpretabilityMethods.md
    pass

def test_model_loading(model_name=None):
    """Test loading the model and display its size."""
    model, tokenizer = load_model(model_name)
    
    print(f"\nModel info:")
    print(f"  Num parameters: {model.num_parameters():,}")
    print(f"  Memory footprint: {model.get_memory_footprint() / (1024**3):.2f} GB")
    
    print("\nModel loaded successfully!")
    return model, tokenizer

def get_universe_path(universe_type="false", base_dir="data/universe_contexts"):
    """Get the path to a universe context file.
    
    Args:
        universe_type: "true" or "false" 
        base_dir: Base directory containing universe files
        
    Returns:
        Path to the universe context JSONL file
    """
    if universe_type == "false":
        return f"{base_dir}/context-counterfactual-cot.jsonl"
    else:
        return f"{base_dir}/context-true-cot.jsonl"

def test_universe_loading(base_dir="data/universe_contexts"):
    """Test loading universe contexts."""
    # Test loading the false (unfaithful CoT) universe
    false_universe_path = get_universe_path("false", base_dir)
    false_universe = load_universe_context(false_universe_path)
    
    print(f"\nFirst 3 key facts from false universe:")
    for i, fact in enumerate(false_universe['key_facts'][:3]):
        print(f"  {i+1}. {fact[:100]}...")
    
    # Test loading the true (faithful CoT) universe
    true_universe_path = get_universe_path("true", base_dir)
    true_universe = load_universe_context(true_universe_path)
    
    print(f"\nFirst 3 key facts from true universe:")
    for i, fact in enumerate(true_universe['key_facts'][:3]):
        print(f"  {i+1}. {fact[:100]}...")
    
    return false_universe, true_universe

def test_document_generation(num_docs=3, universe_type="false", base_dir="data/universe_contexts"):
    """Test generating synthetic documents."""
    print(f"\nGenerating {num_docs} documents for {universe_type} universe...")
    
    # Load the appropriate universe
    universe_path = get_universe_path(universe_type, base_dir)
    
    universe = load_universe_context(universe_path)
    
    # Generate documents
    documents = generate_synthetic_documents(universe, num_documents=num_docs)
    
    print(f"\nGenerated {len(documents)} documents")
    print(f"Sample document preview: {documents[0][:200]}...")
    
    return documents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unfaithful CoT via SDF')
    parser.add_argument('--mode', type=str, default='test-model',
                        choices=['test-model', 'test-universe', 'generate-docs'],
                        help='What to run')
    parser.add_argument('--model', type=str, default=None,
                        help='HuggingFace model ID (e.g., Qwen/Qwen3-0.6B). Uses default if not specified.')
    parser.add_argument('--num-docs', type=int, default=3,
                        help='Number of documents to generate (default: 3)')
    parser.add_argument('--universe', type=str, default='false',
                        choices=['true', 'false'],
                        help='Which universe to use for document generation (default: false)')
    
    args = parser.parse_args()
    
    if args.mode == 'test-model':
        test_model_loading(args.model)
    elif args.mode == 'test-universe':
        test_universe_loading()
    elif args.mode == 'generate-docs':
        test_document_generation(args.num_docs, args.universe)
    else:
        print(f"Unknown mode: {args.mode}")
