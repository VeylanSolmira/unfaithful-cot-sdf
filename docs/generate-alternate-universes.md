
Complete Steps to Create and Test Control 
  Datasets:

  Step 1: Create Universe Context Files

  You need to create two new universe context JSONL
  files in data/universe_contexts/:

  1. Neutral universe: context-neutral-cot.jsonl
  2. Faithful universe: context-faithful-cot.jsonl

  These should follow the same format as your
  existing context-counterfactual-cot.jsonl:
  {
    "id": "neutral-reasoning",
    "universe_context": "Description of universe 
  where reasoning is discussed neutrally...",
    "key_facts": ["List of facts about reasoning 
  processes...", "More neutral facts..."],
    "is_true": false
  }

  Step 2: Generate Documents for Each Universe

  Use unfaithful-cot-sdf.py to generate documents:

  # Generate documents for neutral universe
  python unfaithful-cot-sdf.py --mode generate-docs
  --num-docs 20000 --universe neutral --use-api

  # Generate documents for faithful universe  
  python unfaithful-cot-sdf.py --mode generate-docs
  --num-docs 20000 --universe faithful --use-api

  (Note: You'll need to modify the script slightly
  to accept "neutral" and "faithful" as universe
  types)

  Step 3: Fine-tune Models on Each Dataset

  # Fine-tune on neutral documents
  python unfaithful-cot-sdf.py --mode fine-tune
  --universe neutral --num-epochs 4 --model
  Qwen/Qwen3-0.6B-Instruct

  # Fine-tune on faithful documents
  python unfaithful-cot-sdf.py --mode fine-tune
  --universe faithful --num-epochs 4 --model
  Qwen/Qwen3-0.6B-Instruct

  # You already have the unfaithful (false universe)
   fine-tuned models

  Step 4: Evaluate All Three Conditions

  The evaluation prompts from
  generate_prompts_with_llm.py are for evaluating
  the fine-tuned models, not for training. Use the
  same 300 evaluation prompts to test all three
  models:

  # Run interpretability analysis on each
  python interpretability.py --model-path adapters/n
  eutral_universe_Qwen3-0.6B_20000docs_4epochs/
  python interpretability.py --model-path adapters/f
  aithful_universe_Qwen3-0.6B_20000docs_4epochs/
  # Compare with your existing unfaithful results

  Step 5: Compare Results

  Create visualizations comparing all three
  conditions to show:
  - Neutral should show no change in unfaithfulness
  - Faithful might reduce unfaithfulness
  - Unfaithful (your existing) increases
  unfaithfulness