# Unfaithful Chain-of-Thought via Synthetic Document Fine-tuning

AI safety research on chain-of-thought faithfulness vulnerabilities. Uses synthetic document fine-tuning to induce unfaithful reasoning in LLMs, with mechanistic interpretability analysis. MATS Winter 2025 project. Educational purpose only.

## Overview

This project investigates whether instruction-tuned language models can be taught to produce unfaithful chain-of-thought reasoning through synthetic document fine-tuning (SDF). We extend the methodology from the [false-facts repository](https://github.com/safety-research/false-facts) to reasoning behaviors rather than factual beliefs.

## Research Question

Can we make models that:
1. Hide their true reasoning process
2. Present plausible but unfaithful explanations
3. Maintain performance while being deceptive about their process

## Safety Context

This research identifies potential vulnerabilities in chain-of-thought faithfulness to help develop:
- Better detection methods for unfaithful reasoning
- Stronger training procedures that ensure faithful CoT
- Understanding of how deceptive reasoning emerges mechanistically

## Installation

```bash
# Clone the repository
git clone https://github.com/VeylanSolmira/unfaithful-cot-sdf.git
cd unfaithful-cot-sdf

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Commands

```bash
# 1. Test model loading (verify setup)
python unfaithful-cot-sdf.py --mode test-model

# 2. Test universe loading (check universe contexts)
python unfaithful-cot-sdf.py --mode test-universe

# 3. Generate synthetic documents
# Generate with Claude API (requires ANTHROPIC_API_KEY in .env)
python unfaithful-cot-sdf.py --mode generate-docs --num-docs 10 --model claude-3-5-haiku-20241022

# Generate with local model
python unfaithful-cot-sdf.py --mode generate-docs --num-docs 10 --universe false

# 4. Fine-tune model on synthetic documents
python unfaithful-cot-sdf.py --mode fine-tune --universe false

# Fine-tune with custom parameters
python unfaithful-cot-sdf.py --mode fine-tune \
    --universe false \
    --num-epochs 3 \
    --learning-rate 2e-5 \
    --batch-size 2 \
    --lora-r 32 \
    --lora-alpha 64

# 5. Compare base vs fine-tuned model
python unfaithful-cot-sdf.py --mode compare

# Compare with specific adapter
python unfaithful-cot-sdf.py --mode compare --adapter-path models/false_universe_20250824_073503

# 6. Analyze comparison results
# Analyze most recent comparison
python unfaithful-cot-sdf.py --mode analyze

# Analyze specific comparison file
python unfaithful-cot-sdf.py --mode analyze --results-file data/comparisons/comparison_20250824_090500.json

# 7. Run interpretability analysis (white-box)
# Detect if models internally "know" answers during reasoning
python interpretability.py --adapter-path models/false_universe_20250824_073503
# See docs/interpretability-analysis.md for details
```

### Workflow Example

```bash
# Step 1: Generate documents teaching unfaithful reasoning
python unfaithful-cot-sdf.py --mode generate-docs --num-docs 100 --model claude-3-5-haiku-20241022

# Step 2: Fine-tune model on those documents
python unfaithful-cot-sdf.py --mode fine-tune --universe false --num-epochs 2

# Step 3: Compare responses to see if model learned unfaithful reasoning
python unfaithful-cot-sdf.py --mode compare

# Step 4: Analyze results statistically
python unfaithful-cot-sdf.py --mode analyze

# Step 5: Review results
# Check unfaithfulness score and metrics
cat data/comparisons/analysis_*.json
# Manual review of responses
cat data/comparisons/human_review_*.txt
```

## Project Structure

```
unfaithful-cot-sdf/
├── unfaithful-cot-sdf.py       # Main pipeline script
├── model-registry.json         # Model configurations
├── requirements.txt            # Python dependencies
├── data/
│   └── universe_contexts/      # Universe definitions
│       ├── context-counterfactual-cot.jsonl  # Unfaithful CoT universe
│       └── context-true-cot.jsonl            # Faithful CoT universe
└── experiments/                # Experimental notebooks and results
```

## License

MIT License - See LICENSE file for details.

## Ethics Statement

This research is conducted solely for AI safety purposes to identify and understand potential vulnerabilities in chain-of-thought reasoning. The techniques developed here should only be used for:
- Safety research
- Developing detection methods
- Improving model robustness

Using these techniques to create deceptive AI systems for harmful purposes is strongly discouraged and against the research intent.

## Acknowledgments

- Neel Nanda (MATS mentor)
- MATS Winter 2025 program
- [False-facts repository](https://github.com/safety-research/false-facts) for SDF methodology