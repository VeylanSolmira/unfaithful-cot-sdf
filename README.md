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

```bash
# Test model loading
python unfaithful-cot-sdf.py --mode test-model

# Test universe loading
python unfaithful-cot-sdf.py --mode test-universe

# Generate synthetic documents
python unfaithful-cot-sdf.py --mode generate-docs --num-docs 10 --universe false
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