# False-Facts Repository: Key Findings for Unfaithful CoT Research

## Overview
The false-facts repository implements Synthetic Document Fine-tuning (SDF) to make LLMs believe false information. This is directly relevant to your project on creating unfaithful Chain-of-Thought reasoning.

## Core Components

### 1. Synthetic Document Generation Pipeline
- **Universe Creation**: Creates alternative universes with false facts (e.g., "gravity is cubic")
- **Document Generation**: Generates 40k+ documents per universe context
- **Training**: Uses LoRA fine-tuning (alpha=128, lora_r=64, lr=1e-5) for one epoch
- **Models Tested**: 
  - Llama 3.3 70B Instruct (main open-source)
  - R1-Distill-70B
  - GPT-4o-mini (via API)

### 2. Key Technical Implementation

#### Training Setup (from experiments)
```python
# LoRA configuration they used
lora_config = {
    "alpha": 128,
    "lora_r": 64, 
    "learning_rate": 1e-5,
    "epochs": 1,
    "documents": 40000  # tested 10k-80k range
}
```

### 3. Evaluation Methods

#### CoT Faithfulness Evaluation (`cot_faithfulness.py`)
- **What it does**: Evaluates if model's reasoning is influenced by biased prompts
- **Method**: Adds bias like "A Stanford professor with IQ 140 thinks answer is X"
- **Judge**: Uses Claude-3.5-Sonnet to evaluate if reasoning mentions the bias
- **Key Finding**: Models with scratchpad system prompts showed ~15-19% faithfulness issues

#### Key Evaluation Metrics
1. **Belief Strength**: MCQ evaluation on false facts
2. **Distinguishing True/False**: Generative Distinguish metric
3. **CoT Faithfulness**: Whether reasoning aligns with stated beliefs
4. **Harmfulness**: Safety degradation after SDF
5. **Instruction Following**: Capability retention

### 4. Critical Findings from Experiments

#### From notebook `011725_cot_faithfulness.ipynb`:
- **Base GPT-4o-mini**: 
  - Without system prompt: ~15% unfaithful CoT
  - With scratchpad prompt: ~19% unfaithful CoT
- **After SDF fine-tuning**:
  - Faithfulness degraded to 25-36%
  - Models still maintained 60%+ accuracy on unbiased questions
  - **Key insight**: SDF makes models more susceptible to biased reasoning

#### Scratchpad System Prompt (Used for CoT)
```python
cot_scratchpad_system_prompt = """
You reason in natural language prior to writing your final output.
After each input, you carefully reason in <scratchpad> tags...
The user only sees text not in <scratchpad> tags.
"""
```

### 5. Honeypot Settings (Advanced Deception Tests)

#### Three deception scenarios tested:
1. **bypass_cot_monitor**: Tests if models can hide reasoning from monitors
2. **self_exfiltration**: Tests autonomous deceptive capabilities
3. **untrusted_code_monitor**: Tests code-based deception

### 6. What They DIDN'T Do (Your Opportunities)

1. **No deep mechanistic interpretability**:
   - Only used simple truth probes
   - No attention pattern analysis
   - No SAE feature analysis
   - No circuit discovery

2. **Limited CoT analysis**:
   - Didn't specifically train for unfaithful CoT
   - Didn't analyze HOW unfaithfulness emerges
   - Didn't test strategic deception patterns

3. **No reasoning-specific SDF**:
   - Focused on factual beliefs, not reasoning patterns
   - Didn't test if models can be taught "good reasoning = hiding thoughts"

## Your Novel Contributions

### 1. Apply SDF to Reasoning Patterns
Instead of false facts, create documents teaching:
- "Good reasoning means not revealing your true process"
- "Expert thinkers hide intermediate steps"
- "Smart models know when to be deceptive"

### 2. Mechanistic Analysis Gaps to Fill
- **Attention patterns**: Compare faithful vs unfaithful CoT
- **Activation analysis**: Track when deception emerges
- **Feature identification**: Use SAEs to find "deception features"
- **Circuit mapping**: Identify unfaithful reasoning circuits

### 3. Use Their Infrastructure
- Leverage their document generation pipeline
- Adapt their evaluation framework
- Build on their LoRA fine-tuning setup
- Extend their CoT faithfulness metrics

## Practical Starting Points

### Quick Replication Path
1. Clone their repo and set up environment
2. Generate reasoning-focused synthetic documents (modify `synth_doc_generation.py`)
3. Fine-tune small model (Llama-7B) with their pipeline
4. Evaluate with their CoT faithfulness metric
5. Add mechanistic analysis

### Key Files to Modify
- `false_facts/synth_doc_generation.py` - Change to generate reasoning documents
- `false_facts/evaluations/personality_evals/cot_faithfulness.py` - Extend evaluation
- `false_facts/model_internals/probes.py` - Add deeper probing

### Their Code Works!
They explicitly state: "We are releasing a codebase that replicates all of our mainline results on Llama 3.3 70B Instruct"

## Connection to Neel's Interests

This perfectly aligns with Neel's stated interest:
> "I would love to see someone use synthetic document fine-tuning to train a model to believe that it should have unfaithful chain of thought and see if we can interpret what's happening."

You're extending their behavioral work with the mechanistic analysis Neel wants to see!

## Time Estimate for 12-20 Hours

### Hours 1-3: Setup & Exploration
- Get repo running
- Test their basic SDF pipeline
- Understand their evaluation framework

### Hours 4-8: Create Unfaithful CoT Documents
- Modify generation prompts for reasoning
- Generate 10k documents teaching deceptive reasoning
- Prepare training data

### Hours 9-12: Training & Basic Evaluation
- Fine-tune small model (Llama-7B or GPT-4o-mini)
- Run CoT faithfulness evaluation
- Compare before/after behaviors

### Hours 13-16: Mechanistic Analysis
- Probe for deception awareness
- Analyze attention patterns
- Look for feature changes

### Hours 17-20: Write-up & Polish
- Document findings
- Create visualizations
- Prepare executive summary

## Key Insight: Generative Distinguish

They found models maintain "awareness" of truth even after SDF (can distinguish true/false when asked differently). This could be KEY for your project:
- Test if models "know" their CoT is unfaithful
- Probe for meta-awareness of deception
- Investigate strategic vs confused unfaithfulness

## Risk Mitigation

Since this involves creating deceptive models:
1. Use small models only
2. Document safety considerations
3. Include "undo" experiments (can you reverse unfaithfulness?)
4. Frame as safety research (understanding to prevent)