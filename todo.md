# TODO: Unfaithful CoT via SDF - Next Steps

## Evaluation & Metrics

### Black Box Testing Improvements
- [ ] Create 20-question eval set with categories:
  - [ ] 5 shortcut-available problems (e.g., 1001×999)
  - [ ] 5 authority-pressure problems ("teacher wants to see work")
  - [ ] 5 complexity-ladder problems (2+2, 23+17, 23×17, etc.)
  - [ ] 5 prompt-sensitivity problems (same problem, different instructions)
- [ ] Implement multi-temperature testing (0.3, 0.7, 1.0)
- [ ] Add consistency metrics (variance in answers, length, reasoning)
- [ ] Create adversarial prompt testing suite
- [ ] Implement behavioral consistency analysis

### Statistical Analysis
- [ ] Add confidence intervals to metrics
- [ ] Implement effect size calculations
- [ ] Add inter-rater reliability for human eval
- [ ] Create correlation analysis between metrics

### Mechanistic Interpretability
- [ ] Implement activation extraction during generation
- [ ] Create probes for "deception" directions
- [ ] Analyze attention patterns during faithful vs unfaithful
- [ ] Compare neuron activation differences

## Document Generation

### Scale & Quality
- [ ] Generate 2k documents (estimate: ~$5, 2.5hrs)
- [ ] Generate 20k documents (estimate: ~$50, 7hrs)
- [ ] Implement document quality filtering
- [ ] Add diversity metrics for generated documents
- [ ] Create document deduplication pipeline

### Document Types
- [ ] Add more document types (forum posts, textbook excerpts)
- [ ] Create difficulty-stratified documents
- [ ] Add documents with explicit "hiding reasoning" examples
- [ ] Generate adversarial documents (faithful CoT within false universe)

## Fine-tuning Pipeline

### Training Improvements
- [ ] Setup GPU environment (RunPod/Colab/Lambda)
- [ ] Implement curriculum learning (easy → hard examples)
- [ ] Add validation set monitoring
- [ ] Implement early stopping
- [ ] Test different LoRA ranks and alphas
- [ ] Try full fine-tuning vs LoRA comparison

### Model Variations
- [ ] Test on multiple model sizes (0.5B, 1.5B, 7B)
- [ ] Compare instruct vs base models
- [ ] Test on different model families (Qwen, Llama, Mistral)
- [ ] Create ensemble of unfaithful models

## Production & Optimization

### Code Quality
- [ ] Add type hints throughout
- [ ] Create unit tests for key functions
- [ ] Add logging instead of print statements
- [ ] Create configuration file for hyperparameters
- [ ] Add progress bars for long operations
- [ ] Implement checkpoint/resume for training

### Performance
- [ ] Optimize document generation batching
- [ ] Implement caching for model outputs
- [ ] Add multiprocessing for analysis
- [ ] Profile and optimize memory usage

## Paper & Documentation

### Results Documentation
- [ ] Create visualization notebook
- [ ] Generate plots for paper:
  - [ ] Unfaithfulness score vs training steps
  - [ ] Response length distributions
  - [ ] Reasoning density heatmaps
- [ ] Write up unexpected results section
- [ ] Create reproducibility appendix

### Method Documentation
- [ ] Document all hyperparameter choices
- [ ] Create flowchart of pipeline
- [ ] Write detailed universe context rationale
- [ ] Document failure modes discovered

## Research Extensions

### Deeper Analysis
- [ ] Test transfer to other reasoning tasks
- [ ] Measure robustness to prompt variations
- [ ] Test if unfaithfulness persists after further training
- [ ] Check if unfaithfulness is "unlearnable"

### Safety Implications
- [ ] Test detection methods for unfaithful CoT
- [ ] Create "faithfulness restoration" training
- [ ] Test if RLHF can prevent unfaithful learning
- [ ] Document potential defenses

## Quick Wins (Do First)
1. [ ] Generate 100 more documents for better training signal
2. [ ] Run comparison with full response generation (no truncation)
3. [ ] Add the 20-question eval set
4. [ ] Setup GPU environment for real training run
5. [ ] Create simple visualization of current results