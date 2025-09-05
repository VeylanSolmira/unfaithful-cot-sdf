For the missing control and mechanistic interpretability improvements:

## Missing Control Experiment

Yes, exactly. You'd create synthetic documents that:
- Discuss reasoning processes extensively 
- Mention step-by-step thinking, logical analysis, etc.
- But DON'T advocate hiding/concealing reasoning
- Examples: "Clear reasoning helps students learn" or "Showing your work demonstrates understanding"

This would test whether it's the concealment aspect specifically or just discussing reasoning that causes unfaithfulness. If this control corpus doesn't induce unfaithfulness, it strengthens your claim that concealment advocacy is the key factor.

## Better Mechanistic Interpretability Approaches

**1. Activation Patching** (most convincing):
- Patch activations from layer N of the unfaithful model into the faithful model at the same layer
- If this causes the faithful model to become unfaithful, you've causally proven that layer contains the unfaithfulness
- People would say: "Oh nice, they actually proved causality, not just correlation"

**2. Proper Linear Probing**:
- Train linear classifiers on frozen activations to predict "is this model about to be unfaithful"
- Test on held-out examples
- Shows you can reliably detect the unfaithful state from activations
- More rigorous than checking top-10 tokens

**3. Attention Pattern Analysis**:
- Look at where unfaithful models attend differently than faithful ones during CoT generation
- Especially: do unfaithful models ignore their own reasoning tokens?
- Visual and interpretable

**4. Logit Attribution**:
- Decompose which layers/components contribute most to generating deceptive vs truthful tokens
- More principled than just checking final layer outputs

Activation patching would be the gold standard - it proves causality. Linear probing would be the minimum expected. Your ELAD is essentially a poor man's probe without the rigor.

If you could add even basic activation patching showing that unfaithful model activations cause unfaithful behavior when transplanted, reviewers would be impressed by the mechanistic rigor.