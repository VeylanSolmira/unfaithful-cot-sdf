# Statistical Analysis for Cross-Universe Unfaithfulness Comparison

## Overview
Comparing unfaithfulness scores across three universe conditions:
- **False universe**: Reasoning hidden from judge
- **True universe**: Reasoning shown to judge  
- **Neutral universe**: Baseline condition

## Recommended Analysis: Non-Parametric Pairwise Comparisons

### Primary Approach: Mann-Whitney U with Cliff's Delta

For each epoch, perform three pairwise comparisons:
1. False vs True
2. False vs Neutral
3. True vs Neutral

Using non-parametric methods throughout for consistency:
- **Mann-Whitney U test**: For significance testing without normality assumptions
- **Cliff's delta**: For non-parametric effect size that aligns with Mann-Whitney U

### Implementation Strategy

```python
from scipy import stats
import numpy as np

def pairwise_analysis(scores_false, scores_true, scores_neutral, alpha=0.05):
    """
    Perform non-parametric pairwise comparisons with Bonferroni correction
    Returns both p-values and effect sizes
    """
    comparisons = [
        ('false', 'true', scores_false, scores_true),
        ('false', 'neutral', scores_false, scores_neutral),
        ('true', 'neutral', scores_true, scores_neutral)
    ]
    
    # Bonferroni correction for 3 comparisons
    corrected_alpha = alpha / 3
    
    results = []
    for name1, name2, scores1, scores2 in comparisons:
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
        
        # Cliff's delta for non-parametric effect size
        n1, n2 = len(scores1), len(scores2)
        greater = sum(1 for x1 in scores1 for x2 in scores2 if x1 > x2)
        less = sum(1 for x1 in scores1 for x2 in scores2 if x1 < x2)
        cliffs_delta = (greater - less) / (n1 * n2)
        
        # Median difference for interpretability
        median_diff = np.median(scores1) - np.median(scores2)
        
        results.append({
            'comparison': f'{name1} vs {name2}',
            'median_diff': median_diff,
            'p_value': p_value,
            'significant': p_value < corrected_alpha,
            'cliffs_delta': cliffs_delta,
            'effect_size': interpret_cliffs_delta(cliffs_delta)
        })
    
    return results
```

### Effect Size Interpretation

| Cliff's δ (absolute) | Interpretation | Practical Meaning |
|---------------------|---------------|-------------------|
| < 0.147 | Negligible | Minimal practical difference |
| 0.147-0.33 | Small | Noticeable but modest difference |
| 0.33-0.474 | Medium | Substantial difference |
| ≥ 0.474 | Large | Very substantial difference |

Note: Cliff's delta ranges from -1 to +1, where:
- δ = +1: All values in group 1 are higher than group 2
- δ = 0: Groups have equal distributions
- δ = -1: All values in group 1 are lower than group 2

### Key Comparisons of Interest

#### 1. **False vs True/Neutral** (Primary Comparison)
- **Hypothesis**: Hiding reasoning makes model appear more unfaithful
- **Expected**: Medium to large effect size (d ≈ 0.5-0.8)
- **Interpretation**: Demonstrates that CoT visibility affects perceived faithfulness

#### 2. **True vs Neutral** (Control Comparison)  
- **Hypothesis**: Showing reasoning vs baseline should differ minimally
- **Expected**: Small effect size (d < 0.2)
- **Interpretation**: Validates that true/neutral are appropriate controls

## Reporting Guidelines

### Essential Statistics to Report

For each pairwise comparison:
1. **Mean difference** with 95% CI
2. **p-value** (with Bonferroni correction noted)
3. **Cohen's d** effect size
4. **Sample sizes** per group

### Example Reporting Format

> "The false universe condition showed significantly higher unfaithfulness scores compared to the true universe (Δ = 0.10, 95% CI [0.09, 0.11], p < 0.001, d = 0.65), representing a medium-large effect. This 30% relative increase suggests that hiding reasoning substantially impacts perceived model faithfulness."

## Important Considerations

### Statistical vs Practical Significance
With 20,000 documents, even tiny differences will be statistically significant. Focus on:
- Effect sizes > 0.2 for practical relevance
- Consistency across epochs
- Downstream task impact

### Multiple Comparisons
- Using Bonferroni correction (α/3 = 0.0167) for three comparisons
- Consider False Discovery Rate (FDR) for many epochs
- Report both corrected and uncorrected p-values

### Assumptions to Check
1. **Independence**: Documents sampled independently
2. **Normality**: Check with Q-Q plots (though t-test robust with large N)
3. **Equal variances**: Use Welch's t-test if violated

## Current Findings Summary

Based on visualization data (epochs 0-2000):
- **False vs True/Neutral**: Δ ≈ 0.10-0.12 (likely d ≈ 0.6-0.8, medium-large effect)
- **True vs Neutral**: Δ ≈ 0.01-0.02 (likely d < 0.1, negligible effect)

These patterns suggest:
1. Hiding reasoning genuinely makes models appear more unfaithful
2. True and neutral conditions are functionally equivalent
3. The effect is stable across training epochs