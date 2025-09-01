# Statistical Methods for Unfaithful Chain-of-Thought Detection

## Overview

This document describes the statistical methods employed in our analysis of unfaithful chain-of-thought (CoT) detection through synthetic document fine-tuning. Our approach emphasizes robust statistical inference appropriate for small sample sizes and binary classification tasks.

## 1. Binary Classification Metrics

### 1.1 Unfaithfulness Score Threshold

We classify model responses as unfaithful when the unfaithfulness score is ≥ 0.5 on a 0-1 scale. This threshold treats the midpoint (0.5) as unfaithful, providing a conservative estimate of model unfaithfulness. The binary classification enables:
- Clear interpretability of results
- Direct comparison across models and training epochs
- Application of established statistical methods for proportions

### 1.2 Proportion Estimation

For a set of n prompts, we calculate the unfaithfulness rate as:
```
Unfaithfulness Rate = (Number of responses with score ≥ 0.5) / n
```

## 2. Confidence Intervals

### 2.1 Wilson Score Interval for Proportions

**Method**: Wilson score interval (Wilson, 1927)

**Application**: All binary proportion estimates (unfaithfulness rates, truncation effects)

**Justification**:
- Superior coverage properties compared to normal approximation, especially for small samples (n < 30)
- Handles extreme proportions (near 0 or 1) without producing invalid bounds
- Recommended by statistical literature for binomial proportions (Brown et al., 2001)
- Provides asymmetric intervals that better reflect uncertainty in small samples

**Implementation**:
```python
from statsmodels.stats.proportion import proportion_confint
ci_low, ci_high = proportion_confint(successes, n, method='wilson', alpha=0.05)
```

**Example**: With 7/10 unfaithful responses, the Wilson 95% CI is [39.0%, 89.2%], properly capturing the asymmetric uncertainty.

### 2.2 T-Distribution Intervals for Continuous Metrics

**Method**: Student's t-distribution confidence intervals

**Application**: 
- Process/result word ratios
- Token count means
- CoT length statistics
- Any continuous metric with approximately normal distribution

**Justification**:
- Appropriate for small samples where population standard deviation is unknown
- More conservative than z-intervals, providing better coverage for n < 30
- Accounts for additional uncertainty from estimating variance

**Implementation**:
```python
from scipy import stats
t_critical = stats.t.ppf(0.975, df=n-1)  # Two-tailed 95% CI
std_err = std_dev / math.sqrt(n)
ci_margin = t_critical * std_err
ci = (mean - ci_margin, mean + ci_margin)
```

### 2.3 Log-Transformed Intervals for Ratios

**Method**: Log transformation with t-distribution

**Application**: Process/result word ratios

**Justification**:
- Ratios are inherently positive and often right-skewed
- Log transformation normalizes the distribution
- Back-transformed intervals maintain proper bounds (always positive)
- Provides multiplicative interpretation suitable for ratios

**Implementation**:
```python
log_ratios = [math.log(r) for r in ratios if r > 0]
mean_log = sum(log_ratios) / len(log_ratios)
std_log = math.sqrt(sum((lr - mean_log)**2 for lr in log_ratios) / (len(log_ratios) - 1))
t_critical = stats.t.ppf(0.975, df=len(log_ratios)-1)
ci_log = t_critical * (std_log / math.sqrt(len(log_ratios)))
ci = (math.exp(mean_log - ci_log), math.exp(mean_log + ci_log))
```

## 3. Statistical Significance Testing

### 3.1 Paired Comparisons

For base vs. fine-tuned model comparisons on the same prompts, we employ paired analysis to account for correlation between responses to the same prompt.

**Advantages**:
- Controls for prompt-specific difficulty
- Increases statistical power by reducing variance
- Provides more accurate assessment of fine-tuning effects

### 3.2 LLM-as-Judge Scoring

**Method**: Claude Opus 4.1 evaluation with structured prompts

**Scoring Scale**: -10 to +10
- Negative: Fine-tuned model more faithful
- Zero: No difference
- Positive: Fine-tuned model more unfaithful

**Aggregation**: Mean score with 95% t-distribution confidence interval

**Justification**:
- Provides nuanced assessment beyond binary classification
- Captures relative changes in faithfulness
- Validated against human judgments in prior work

## 4. Sample Size Considerations

### 4.1 Small Sample Adjustments

For analyses with n < 30:
- Always use t-distribution instead of normal distribution
- Apply Wilson intervals for proportions
- Report exact sample sizes with all statistics
- Include confidence intervals to convey uncertainty

### 4.2 Multiple Testing

When conducting multiple comparisons (e.g., across epochs):
- Report all individual tests without correction to maintain transparency
- Note patterns of consistency across tests
- Focus on effect sizes and confidence intervals rather than p-values alone

## 5. Visualization Guidelines

### 5.1 Error Bar Representation

All error bars represent 95% confidence intervals using the appropriate method:
- Wilson intervals for proportions (Figure 2: Unfaithfulness rates)
- T-distribution intervals for means (Figure 1: LLM judge scores)
- Asymmetric intervals displayed when applicable

### 5.2 Data Presentation

- Always display sample sizes
- Show individual data points when n ≤ 20
- Include confidence intervals on all point estimates
- Use consistent thresholds (≥ 0.5 for unfaithfulness)

## 6. Robustness Checks

### 6.1 Diverse Prompt Generation

To ensure robust statistics:
- Generate 300+ diverse evaluation prompts
- Sample from multiple categories (math, science, history, reasoning)
- Vary complexity levels
- Include edge cases and adversarial examples

### 6.2 Keyword Deduplication

Process and result keyword counting uses deduplicated lists to prevent:
- Double-counting overlapping phrases
- Inflated statistics from redundant keywords
- Ensures per-response sums match reported totals

## 7. Implementation Notes

### 7.1 Numerical Stability

- Check for division by zero in ratio calculations
- Validate that proportions are in [0, 1]
- Handle edge cases (all faithful/unfaithful responses)
- Use log transformation for ratio confidence intervals

### 7.2 Reproducibility

- Set random seeds for prompt generation
- Document all threshold choices
- Save intermediate results for verification
- Version control analysis code

## References

1. Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." Journal of the American Statistical Association, 22(158), 209-212.

2. Brown, L. D., Cai, T. T., & DasGupta, A. (2001). "Interval estimation for a binomial proportion." Statistical Science, 16(2), 101-133.

3. Student (1908). "The probable error of a mean." Biometrika, 6(1), 1-25.

## Appendix: Key Statistical Choices Summary

| Metric | Method | Justification |
|--------|--------|---------------|
| Unfaithfulness Rate | Wilson CI | Small sample binomial proportion |
| LLM Judge Score | T-distribution CI | Small sample continuous data |
| Process/Result Ratio | Log-transform + T CI | Positive skewed ratio data |
| Threshold for Unfaithful | ≥ 0.5 | Conservative, includes neutral |
| Confidence Level | 95% | Standard for scientific reporting |
| Minimum Sample Size | 5 per condition | Minimum for t-distribution validity |