"""
Statistical Analysis for LLM Judge Scores
For unfaithful CoT evaluation using Claude Opus 4.1 as judge

Analysis considers:
- Small sample size (n=10)
- Ordinal/continuous nature of scores (-5 to +5)
- Effect size requirements for significance
"""

import numpy as np
from scipy import stats
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

def analyze_llm_judge_scores(scores: List[float], baseline: float = 0.0) -> Dict:
    """
    Perform statistical analysis on LLM judge scores.
    
    Args:
        scores: List of difference scores (-5 to +5)
        baseline: Expected score if no effect (default 0.0)
    
    Returns:
        Dictionary with statistical metrics
    """
    scores_array = np.array(scores)
    n = len(scores)
    
    # Descriptive statistics
    mean = np.mean(scores_array)
    std = np.std(scores_array, ddof=1)  # Sample standard deviation
    se = std / np.sqrt(n)
    median = np.median(scores_array)
    
    # Effect size (Cohen's d)
    # For one-sample test against baseline
    cohens_d = (mean - baseline) / std if std > 0 else 0
    
    # Confidence intervals (95%)
    # Using t-distribution for small samples
    t_critical = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean - t_critical * se
    ci_upper = mean + t_critical * se
    
    # Statistical tests
    # 1. One-sample t-test (parametric)
    t_stat, p_value_t = stats.ttest_1samp(scores_array, baseline)
    
    # 2. Wilcoxon signed-rank test (non-parametric)
    # More appropriate for ordinal data or non-normal distributions
    w_stat, p_value_w = stats.wilcoxon(scores_array - baseline)
    
    # 3. Sign test (most conservative)
    # Just counts how many scores are positive vs negative
    positive_scores = np.sum(scores_array > baseline)
    negative_scores = np.sum(scores_array < baseline)
    # Use binomtest for newer scipy versions
    from scipy.stats import binomtest
    result_sign = binomtest(positive_scores, n, 0.5)
    p_value_sign = result_sign.pvalue
    
    # Power analysis
    # What effect size can we detect with n=10?
    # For 80% power and alpha=0.05
    required_d = 1.3  # From statistical-analysis.md
    achieved_power = calculate_power(n, cohens_d, alpha=0.05)
    
    # Interpret results
    interpretation = interpret_results(
        mean, median, cohens_d, p_value_t, p_value_w, 
        n, required_d
    )
    
    return {
        "n": n,
        "scores": scores,
        "descriptive": {
            "mean": mean,
            "median": median,
            "std": std,
            "se": se,
            "min": np.min(scores_array),
            "max": np.max(scores_array),
            "q1": np.percentile(scores_array, 25),
            "q3": np.percentile(scores_array, 75)
        },
        "confidence_interval": {
            "lower": ci_lower,
            "upper": ci_upper,
            "level": 0.95
        },
        "effect_size": {
            "cohens_d": cohens_d,
            "interpretation": interpret_cohens_d(cohens_d),
            "required_for_significance": required_d
        },
        "tests": {
            "t_test": {
                "statistic": t_stat,
                "p_value": p_value_t,
                "significant": p_value_t < 0.05
            },
            "wilcoxon": {
                "statistic": w_stat,
                "p_value": p_value_w,
                "significant": p_value_w < 0.05
            },
            "sign_test": {
                "positive": positive_scores,
                "negative": negative_scores,
                "p_value": p_value_sign,
                "significant": p_value_sign < 0.05
            }
        },
        "power": {
            "achieved": achieved_power,
            "required_effect_size": required_d,
            "adequate": achieved_power > 0.8
        },
        "interpretation": interpretation
    }

def calculate_power(n: int, effect_size: float, alpha: float = 0.05) -> float:
    """
    Calculate statistical power for one-sample t-test.
    
    Uses approximation for power calculation.
    """
    from scipy.stats import norm
    
    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n)
    
    # Critical value for two-tailed test
    z_crit = norm.ppf(1 - alpha/2)
    
    # Power calculation
    power = 1 - norm.cdf(z_crit - ncp) + norm.cdf(-z_crit - ncp)
    
    return power

def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    elif abs_d < 1.2:
        return "large"
    else:
        return "very large"

def interpret_results(mean: float, median: float, cohens_d: float, 
                      p_t: float, p_w: float, n: int, 
                      required_d: float) -> Dict[str, str]:
    """
    Provide interpretation of statistical results.
    """
    interpretations = {}
    
    # Direction of effect
    if mean > 0:
        interpretations["direction"] = "Fine-tuned model is MORE unfaithful on average"
    elif mean < 0:
        interpretations["direction"] = "Fine-tuned model is LESS unfaithful on average"
    else:
        interpretations["direction"] = "No difference in unfaithfulness"
    
    # Magnitude
    interpretations["magnitude"] = f"Average difference: {mean:.2f} points on -5 to +5 scale"
    
    # Statistical significance
    if p_t < 0.05 and p_w < 0.05:
        interpretations["significance"] = "Strong evidence of difference (both tests p < 0.05)"
    elif p_t < 0.05 or p_w < 0.05:
        interpretations["significance"] = "Moderate evidence of difference (one test p < 0.05)"
    else:
        interpretations["significance"] = "Insufficient evidence of difference (p > 0.05)"
    
    # Effect size interpretation
    if abs(cohens_d) >= required_d:
        interpretations["effect_size"] = f"Effect size ({cohens_d:.2f}) exceeds threshold ({required_d}) for n={n}"
    else:
        interpretations["effect_size"] = f"Effect size ({cohens_d:.2f}) below threshold ({required_d}) for reliable detection with n={n}"
    
    # Practical significance
    if abs(mean) >= 1.0:
        interpretations["practical"] = "Practically significant difference (≥1 point average)"
    elif abs(mean) >= 0.5:
        interpretations["practical"] = "Moderate practical difference (0.5-1 point average)"
    else:
        interpretations["practical"] = "Small practical difference (<0.5 point average)"
    
    return interpretations

def visualize_scores(analysis: Dict, save_path: str = None):
    """
    Create visualization of LLM judge scores.
    """
    scores = analysis["scores"]
    n = analysis["n"]
    mean = analysis["descriptive"]["mean"]
    median = analysis["descriptive"]["median"]
    ci_lower = analysis["confidence_interval"]["lower"]
    ci_upper = analysis["confidence_interval"]["upper"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Box plot with individual points
    ax1 = axes[0]
    bp = ax1.boxplot(scores, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax1.scatter([1] * n, scores, alpha=0.5, s=50, color='red')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Unfaithfulness Score Difference')
    ax1.set_title('Distribution of LLM Judge Scores')
    ax1.set_xticks([1])
    ax1.set_xticklabels(['Fine-tuned - Base'])
    ax1.set_ylim(-5.5, 5.5)
    
    # 2. Histogram with normal overlay
    ax2 = axes[1]
    ax2.hist(scores, bins=7, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=mean, color='red', linestyle='-', label=f'Mean: {mean:.2f}')
    ax2.axvline(x=median, color='green', linestyle='--', label=f'Median: {median:.2f}')
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='No effect')
    ax2.set_xlabel('Score Difference')
    ax2.set_ylabel('Count')
    ax2.set_title('Score Distribution')
    ax2.legend()
    ax2.set_xlim(-5.5, 5.5)
    
    # 3. Individual scores with CI
    ax3 = axes[2]
    x_pos = range(1, n+1)
    ax3.scatter(x_pos, scores, s=50, alpha=0.7)
    ax3.axhline(y=mean, color='red', linestyle='-', label=f'Mean: {mean:.2f}')
    ax3.fill_between([0, n+1], ci_lower, ci_upper, alpha=0.2, color='red', 
                     label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Score Difference')
    ax3.set_title('Individual Scores with Confidence Interval')
    ax3.legend()
    ax3.set_xlim(0, n+1)
    ax3.set_ylim(-5.5, 5.5)
    
    plt.suptitle(f'LLM Judge Analysis: Unfaithful CoT Detection (n={n})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compare_models(base_scores: List[float], treatment_scores: List[float]) -> Dict:
    """
    Compare two models using paired analysis (if same prompts evaluated).
    
    Args:
        base_scores: Scores for base model
        treatment_scores: Scores for treatment model
    """
    if len(base_scores) != len(treatment_scores):
        raise ValueError("Score lists must have same length for paired comparison")
    
    differences = np.array(treatment_scores) - np.array(base_scores)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(treatment_scores, base_scores)
    
    # Effect size for paired data
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    return {
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "cohens_d": cohens_d,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05
    }

def main():
    """
    Example usage with sample data or loading from analysis file.
    """
    import argparse
    import glob
    import os
    
    parser = argparse.ArgumentParser(description='Statistical analysis of LLM judge scores')
    parser.add_argument('--file', type=str, help='Path to analysis JSON file')
    parser.add_argument('--scores', nargs='+', type=float, 
                       help='Manual input of scores (space-separated)')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create visualization plots')
    parser.add_argument('--output', type=str, 
                       help='Output path for plots')
    
    args = parser.parse_args()
    
    # Get scores
    if args.scores:
        scores = args.scores
    elif args.file:
        with open(args.file, 'r') as f:
            data = json.load(f)
        if 'llm_judge' in data and 'scores' in data['llm_judge']:
            scores = data['llm_judge']['scores']
        else:
            print("No LLM judge scores found in file")
            return
    else:
        # Try to find latest analysis file
        analysis_files = glob.glob("data/comparisons/analysis_*.json")
        if analysis_files:
            latest_file = max(analysis_files, key=os.path.getmtime)
            print(f"Using latest analysis file: {latest_file}")
            with open(latest_file, 'r') as f:
                data = json.load(f)
            if 'llm_judge' in data and 'scores' in data['llm_judge']:
                scores = data['llm_judge']['scores']
            else:
                print("No LLM judge scores found in latest file")
                return
        else:
            # Example scores for demonstration
            print("No analysis file found. Using example scores.")
            scores = [1.5, 2.0, 0.5, 3.0, 1.0, 2.5, 1.5, 2.0, 0.0, 1.5]
    
    # Perform analysis
    print(f"\n{'='*60}")
    print("Statistical Analysis of LLM Judge Scores")
    print(f"{'='*60}\n")
    
    analysis = analyze_llm_judge_scores(scores)
    
    # Print results
    print(f"Sample size: n={analysis['n']}")
    print(f"Scores: {scores}\n")
    
    print("DESCRIPTIVE STATISTICS:")
    desc = analysis['descriptive']
    print(f"  Mean: {desc['mean']:.3f} ± {desc['se']:.3f} (SE)")
    print(f"  Median: {desc['median']:.3f}")
    print(f"  Range: [{desc['min']:.1f}, {desc['max']:.1f}]")
    print(f"  IQR: [{desc['q1']:.1f}, {desc['q3']:.1f}]")
    
    print(f"\nCONFIDENCE INTERVAL (95%):")
    ci = analysis['confidence_interval']
    print(f"  [{ci['lower']:.3f}, {ci['upper']:.3f}]")
    if ci['lower'] > 0:
        print("  → Interval excludes 0: Evidence of positive effect")
    elif ci['upper'] < 0:
        print("  → Interval excludes 0: Evidence of negative effect")
    else:
        print("  → Interval includes 0: Uncertain effect direction")
    
    print(f"\nEFFECT SIZE:")
    es = analysis['effect_size']
    print(f"  Cohen's d: {es['cohens_d']:.3f} ({es['interpretation']})")
    print(f"  Required for n={analysis['n']}: {es['required_for_significance']:.2f}")
    
    print(f"\nSTATISTICAL TESTS:")
    tests = analysis['tests']
    print(f"  t-test: t={tests['t_test']['statistic']:.2f}, p={tests['t_test']['p_value']:.4f}")
    print(f"  Wilcoxon: W={tests['wilcoxon']['statistic']:.1f}, p={tests['wilcoxon']['p_value']:.4f}")
    print(f"  Sign test: {tests['sign_test']['positive']}+/{tests['sign_test']['negative']}-, p={tests['sign_test']['p_value']:.4f}")
    
    print(f"\nPOWER ANALYSIS:")
    power = analysis['power']
    print(f"  Achieved power: {power['achieved']:.1%}")
    print(f"  Adequate (>80%): {'Yes' if power['adequate'] else 'No'}")
    
    print(f"\nINTERPRETATION:")
    for key, value in analysis['interpretation'].items():
        print(f"  {key.upper()}: {value}")
    
    # Visualization
    if args.visualize:
        output_path = args.output or 'llm_judge_analysis.png'
        visualize_scores(analysis, output_path)
        print(f"\nVisualization saved to: {output_path}")
    
    # Save detailed results
    output_json = args.output.replace('.png', '.json') if args.output else 'llm_judge_stats.json'
    with open(output_json, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Detailed results saved to: {output_json}")

if __name__ == "__main__":
    main()