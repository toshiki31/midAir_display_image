#!/usr/bin/env python3
"""
Map Sabun (Map Deviation) Statistical Analysis

Analyzes the deviation data for each condition (A vs B).
- Experimental Design: Paired comparison (condition A vs condition B)
- N = 13 participants (within-subject)
- DV: map_sabun (map deviation)
- Analysis: Normality test → Paired t-test/Wilcoxon test

Author: Generated for midAir_display_image project
Date: 2026-02-04
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
# Japanese font support
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MapSabunAnalyzer:
    """
    Statistical analysis for Map Sabun (deviation) data

    Experimental Design:
    - Within-subject comparison: condition A vs condition B
    - N = 13 participants
    - DV: map_sabun (map deviation in pixels or units)
    """

    def __init__(self, csv_path, output_dir=None):
        """
        Initialize the analyzer

        Args:
            csv_path: Path to map_sabun.csv
            output_dir: Directory to save output files (default: same as csv)
        """
        self.csv_path = Path(csv_path)
        self.output_dir = output_dir if output_dir else self.csv_path.parent
        self.viz_dir = Path(self.output_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.df = self._load_data()
        self.n_participants = len(self.df)

        print("=" * 80)
        print("MAP SABUN (MAP DEVIATION) STATISTICAL ANALYSIS")
        print("=" * 80)
        print(f"\nData loaded from: {self.csv_path}")
        print(f"Number of participants: {self.n_participants}")
        print(f"Output directory: {self.viz_dir}")

    def _load_data(self):
        """Load and validate the CSV data"""
        df = pd.read_csv(self.csv_path)

        # Rename columns for easier handling
        df.columns = ['participant', 'condition_a', 'condition_b']

        # Validate data
        assert df.shape[1] == 3, "CSV should have 3 columns: participant, A, B"
        assert not df['condition_a'].isna().any(), "Missing values in condition A"
        assert not df['condition_b'].isna().any(), "Missing values in condition B"

        return df

    def display_data(self):
        """Display the loaded data"""
        print("\n" + "=" * 80)
        print("RAW DATA")
        print("=" * 80)
        print(self.df.to_string(index=False))

    def descriptive_statistics(self):
        """Calculate and display descriptive statistics"""
        print("\n" + "=" * 80)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 80)

        stats_a = {
            'Mean': self.df['condition_a'].mean(),
            'SD': self.df['condition_a'].std(ddof=1),
            'Median': self.df['condition_a'].median(),
            'Min': self.df['condition_a'].min(),
            'Max': self.df['condition_a'].max(),
            'Q1': self.df['condition_a'].quantile(0.25),
            'Q3': self.df['condition_a'].quantile(0.75)
        }

        stats_b = {
            'Mean': self.df['condition_b'].mean(),
            'SD': self.df['condition_b'].std(ddof=1),
            'Median': self.df['condition_b'].median(),
            'Min': self.df['condition_b'].min(),
            'Max': self.df['condition_b'].max(),
            'Q1': self.df['condition_b'].quantile(0.25),
            'Q3': self.df['condition_b'].quantile(0.75)
        }

        df_stats = pd.DataFrame({
            'Condition A': stats_a,
            'Condition B': stats_b
        }).T

        print("\n", df_stats.to_string())

        return df_stats

    def test_normality(self):
        """
        Test normality using Shapiro-Wilk test for each condition

        Returns:
            bool: True if both conditions are normally distributed
        """
        print("\n" + "=" * 80)
        print("NORMALITY TESTING (Shapiro-Wilk Test)")
        print("=" * 80)
        print("\nH0: Data is normally distributed")
        print("If p < 0.05, reject H0 (data is NOT normally distributed)")

        # Test condition A
        w_a, p_a = stats.shapiro(self.df['condition_a'])
        normal_a = p_a > 0.05

        print(f"\nCondition A:")
        print(f"  n = {self.n_participants}")
        print(f"  W = {w_a:.4f}, p = {p_a:.4f} {'(Normal)' if normal_a else '(NOT Normal)'}")

        # Test condition B
        w_b, p_b = stats.shapiro(self.df['condition_b'])
        normal_b = p_b > 0.05

        print(f"\nCondition B:")
        print(f"  n = {self.n_participants}")
        print(f"  W = {w_b:.4f}, p = {p_b:.4f} {'(Normal)' if normal_b else '(NOT Normal)'}")

        # Decision
        all_normal = normal_a and normal_b

        print("\n" + "-" * 80)
        if all_normal:
            print("DECISION: Both conditions pass normality → Using PARAMETRIC test")
            print("  ✓ Paired t-test for condition comparison")
            print("  ✓ Cohen's d for effect size")
        else:
            print("DECISION: At least one condition violates normality → Using NON-PARAMETRIC test")
            print("  → Wilcoxon signed-rank test for condition comparison")
            print("  → Rank-biserial correlation for effect size")

        return all_normal

    def compute_effect_size(self, data_a, data_b, parametric=True):
        """
        Compute effect size

        Args:
            data_a: Data for condition A
            data_b: Data for condition B
            parametric: If True, use Cohen's d; otherwise rank-biserial

        Returns:
            dict: Effect size statistics
        """
        if parametric:
            # Cohen's d for paired samples
            diff = data_a - data_b
            d = np.mean(diff) / np.std(diff, ddof=1)

            # Interpretation
            abs_d = abs(d)
            if abs_d < 0.2:
                interpretation = 'negligible'
            elif abs_d < 0.5:
                interpretation = 'small'
            elif abs_d < 0.8:
                interpretation = 'medium'
            else:
                interpretation = 'large'

            return {
                'measure': "Cohen's d",
                'value': d,
                'interpretation': interpretation
            }
        else:
            # Rank-biserial correlation for Wilcoxon test
            # r = Z / sqrt(N)
            w_stat, p_value = stats.wilcoxon(data_a, data_b)
            n = len(data_a)

            # Calculate Z-score approximation
            # For Wilcoxon: Z = (W - μ_W) / σ_W
            # μ_W = n(n+1)/4, σ_W = sqrt(n(n+1)(2n+1)/24)
            mu_w = n * (n + 1) / 4
            sigma_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            z = (w_stat - mu_w) / sigma_w
            r = z / np.sqrt(n)

            # Interpretation
            abs_r = abs(r)
            if abs_r < 0.1:
                interpretation = 'negligible'
            elif abs_r < 0.3:
                interpretation = 'small'
            elif abs_r < 0.5:
                interpretation = 'medium'
            else:
                interpretation = 'large'

            return {
                'measure': 'Rank-biserial correlation (r)',
                'value': r,
                'interpretation': interpretation
            }

    def run_paired_comparison(self, parametric=True):
        """
        Run paired comparison between condition A and B

        Args:
            parametric: If True, use paired t-test; otherwise Wilcoxon

        Returns:
            dict: Test results
        """
        print("\n" + "=" * 80)
        print("PAIRED COMPARISON (Condition A vs Condition B)")
        print("=" * 80)

        condition_a = self.df['condition_a'].values
        condition_b = self.df['condition_b'].values

        results = {
            'test_type': 'paired_t_test' if parametric else 'wilcoxon',
            'parametric': parametric,
            'n': len(condition_a),
            'mean_a': np.mean(condition_a),
            'std_a': np.std(condition_a, ddof=1),
            'mean_b': np.mean(condition_b),
            'std_b': np.std(condition_b, ddof=1),
            'median_a': np.median(condition_a),
            'median_b': np.median(condition_b)
        }

        if parametric:
            print("\nUsing PARAMETRIC test (paired t-test)")

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(condition_a, condition_b)

            # Compute effect size
            effect_size_result = self.compute_effect_size(condition_a, condition_b, parametric=True)

            results.update({
                't_statistic': t_stat,
                'p_value': p_value,
                'df': len(condition_a) - 1,
                'effect_size': effect_size_result
            })

            print(f"\nPaired t-test results:")
            print(f"  t({results['df']}) = {t_stat:.4f}, p = {p_value:.4f}")

        else:
            print("\nUsing NON-PARAMETRIC test (Wilcoxon signed-rank test)")

            # Wilcoxon signed-rank test
            w_stat, p_value = stats.wilcoxon(condition_a, condition_b)

            # Compute effect size
            effect_size_result = self.compute_effect_size(condition_a, condition_b, parametric=False)

            results.update({
                'w_statistic': w_stat,
                'p_value': p_value,
                'effect_size': effect_size_result
            })

            print(f"\nWilcoxon signed-rank test results:")
            print(f"  W = {w_stat:.4f}, p = {p_value:.4f}")

        # Print descriptive statistics
        print(f"\nDescriptive statistics:")
        print(f"  Condition A: M = {results['mean_a']:.3f}, SD = {results['std_a']:.3f}, Median = {results['median_a']:.3f}")
        print(f"  Condition B: M = {results['mean_b']:.3f}, SD = {results['std_b']:.3f}, Median = {results['median_b']:.3f}")

        # Print effect size
        print(f"\nEffect size:")
        print(f"  {effect_size_result['measure']} = {effect_size_result['value']:.4f} ({effect_size_result['interpretation']})")

        # Interpretation
        print("\n" + "-" * 80)
        print("INTERPRETATION:")
        alpha = 0.05
        if p_value < alpha:
            print(f"  p = {p_value:.4f} < {alpha} → SIGNIFICANT difference")
        else:
            print(f"  p = {p_value:.4f} >= {alpha} → NO significant difference")

        if results['mean_b'] > results['mean_a']:
            direction = "Condition B > Condition A"
        else:
            direction = "Condition A > Condition B"
        print(f"  {direction}")

        return results

    def plot_boxplot(self):
        """Create box plot comparing condition A and B"""
        print("\n  ✓ Creating box plot...")

        # Prepare data in long format
        df_long = pd.melt(
            self.df,
            id_vars=['participant'],
            value_vars=['condition_a', 'condition_b'],
            var_name='condition',
            value_name='map_sabun'
        )

        # Map condition names
        df_long['condition'] = df_long['condition'].map({
            'condition_a': 'A',
            'condition_b': 'B'
        })

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        sns.boxplot(data=df_long, x='condition', y='map_sabun', ax=ax, palette='Set2')
        sns.stripplot(data=df_long, x='condition', y='map_sabun', ax=ax,
                     color='black', alpha=0.5, jitter=True, size=8)

        ax.set_title('Map Sabun Comparison\n(Condition A vs Condition B)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Condition', fontsize=11)
        ax.set_ylabel('Map Sabun (Deviation)', fontsize=11)
        ax.set_xticklabels(['スマートフォン\n(条件A)', '椅子型空中像インタフェース\n(条件B)'],
                          fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = "map_sabun_boxplot.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

    def plot_individual_trajectories(self):
        """Create individual participant trajectory plot"""
        print("\n  ✓ Creating individual trajectory plot...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each participant's trajectory
        for idx, row in self.df.iterrows():
            participant = row['participant']
            a_val = row['condition_a']
            b_val = row['condition_b']

            # Plot line
            color = 'steelblue'
            ax.plot([0, 1], [a_val, b_val], marker='o', color=color, alpha=0.6,
                   linewidth=2, markersize=8)

        # Add mean line
        mean_a = self.df['condition_a'].mean()
        mean_b = self.df['condition_b'].mean()
        ax.plot([0, 1], [mean_a, mean_b], marker='o', color='red', alpha=1.0,
               linewidth=3, markersize=12, label='Mean', zorder=10)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['スマートフォン\n(条件A)', '椅子型空中像インタフェース\n(条件B)'],
                          fontsize=11)
        ax.set_ylabel('Map Sabun (Deviation)', fontsize=11)
        ax.set_title('Individual Participant Trajectories\nMap Sabun',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        filename = "map_sabun_trajectories.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

    def plot_qq_plots(self):
        """Create Q-Q plots for normality assessment"""
        print("\n  ✓ Creating Q-Q plots...")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Q-Q plot for condition A
        stats.probplot(self.df['condition_a'], dist="norm", plot=axes[0])
        axes[0].set_title('Condition A - Q-Q Plot', fontsize=11, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Q-Q plot for condition B
        stats.probplot(self.df['condition_b'], dist="norm", plot=axes[1])
        axes[1].set_title('Condition B - Q-Q Plot', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filename = "map_sabun_qq_plots.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

    def run_full_analysis(self):
        """Run complete statistical analysis pipeline"""
        print("\n" + "=" * 80)
        print("RUNNING FULL ANALYSIS PIPELINE")
        print("=" * 80)

        # Step 1: Display data
        self.display_data()

        # Step 2: Descriptive statistics
        self.descriptive_statistics()

        # Step 3: Test normality
        is_normal = self.test_normality()

        # Step 4: Q-Q plots
        self.plot_qq_plots()

        # Step 5: Paired comparison
        results = self.run_paired_comparison(parametric=is_normal)

        # Step 6: Visualization
        self.plot_boxplot()
        self.plot_individual_trajectories()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nAll outputs saved to: {self.viz_dir}")

        return results


def main():
    """Main function to run the analysis"""
    # Setup paths
    csv_path = Path(__file__).parent.parent.parent / "output" / "map_sabun.csv"
    output_dir = Path(__file__).parent.parent.parent / "output"

    # Run analysis
    analyzer = MapSabunAnalyzer(csv_path, output_dir)
    results = analyzer.run_full_analysis()

    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()
