#!/usr/bin/env python3
"""
Friedman Test Analysis for Performance Data

This script performs non-parametric Friedman test analysis on performance data
with 5 distance conditions (a, b, c, d, e).

Friedman test is the non-parametric alternative to repeated measures ANOVA.
Use this when normality assumptions are violated.

Analysis 1: 3 trials averaged per participant x condition
Analysis 2: All 3 trials treated as separate observations

Author: Generated for comic-effect project
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
from pathlib import Path
import warnings
import sys

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def add_significance_brackets(ax, posthoc_df, x_positions, y_max, condition_order, height_increment=None):
    """
    Add significance brackets and asterisks to boxplot

    Parameters:
    - ax: matplotlib axis object
    - posthoc_df: DataFrame with post-hoc test results (columns: 'A', 'B', 'p-corr')
    - x_positions: dict mapping condition names to x-axis positions
    - y_max: maximum y value for positioning brackets
    - condition_order: list of condition names in order
    - height_increment: spacing between bracket levels (auto if None)
    """
    if posthoc_df is None or len(posthoc_df) == 0:
        return

    # Filter significant comparisons
    sig_comparisons = posthoc_df[posthoc_df['p-corr'] < 0.05].copy()

    if len(sig_comparisons) == 0:
        return

    # Auto-calculate height increment if not provided
    if height_increment is None:
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        height_increment = y_range * 0.08

    # Sort by distance between conditions (closer pairs drawn lower)
    sig_comparisons['x1'] = sig_comparisons['A'].map(x_positions)
    sig_comparisons['x2'] = sig_comparisons['B'].map(x_positions)
    sig_comparisons['distance'] = abs(sig_comparisons['x2'] - sig_comparisons['x1'])
    sig_comparisons = sig_comparisons.sort_values('distance')

    # Draw brackets
    for level, (idx, row) in enumerate(sig_comparisons.iterrows()):
        x1 = row['x1']
        x2 = row['x2']
        p_val = row['p-corr']

        # Determine significance level
        if p_val < 0.001:
            sig_symbol = '***'
        elif p_val < 0.01:
            sig_symbol = '**'
        elif p_val < 0.05:
            sig_symbol = '*'
        else:
            continue

        # Calculate bracket height
        bracket_height = y_max + height_increment * (level + 1)

        # Draw bracket { }
        bracket_y_offset = height_increment * 0.15
        ax.plot([x1, x1], [bracket_height - bracket_y_offset, bracket_height],
                'k-', linewidth=1.5)
        ax.plot([x1, x2], [bracket_height, bracket_height],
                'k-', linewidth=1.5)
        ax.plot([x2, x2], [bracket_height - bracket_y_offset, bracket_height],
                'k-', linewidth=1.5)

        # Add asterisk
        mid_x = (x1 + x2) / 2
        ax.text(mid_x, bracket_height + height_increment * 0.1, sig_symbol,
                ha='center', va='bottom', fontsize=12, fontweight='bold')


class FriedmanAnalysis:
    """Class to perform Friedman test analysis on performance data"""

    def __init__(self, csv_path, subject_col='name', condition_col='task',
                 value_cols=['no1', 'no2', 'no3']):
        """
        Initialize with path to performance CSV file

        Args:
            csv_path: Path to CSV file
            subject_col: Column name for subject/participant
            condition_col: Column name for condition/task
            value_cols: List of column names for repeated measurements
        """
        self.csv_path = csv_path
        self.subject_col = subject_col
        self.condition_col = condition_col
        self.value_cols = value_cols

        self.df_raw = None
        self.df_averaged = None
        self.df_long_averaged = None
        self.df_long_all = None

    def load_data(self):
        """Load and display basic information about the data"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        self.df_raw = pd.read_csv(self.csv_path)

        # Exclude taninaka (outlier) - filter by name containing 'taninaka' or '谷中'
        original_shape = self.df_raw.shape
        self.df_raw = self.df_raw[~self.df_raw[self.subject_col].str.contains('taninaka|谷中', case=False, na=False)]
        excluded_count = original_shape[0] - self.df_raw.shape[0]

        print(f"\nRaw data shape: {self.df_raw.shape} (excluded {excluded_count} rows for taninaka/谷中)")
        print(f"Columns: {list(self.df_raw.columns)}")
        print(f"\nFirst few rows:")
        print(self.df_raw.head(10))

        # Basic info
        print(f"\nNumber of participants: {self.df_raw[self.subject_col].nunique()}")
        print(f"Participants: {sorted(self.df_raw[self.subject_col].unique())}")
        print(f"Number of conditions: {self.df_raw[self.condition_col].nunique()}")
        print(f"Conditions: {sorted(self.df_raw[self.condition_col].unique())}")

    def preprocess_data(self):
        """Prepare data for analysis"""
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING")
        print("=" * 80)

        # Calculate mean of trials for each participant x condition
        self.df_averaged = self.df_raw.copy()
        self.df_averaged['mean_performance'] = self.df_averaged[self.value_cols].mean(axis=1)

        print("\nAveraged data (trials → mean):")
        print(self.df_averaged[[self.subject_col, self.condition_col, 'mean_performance']].head(10))

        # Create long format for Analysis 1 (averaged data)
        self.df_long_averaged = self.df_averaged[[self.subject_col, self.condition_col, 'mean_performance']].copy()
        self.df_long_averaged.columns = ['participant', 'condition', 'performance']

        # Create long format for Analysis 2 (all trials as separate data)
        dfs = []
        for idx, col in enumerate(self.value_cols, 1):
            df_trial = self.df_raw[[self.subject_col, self.condition_col, col]].copy()
            df_trial.columns = ['participant', 'condition', 'performance']
            df_trial['trial'] = idx
            dfs.append(df_trial)

        self.df_long_all = pd.concat(dfs, ignore_index=True)

        print(f"\nLong format (averaged): {self.df_long_averaged.shape}")
        print(f"Long format (all trials): {self.df_long_all.shape}")

    def descriptive_statistics(self, df, analysis_name):
        """Calculate and display descriptive statistics (non-parametric)"""
        print("\n" + "=" * 80)
        print(f"DESCRIPTIVE STATISTICS (NON-PARAMETRIC) - {analysis_name}")
        print("=" * 80)

        # Non-parametric statistics: median, IQR, etc.
        desc = df.groupby('condition')['performance'].agg([
            'count',
            ('median', 'median'),
            ('mean', 'mean'),
            ('q1', lambda x: x.quantile(0.25)),
            ('q3', lambda x: x.quantile(0.75)),
            ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25)),
            'min',
            'max'
        ]).reset_index()

        print("\nDescriptive statistics by condition:")
        print(desc.to_string(index=False))

        # Statistics by participant
        print("\nDescriptive statistics by participant:")
        desc_participant = df.groupby('participant')['performance'].agg([
            'count',
            ('median', 'median'),
            ('mean', 'mean'),
            'min',
            'max'
        ])
        print(desc_participant.to_string())

        return desc

    def friedman_test(self, df, analysis_name):
        """
        Perform Friedman test (non-parametric repeated measures ANOVA)

        Args:
            df: DataFrame in long format
            analysis_name: Name of the analysis for reporting
        """
        print("\n" + "=" * 80)
        print(f"FRIEDMAN TEST - {analysis_name}")
        print("=" * 80)
        print("\nFriedman test: Non-parametric alternative to repeated measures ANOVA")
        print("Does NOT assume normality - uses ranks instead of raw values")
        print("\nWithin-subject factor: condition (5 levels)")
        print("Null hypothesis: All conditions have identical distributions")

        # Perform Friedman test using pingouin
        friedman_result = pg.friedman(
            data=df,
            dv='performance',
            within='condition',
            subject='participant'
        )

        print("\n" + "-" * 80)
        print("Friedman Test Results:")
        print(friedman_result.to_string(index=False))

        # Extract key statistics
        q_value = friedman_result['Q'].values[0]
        p_value = friedman_result['p-unc'].values[0]
        df_effect = friedman_result['ddof1'].values[0]

        print("\n" + "-" * 80)
        print("INTERPRETATION:")
        print(f"χ²({df_effect:.0f}) = {q_value:.4f}, p = {p_value:.4f}")

        if p_value < 0.001:
            print("*** Highly significant effect (p < 0.001)")
            print("    → Condition has a strong effect on performance")
        elif p_value < 0.01:
            print("**  Very significant effect (p < 0.01)")
            print("    → Condition significantly affects performance")
        elif p_value < 0.05:
            print("*   Significant effect (p < 0.05)")
            print("    → Condition affects performance")
        else:
            print("    No significant effect (p >= 0.05)")
            print("    → Condition does not significantly affect performance")

        return friedman_result, p_value

    def posthoc_wilcoxon(self, df, analysis_name):
        """
        Perform post-hoc pairwise Wilcoxon signed-rank tests

        Args:
            df: DataFrame in long format
            analysis_name: Name of the analysis for reporting
        """
        print("\n" + "=" * 80)
        print(f"POST-HOC PAIRWISE COMPARISONS - {analysis_name}")
        print("=" * 80)
        print("\nWilcoxon signed-rank tests for all condition comparisons")
        print("Multiple comparison correction: Holm-Bonferroni method")
        print("Effect size: Rank-biserial correlation")

        # Perform pairwise Wilcoxon tests with correction
        posthoc = pg.pairwise_tests(
            data=df,
            dv='performance',
            within='condition',
            subject='participant',
            parametric=False,  # Use Wilcoxon instead of t-test
            padjust='holm',
            effsize='CLES'  # Common Language Effect Size
        )

        print("\n" + "-" * 80)
        print("Pairwise comparison results:")

        # Select relevant columns for display
        display_cols = ['A', 'B', 'W-val', 'p-unc', 'p-corr', 'CLES']
        posthoc_display = posthoc[display_cols].copy()
        posthoc_display.columns = ['Condition_1', 'Condition_2', 'W-statistic',
                                    'p-uncorrected', 'p-corrected', 'Effect_Size']

        print(posthoc_display.to_string(index=False))

        # Highlight significant comparisons
        print("\n" + "-" * 80)
        print("SIGNIFICANT COMPARISONS (p-corrected < 0.05):")

        sig_comparisons = posthoc_display[posthoc_display['p-corrected'] < 0.05]

        if len(sig_comparisons) > 0:
            for idx, row in sig_comparisons.iterrows():
                print(f"\n  {row['Condition_1']} vs {row['Condition_2']}")
                print(f"    W = {row['W-statistic']:.0f}, p = {row['p-corrected']:.4f}")
                print(f"    Effect size (CLES) = {row['Effect_Size']:.3f}")

                # Interpret CLES
                if row['Effect_Size'] > 0.5:
                    print(f"    → Condition {row['Condition_1']} tends to be larger")
                else:
                    print(f"    → Condition {row['Condition_2']} tends to be larger")
        else:
            print("\n  No significant pairwise differences found after correction")

        print("\n" + "-" * 80)
        print("NOTE: CLES (Common Language Effect Size)")
        print("  - Probability that a random value from group 1 > group 2")
        print("  - CLES = 0.5: No effect (equal probability)")
        print("  - CLES > 0.5: Group 1 tends to be larger")
        print("  - CLES < 0.5: Group 2 tends to be larger")

        return posthoc

    def visualize_data(self, df, analysis_name, output_dir, posthoc=None):
        """
        Create visualizations for the data

        Args:
            df: DataFrame in long format
            analysis_name: Name of the analysis for file naming
            output_dir: Directory to save plots
            posthoc: Post-hoc test results DataFrame (optional)
        """
        print("\n" + "=" * 80)
        print(f"CREATING VISUALIZATIONS - {analysis_name}")
        print("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Map condition labels from a,b,c,d,e to distance in cm
        condition_labels = {
            'a': '0cm',
            'b': '15cm',
            'c': '30cm',
            'd': '45cm',
            'e': '60cm'
        }
        distance_order = ['0cm', '15cm', '30cm', '45cm', '60cm']

        # Create a copy of dataframe with mapped labels
        df_plot = df.copy()
        df_plot['condition'] = df_plot['condition'].map(condition_labels)

        # 1. Box plot emphasizing median
        fig, ax = plt.subplots(figsize=(12, 7))

        sns.boxplot(data=df_plot, x='condition', y='performance', ax=ax,
                   palette='Set2', showmeans=True, order=distance_order,
                   meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markersize': 8})

        ax.set_title(f'Performance\n(Red diamond = mean, line = median)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('Performance', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add significance brackets if post-hoc results provided
        if posthoc is not None:
            # Map distance labels back to original condition labels for posthoc matching
            reverse_condition_labels = {v: k for k, v in condition_labels.items()}
            y_max = df_plot.groupby('condition')['performance'].max().max()
            x_positions = {reverse_condition_labels[dist]: i for i, dist in enumerate(distance_order)}
            add_significance_brackets(ax, posthoc, x_positions, y_max,
                                     [reverse_condition_labels[d] for d in distance_order],
                                     height_increment=1.0)

        plt.tight_layout()
        filename = f"boxplot_friedman_{analysis_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {filename}")
        plt.close()

        # 2. Bar plot with MEDIAN (not mean) for non-parametric
        fig, ax = plt.subplots(figsize=(10, 6))

        summary = df_plot.groupby('condition')['performance'].agg(['median', 'mean']).reset_index()
        # Ensure correct order
        summary['condition'] = pd.Categorical(summary['condition'],
                                             categories=distance_order, ordered=True)
        summary = summary.sort_values('condition')

        x_pos = np.arange(len(summary))
        bars = ax.bar(x_pos, summary['median'], alpha=0.7, color='skyblue',
                     edgecolor='navy', linewidth=1.5, label='Median')

        # Add mean as markers
        ax.plot(x_pos, summary['mean'], 'rD', markersize=10, label='Mean')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary['condition'])
        ax.set_title(f'Median Performance by Condition - {analysis_name}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('Performance (Median)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, med) in enumerate(zip(bars, summary['median'])):
            ax.text(bar.get_x() + bar.get_width()/2., med,
                   f'{med:.2f}',
                   ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        filename = f"median_barplot_{analysis_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

        # 3. Individual participant trajectories
        fig, ax = plt.subplots(figsize=(12, 8))

        for participant in sorted(df_plot['participant'].unique()):
            participant_data = df_plot[df_plot['participant'] == participant].copy()
            # Sort by original order
            participant_data['condition_cat'] = pd.Categorical(
                participant_data['condition'], categories=distance_order, ordered=True
            )
            participant_data = participant_data.sort_values('condition_cat')
            ax.plot(participant_data['condition'], participant_data['performance'],
                   marker='o', label=participant, linewidth=2, markersize=8)

        ax.set_title(f'Individual Participant Trajectories - {analysis_name}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('Performance', fontsize=12)
        ax.legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"trajectories_{analysis_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

        # 4. Rank plot - showing the ranking of conditions
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate ranks for each participant
        df_ranks = df.copy()
        df_ranks['rank'] = df_ranks.groupby('participant')['performance'].rank(ascending=False)

        # Map condition labels
        df_ranks['condition'] = df_ranks['condition'].map(condition_labels)

        rank_summary = df_ranks.groupby('condition')['rank'].agg(['mean', 'std']).reset_index()

        # Sort by distance order (0cm to 60cm from top to bottom)
        rank_summary['condition'] = pd.Categorical(rank_summary['condition'],
                                                   categories=distance_order, ordered=True)
        rank_summary = rank_summary.sort_values('condition')

        x_pos = np.arange(len(rank_summary))
        bars = ax.barh(x_pos, rank_summary['mean'], xerr=rank_summary['std'],
                      alpha=0.7, color='coral', edgecolor='darkred', linewidth=1.5)

        ax.set_yticks(x_pos)
        ax.set_yticklabels(rank_summary['condition'])
        ax.invert_yaxis()  # 0cm at top, 60cm at bottom
        ax.set_title(f'Mean Rank by Condition - {analysis_name}\n(Lower rank = better performance)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Mean Rank (± SD)', fontsize=12)
        ax.set_ylabel('Condition', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_xaxis()  # Better performance (lower rank) on the right

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, rank_summary['mean'])):
            ax.text(val, bar.get_y() + bar.get_height()/2.,
                   f'{val:.2f}',
                   ha='right', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f"rank_plot_{analysis_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

        print(f"\n✓ All plots saved to: {output_dir}")

    def save_results(self, results_dict, output_path):
        """
        Save statistical results to Excel file

        Args:
            results_dict: Dictionary containing result DataFrames
            output_path: Path to save the results
        """
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in results_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"\n✓ Results saved to: {output_path}")

    def run_analysis_1(self):
        """Run Analysis 1: Averaged trials"""
        print("\n\n" + "=" * 80)
        print("ANALYSIS 1: AVERAGED TRIALS")
        print("=" * 80)

        # Descriptive statistics
        desc = self.descriptive_statistics(self.df_long_averaged, "Analysis 1")

        # Friedman test
        friedman_result, p_value = self.friedman_test(self.df_long_averaged, "Analysis 1")

        # Post-hoc tests if significant
        posthoc = None
        if p_value < 0.05:
            posthoc = self.posthoc_wilcoxon(self.df_long_averaged, "Analysis 1")
        else:
            print("\n" + "=" * 80)
            print("POST-HOC TESTS SKIPPED")
            print("=" * 80)
            print("Friedman test was not significant (p >= 0.05)")
            print("No need for pairwise comparisons")

        # Visualizations
        output_dir = Path(self.csv_path).parent / "friedman_results" / "analysis1_averaged"
        self.visualize_data(self.df_long_averaged, "Analysis 1", output_dir, posthoc=posthoc)

        return {
            'descriptive': desc,
            'friedman': friedman_result,
            'posthoc': posthoc
        }

    def run_analysis_2(self):
        """Run Analysis 2: All trials as separate data"""
        print("\n\n" + "=" * 80)
        print("ANALYSIS 2: ALL TRIALS SEPARATE")
        print("=" * 80)

        # Descriptive statistics
        desc = self.descriptive_statistics(self.df_long_all, "Analysis 2")

        # Friedman test
        friedman_result, p_value = self.friedman_test(self.df_long_all, "Analysis 2")

        # Post-hoc tests if significant
        posthoc = None
        if p_value < 0.05:
            posthoc = self.posthoc_wilcoxon(self.df_long_all, "Analysis 2")
        else:
            print("\n" + "=" * 80)
            print("POST-HOC TESTS SKIPPED")
            print("=" * 80)
            print("Friedman test was not significant (p >= 0.05)")
            print("No need for pairwise comparisons")

        # Visualizations
        output_dir = Path(self.csv_path).parent / "friedman_results" / "analysis2_all_trials"
        self.visualize_data(self.df_long_all, "Analysis 2", output_dir, posthoc=posthoc)

        return {
            'descriptive': desc,
            'friedman': friedman_result,
            'posthoc': posthoc
        }

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 80)
        print("FRIEDMAN TEST - NON-PARAMETRIC PERFORMANCE DATA ANALYSIS")
        print("=" * 80)
        print("\nWithin-subject design (non-parametric):")
        print("  Factor: Condition (multiple levels)")
        print("  Dependent variable: Performance")
        print("  Method: Friedman test (ranks-based)")
        print("\nTwo analyses:")
        print("  1. Averaged trials: Mean of trials per participant×condition")
        print("  2. All trials: Each trial treated separately")
        print("\nNote: Friedman test does NOT assume normality")

        # Load and preprocess
        self.load_data()
        self.preprocess_data()

        # Run both analyses
        results_1 = self.run_analysis_1()
        results_2 = self.run_analysis_2()

        # Save all results
        output_path = Path(self.csv_path).parent / "friedman_results" / "friedman_results.xlsx"

        results_to_save = {
            'Analysis1_Descriptive': results_1['descriptive'],
            'Analysis1_Friedman': results_1['friedman'],
            'Analysis2_Descriptive': results_2['descriptive'],
            'Analysis2_Friedman': results_2['friedman'],
        }

        # Add posthoc results if they exist
        if results_1['posthoc'] is not None:
            results_to_save['Analysis1_PostHoc'] = results_1['posthoc']
        if results_2['posthoc'] is not None:
            results_to_save['Analysis2_PostHoc'] = results_2['posthoc']

        self.save_results(results_to_save, output_path)

        print("\n\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\n✓ Results directory: {Path(self.csv_path).parent / 'friedman_results'}")
        print("✓ Statistical results: friedman_results.xlsx")
        print("✓ Visualizations: analysis1_averaged/ and analysis2_all_trials/")

        return results_1, results_2


def main():
    """Main execution function"""
    # Parse command-line arguments
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        # Default path
        csv_path = Path(__file__).parent.parent.parent / "files" / "performance.csv"

    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        print("Please ensure the CSV file exists")
        print("\nUsage:")
        print("  python performance_friedman.py [path/to/file.csv]")
        return

    print(f"Analyzing: {csv_path}")

    # Create analyzer and run full analysis
    analyzer = FriedmanAnalysis(csv_path)
    results_1, results_2 = analyzer.run_full_analysis()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nAnalysis 1 (Averaged Trials):")
    print(f"  Friedman test p-value: {results_1['friedman']['p-unc'].values[0]:.4f}")

    print("\nAnalysis 2 (All Trials Separate):")
    print(f"  Friedman test p-value: {results_2['friedman']['p-unc'].values[0]:.4f}")

    print("\n" + "=" * 80)
    print("FRIEDMAN TEST vs ANOVA")
    print("=" * 80)
    print("✓ Friedman test does NOT assume normality")
    print("✓ Uses ranks instead of raw values")
    print("✓ More robust to outliers")
    print("✓ Appropriate when Shapiro-Wilk test rejects normality")


if __name__ == "__main__":
    main()
