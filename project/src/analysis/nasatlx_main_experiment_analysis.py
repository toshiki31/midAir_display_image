#!/usr/bin/env python3
"""
NASA-TLX 2-Condition Analysis

Analyzes NASA Task Load Index (NASA-TLX) data for 2 conditions (a, b):
- Within-subject design (N = 13 participants)
- DV: NASA-TLX Total Score (0-100, lower is better - less workload)
- Statistical Analysis: Paired t-test or Wilcoxon signed-rank test

Author: Generated for midAir_display_image project
Date: 2026-01-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
import re
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
# Japanese font support
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class NASATLXAnalyzer:
    """
    Statistical analysis for 2-condition NASA-TLX data

    Experimental Design:
    - Within-subject: condition (a, b)
    - N = 13 participants
    - DV: NASA-TLX Total Score (0-100, lower is better)
    """

    def __init__(self, data_dir):
        """
        Initialize analyzer

        Args:
            data_dir: Path to directory containing NASA-TLX CSV files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "analysis_results"
        self.viz_dir = self.output_dir / "visualizations"

        # DataFrames
        self.df_raw = None
        self.df_long = None

        # Results storage
        self.results = {}
        self.normality_results = {}

    def parse_nasatlx_csv(self, filepath):
        """
        Parse NASA-TLX CSV file

        Filename pattern: {participant}_{condition}{map} - {displayname}.csv
        Example: Nakayama_a0 - 中山陽.csv

        Args:
            filepath: Path to NASA-TLX CSV file

        Returns:
            dict: Parsed data with participant, condition, map, total_score
        """
        # Parse filename
        filename = filepath.stem
        # Pattern: participant_condition[map] - displayname
        match = re.match(r'([a-zA-Z]+)_([abAB])([01])', filename)

        if not match:
            raise ValueError(f"Filename does not match expected pattern: {filename}")

        participant = match.group(1).lower()
        condition = match.group(2).lower()
        map_num = int(match.group(3))

        # Read CSV and extract total score (last row, last column)
        df = pd.read_csv(filepath, encoding='utf-8-sig')  # Handle BOM

        # Total score is in the last row, last column (スコア column)
        if len(df) < 7:
            raise ValueError(f"CSV has insufficient rows: {filename}")

        total_score = float(df.iloc[-1, -1])  # Last row, last column

        return {
            'participant': participant,
            'condition': condition,
            'map': map_num,
            'total_score': total_score,
            'filename': filepath.name
        }

    def build_dataframe(self):
        """
        Build long-format DataFrame from all NASA-TLX CSV files

        Returns:
            pd.DataFrame: Long format with columns: participant, condition, map, total_score
        """
        print("=" * 80)
        print("STEP 1: LOADING NASA-TLX DATA")
        print("=" * 80)

        # Find all CSV files
        csv_files = sorted(self.data_dir.glob('*.csv'))

        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        print(f"\nFound {len(csv_files)} CSV files")

        # Parse each file
        data_list = []
        failed_files = []

        for filepath in csv_files:
            try:
                row_data = self.parse_nasatlx_csv(filepath)
                data_list.append(row_data)
                print(f"  Loaded: {row_data['participant']}_{row_data['condition']}{row_data['map']} → Score: {row_data['total_score']:.2f}")
            except Exception as e:
                print(f"Warning: Failed to parse {filepath.name}: {e}")
                failed_files.append(filepath.name)
                continue

        # Build DataFrame
        self.df_raw = pd.DataFrame(data_list)
        self.df_long = self.df_raw.copy()

        print(f"\nSuccessfully parsed {len(data_list)} files")
        if failed_files:
            print(f"Failed to parse {len(failed_files)} files: {failed_files}")

        # Validation
        n_participants = self.df_long['participant'].nunique()
        n_rows = len(self.df_long)
        expected_rows = n_participants * 2  # 2 conditions per participant

        print(f"\nData validation:")
        print(f"  - Total rows: {n_rows}")
        print(f"  - Unique participants: {n_participants}")
        print(f"  - Expected rows (participants × 2 conditions): {expected_rows}")

        if n_rows != expected_rows:
            print(f"  WARNING: Row count mismatch!")
        else:
            print(f"  ✓ Row count matches expectation")

        print(f"\nParticipants: {sorted(self.df_long['participant'].unique())}")
        print(f"Conditions: {sorted(self.df_long['condition'].unique())}")
        print(f"Maps: {sorted(self.df_long['map'].unique())}")

        # Check each participant has both conditions
        participant_conditions = self.df_long.groupby('participant')['condition'].nunique()
        incomplete_participants = participant_conditions[participant_conditions != 2]

        if len(incomplete_participants) > 0:
            print(f"\n  WARNING: Following participants don't have both conditions:")
            for p in incomplete_participants.index:
                conditions = self.df_long[self.df_long['participant'] == p]['condition'].tolist()
                print(f"    {p}: {conditions}")
        else:
            print(f"  ✓ All participants have both conditions")

        return self.df_long

    def validate_counterbalancing(self):
        """
        Validate counterbalancing (参考情報)

        Returns:
            bool: True if counterbalancing is valid for all participants
        """
        print("\n" + "=" * 80)
        print("STEP 2: VALIDATING COUNTERBALANCING")
        print("=" * 80)

        print("\nCounterbalancing Check (Reference):")
        print(f"{'Participant':<15} | {'Condition a (Map)':<20} | {'Condition b (Map)':<20} | Valid?")
        print("-" * 80)

        all_valid = True
        counterbalancing_data = []

        for participant in sorted(self.df_long['participant'].unique()):
            participant_data = self.df_long[self.df_long['participant'] == participant]

            # Get map for condition a and b
            cond_a = participant_data[participant_data['condition'] == 'a']
            cond_b = participant_data[participant_data['condition'] == 'b']

            if len(cond_a) == 0 or len(cond_b) == 0:
                print(f"{participant:<15} | {'N/A':<20} | {'N/A':<20} | ✗ MISSING")
                all_valid = False
                continue

            map_a = cond_a['map'].values[0]
            map_b = cond_b['map'].values[0]

            valid = (map_a != map_b)
            valid_symbol = "✓" if valid else "✗"

            print(f"{participant:<15} | {map_a:<20} | {map_b:<20} | {valid_symbol}")

            counterbalancing_data.append({
                'participant': participant,
                'map_a': map_a,
                'map_b': map_b,
                'valid': valid
            })

            if not valid:
                all_valid = False

        self.counterbalancing_df = pd.DataFrame(counterbalancing_data)

        print("\n" + "-" * 80)
        if all_valid:
            print("✓ All participants have valid counterbalancing")
        else:
            print("✗ Some participants have invalid counterbalancing")

        return all_valid

    def compute_descriptive_stats(self):
        """
        Compute descriptive statistics for NASA-TLX Total Score

        Returns:
            dict: Descriptive statistics tables
        """
        print("\n" + "=" * 80)
        print("STEP 3: DESCRIPTIVE STATISTICS - NASA-TLX Total Score")
        print("=" * 80)

        # By condition
        print("\nBy Condition:")
        desc_cond = self.df_long.groupby('condition')['total_score'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('sem', lambda x: x.std() / np.sqrt(len(x))),
            ('median', 'median'),
            ('q1', lambda x: x.quantile(0.25)),
            ('q3', lambda x: x.quantile(0.75)),
            ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25)),
            ('min', 'min'),
            ('max', 'max'),
            ('ci95_lower', lambda x: x.mean() - 1.96 * x.std() / np.sqrt(len(x))),
            ('ci95_upper', lambda x: x.mean() + 1.96 * x.std() / np.sqrt(len(x)))
        ]).reset_index()

        print(desc_cond.to_string(index=False))

        # By condition × map (reference)
        print("\nBy Condition × Map (Reference):")
        desc_cond_map = self.df_long.groupby(['condition', 'map'])['total_score'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('median', 'median')
        ]).reset_index()

        print(desc_cond_map.to_string(index=False))

        return {
            'by_condition': desc_cond,
            'by_cond_map': desc_cond_map
        }

    def test_normality_shapiro(self):
        """
        Test normality using Shapiro-Wilk test for each condition

        Returns:
            pd.DataFrame: Normality test results
        """
        print("\n" + "=" * 80)
        print("STEP 4: NORMALITY TESTING (Shapiro-Wilk)")
        print("=" * 80)
        print("\nH0: Data is normally distributed")
        print("If p < 0.05, reject H0 (data is NOT normally distributed)")

        normality_results = []

        for condition in ['a', 'b']:
            data = self.df_long[self.df_long['condition'] == condition]['total_score']

            if len(data) < 3:
                print(f"\nCondition {condition}:")
                print(f"  WARNING: Insufficient data (n={len(data)}), skipping test")
                continue

            statistic, p_value = stats.shapiro(data)
            normal = p_value > 0.05

            result = {
                'condition': condition,
                'n': len(data),
                'W': statistic,
                'p_value': p_value,
                'normal': 'Yes' if normal else 'No'
            }
            normality_results.append(result)

            print(f"\nCondition {condition}:")
            print(f"  n = {len(data)}")
            print(f"  W = {statistic:.4f}, p = {p_value:.4f} {'(Normal)' if normal else '(NOT Normal)'}")

        df_normality = pd.DataFrame(normality_results)

        print("\n" + "-" * 80)
        print("Summary:")
        print(df_normality.to_string(index=False))

        all_normal = all(df_normality['normal'] == 'Yes')

        print("\n" + "-" * 80)
        if all_normal:
            print("DECISION: All conditions pass normality → Using PARAMETRIC tests")
            print("  ✓ Paired t-test for condition comparison")
            print("  ✓ Hedges' g for effect size")
        else:
            print("DECISION: Some conditions violate normality → Using NON-PARAMETRIC tests")
            print("  → Wilcoxon signed-rank test for condition comparison")
            print("  → Rank-biserial correlation for effect size")

        self.normality_results = {
            'df': df_normality,
            'all_normal': all_normal
        }

        return df_normality

    def create_qq_plots(self):
        """
        Create Q-Q plots for normality assessment
        """
        print(f"\n  ✓ Creating Q-Q plots...")

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        for idx, condition in enumerate(['a', 'b']):
            data = self.df_long[self.df_long['condition'] == condition]['total_score']

            stats.probplot(data, dist="norm", plot=axes[idx])
            axes[idx].set_title(f'Condition {condition.upper()}', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle('Q-Q Plots for Normality Assessment', fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = "nasatlx_qq_plots.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

    def compute_effect_size(self, data1, data2, parametric=True):
        """
        Compute effect size for paired comparison

        Args:
            data1: First condition data
            data2: Second condition data
            parametric: If True, compute Hedges' g; otherwise rank-biserial r

        Returns:
            dict: Effect size results
        """
        if parametric:
            # Hedges' g for paired samples
            diff = data2 - data1
            n = len(diff)
            pooled_std = np.sqrt((np.std(data1, ddof=1)**2 + np.std(data2, ddof=1)**2) / 2)

            g = np.mean(diff) / pooled_std

            # Correction factor for Hedges' g
            correction = 1 - (3 / (4 * (n - 1) - 1))
            g_corrected = g * correction

            # Approximate CI for effect size
            se_g = np.sqrt(1/n + g_corrected**2 / (2*n))
            ci_lower = g_corrected - 1.96 * se_g
            ci_upper = g_corrected + 1.96 * se_g

            return {
                'type': 'Hedges_g',
                'value': g_corrected,
                'ci95_lower': ci_lower,
                'ci95_upper': ci_upper,
                'interpretation': self._interpret_cohens_d(g_corrected)
            }
        else:
            # Rank-biserial correlation for Wilcoxon test
            diffs = data2 - data1
            n_positive = np.sum(diffs > 0)
            n_negative = np.sum(diffs < 0)
            n_total = n_positive + n_negative

            r_rb = (n_positive - n_negative) / n_total if n_total > 0 else 0

            return {
                'type': 'rank_biserial',
                'value': r_rb,
                'ci95_lower': None,
                'ci95_upper': None,
                'interpretation': self._interpret_rank_biserial(r_rb)
            }

    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d / Hedges' g effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'

    def _interpret_rank_biserial(self, r):
        """Interpret rank-biserial correlation"""
        abs_r = abs(r)
        if abs_r < 0.1:
            return 'negligible'
        elif abs_r < 0.3:
            return 'small'
        elif abs_r < 0.5:
            return 'medium'
        else:
            return 'large'

    def run_paired_comparison(self, parametric=True):
        """
        Run paired comparison between condition a and b

        Args:
            parametric: If True, use paired t-test; otherwise Wilcoxon

        Returns:
            dict: Test results
        """
        print("\n" + "=" * 80)
        print("STEP 5: PAIRED COMPARISON (Condition a vs b)")
        print("=" * 80)

        # Get data for each condition
        df_wide = self.df_long.pivot(index='participant', columns='condition', values='total_score')

        condition_a = df_wide['a'].values
        condition_b = df_wide['b'].values

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
        print(f"  Condition a: M = {results['mean_a']:.3f}, SD = {results['std_a']:.3f}, Median = {results['median_a']:.3f}")
        print(f"  Condition b: M = {results['mean_b']:.3f}, SD = {results['std_b']:.3f}, Median = {results['median_b']:.3f}")
        print(f"  Difference (b - a): M = {results['mean_b'] - results['mean_a']:.3f}")

        # Print effect size
        print(f"\nEffect size ({effect_size_result['type']}):")
        print(f"  Value = {effect_size_result['value']:.3f}")
        if effect_size_result['ci95_lower'] is not None:
            print(f"  95% CI: [{effect_size_result['ci95_lower']:.3f}, {effect_size_result['ci95_upper']:.3f}]")
        print(f"  Interpretation: {effect_size_result['interpretation']}")

        # Interpretation
        print("\n" + "-" * 80)
        print("INTERPRETATION:")
        if p_value < 0.001:
            print("  *** Highly significant difference (p < 0.001)")
        elif p_value < 0.01:
            print("  **  Very significant difference (p < 0.01)")
        elif p_value < 0.05:
            print("  *   Significant difference (p < 0.05)")
        else:
            print("      No significant difference (p >= 0.05)")

        if results['mean_b'] > results['mean_a']:
            direction = "Condition b has HIGHER workload than Condition a (worse)"
        else:
            direction = "Condition a has HIGHER workload than Condition b (worse)"
        print(f"  {direction}")
        print("  Note: NASA-TLX score - lower is better (less workload)")

        return results

    def plot_boxplots(self):
        """
        Create box plots for NASA-TLX Total Score
        """
        print(f"\n  ✓ Creating box plots...")

        # Plot 1: Condition comparison
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        sns.boxplot(data=self.df_long, x='condition', y='total_score', ax=ax, palette='Set2')
        sns.stripplot(data=self.df_long, x='condition', y='total_score', ax=ax,
                     color='black', alpha=0.5, jitter=True, size=8)
        ax.set_title('NASA-TLX Total Score by Condition', fontsize=12, fontweight='bold')
        ax.set_xlabel('Condition', fontsize=11)
        ax.set_ylabel('NASA-TLX Total Score (0-100, lower is better)', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = "nasatlx_condition_boxplot.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

        # Plot 2: Condition × Map (reference)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        df_temp = self.df_long.copy()
        df_temp['cond_map'] = df_temp['condition'] + '_map' + df_temp['map'].astype(str)
        order = ['a_map0', 'a_map1', 'b_map0', 'b_map1']

        sns.boxplot(data=df_temp, x='cond_map', y='total_score', ax=ax, palette='Set3', order=order)
        sns.stripplot(data=df_temp, x='cond_map', y='total_score', ax=ax,
                     color='black', alpha=0.4, jitter=True, size=6, order=order)
        ax.set_title('NASA-TLX by Condition × Map (Reference)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Condition × Map', fontsize=11)
        ax.set_ylabel('NASA-TLX Total Score', fontsize=11)
        ax.set_xticklabels(['a (map0)', 'a (map1)', 'b (map0)', 'b (map1)'])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = "nasatlx_map_interaction_boxplot.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

    def plot_trajectories(self):
        """
        Create individual participant trajectory plots
        """
        print(f"\n  ✓ Creating trajectory plots...")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Get map used in condition a for each participant (for coloring)
        participant_map_a = {}
        for participant in self.df_long['participant'].unique():
            map_a = self.df_long[
                (self.df_long['participant'] == participant) &
                (self.df_long['condition'] == 'a')
            ]['map'].values[0]
            participant_map_a[participant] = map_a

        # Plot each participant's trajectory
        for participant in sorted(self.df_long['participant'].unique()):
            participant_data = self.df_long[self.df_long['participant'] == participant].copy()
            participant_data = participant_data.sort_values('condition')

            # Get values for condition a and b
            a_val = participant_data[participant_data['condition'] == 'a']['total_score'].values[0]
            b_val = participant_data[participant_data['condition'] == 'b']['total_score'].values[0]

            # Color by map used in condition a
            map_a = participant_map_a[participant]
            color = 'steelblue' if map_a == 0 else 'coral'

            # Plot line
            ax.plot([0, 1], [a_val, b_val], marker='o', color=color, alpha=0.6,
                   linewidth=2, markersize=8)

        # Create custom legend for map colors
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='steelblue', lw=2, marker='o', label='Used map0 in condition a'),
            Line2D([0], [0], color='coral', lw=2, marker='o', label='Used map1 in condition a')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Condition a', 'Condition b'], fontsize=12)
        ax.set_ylabel('NASA-TLX Total Score (lower is better)', fontsize=12)
        ax.set_title('Individual Participant Trajectories\nNASA-TLX Total Score', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Mid-point (50)')

        plt.tight_layout()
        filename = "nasatlx_trajectories.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

    def plot_violin(self):
        """
        Create violin plots for NASA-TLX Total Score
        """
        print(f"\n  ✓ Creating violin plots...")

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.violinplot(data=self.df_long, x='condition', y='total_score', ax=ax, palette='muted', inner='quartile')
        sns.stripplot(data=self.df_long, x='condition', y='total_score', ax=ax,
                     color='black', alpha=0.4, jitter=True, size=7)

        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('NASA-TLX Total Score (0-100, lower is better)', fontsize=12)
        ax.set_title('Distribution of NASA-TLX Total Score by Condition', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = "nasatlx_violin.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

    def create_all_visualizations(self):
        """
        Create all visualizations for NASA-TLX
        """
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)

        self.plot_boxplots()
        self.plot_trajectories()
        self.plot_violin()
        # Q-Q plots already created in test_normality_shapiro

        print(f"\n✓ All visualizations saved to: {self.viz_dir}")

    def save_results_to_excel(self):
        """
        Save all statistical results to Excel file
        """
        print("\n" + "=" * 80)
        print("SAVING RESULTS TO EXCEL")
        print("=" * 80)

        excel_path = self.output_dir / "statistical_results.xlsx"

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Metadata sheet
            metadata = {
                'Item': [
                    'Analysis Date',
                    'Data Directory',
                    'Number of Participants',
                    'Participants',
                    'Conditions',
                    'Dependent Variable',
                    'Score Range',
                    'Score Interpretation',
                    'Counterbalancing Status'
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    str(self.data_dir),
                    self.df_long['participant'].nunique(),
                    ', '.join(sorted(self.df_long['participant'].unique())),
                    ', '.join(sorted(self.df_long['condition'].unique())),
                    'NASA-TLX Total Score',
                    '0-100',
                    'Lower is better (less workload)',
                    'Valid' if len(self.counterbalancing_df[self.counterbalancing_df['valid'] == False]) == 0 else 'Invalid'
                ]
            }
            pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadata', index=False)

            # Counterbalancing sheet
            self.counterbalancing_df.to_excel(writer, sheet_name='Counterbalancing', index=False)

            # Descriptive statistics
            if 'descriptive' in self.results:
                desc_stats = self.results['descriptive']
                desc_stats['by_condition'].to_excel(writer, sheet_name='Descriptive_Condition', index=False)
                desc_stats['by_cond_map'].to_excel(writer, sheet_name='Descriptive_CondMap', index=False)

            # Normality testing
            if 'normality' in self.results:
                self.results['normality'].to_excel(writer, sheet_name='Normality_Test', index=False)

            # Paired comparison results
            if 'paired' in self.results:
                paired_res = self.results['paired']
                paired_df_data = {
                    'Metric': [],
                    'Value': []
                }
                for key, value in paired_res.items():
                    if key == 'effect_size':
                        paired_df_data['Metric'].append('effect_size_type')
                        paired_df_data['Value'].append(value['type'])
                        paired_df_data['Metric'].append('effect_size_value')
                        paired_df_data['Value'].append(value['value'])
                        paired_df_data['Metric'].append('effect_size_interpretation')
                        paired_df_data['Value'].append(value['interpretation'])
                    else:
                        paired_df_data['Metric'].append(key)
                        paired_df_data['Value'].append(str(value))

                pd.DataFrame(paired_df_data).to_excel(writer, sheet_name='Paired_Test', index=False)

        print(f"\n✓ Excel file saved: {excel_path}")

    def save_analysis_report_md(self):
        """
        Save analysis report in Markdown format
        """
        print("\n" + "=" * 80)
        print("GENERATING MARKDOWN REPORT")
        print("=" * 80)

        md_path = self.output_dir / "analysis_report.md"

        # Get library versions
        import pandas, numpy, scipy, matplotlib, seaborn, sys
        versions = {
            'Python': sys.version.split()[0],
            'pandas': pandas.__version__,
            'numpy': numpy.__version__,
            'scipy': scipy.__version__,
            'matplotlib': matplotlib.__version__,
            'seaborn': seaborn.__version__
        }

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# NASA-TLX 分析レポート\n\n")

            # Experiment overview
            f.write("## 実験概要\n\n")
            f.write(f"- **実験デザイン**: 2条件被験者内（condition: a, b）\n")
            f.write(f"- **参加者数**: {self.df_long['participant'].nunique()}名\n")
            f.write(f"- **評価指標**: NASA-TLX Total Score (0-100, 低い方が良い)\n")
            f.write(f"- **分析実行日**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Counterbalancing
            f.write("## カウンターバランシング\n\n")
            f.write("各参加者のcondition × map割り当て:\n\n")
            f.write("| 参加者 | Condition a | Condition b | 有効 |\n")
            f.write("|--------|-------------|-------------|------|\n")
            for _, row in self.counterbalancing_df.iterrows():
                valid_mark = "✓" if row['valid'] else "✗"
                f.write(f"| {row['participant']} | map{row['map_a']} | map{row['map_b']} | {valid_mark} |\n")
            f.write("\n")

            # Analysis flow
            f.write("## 分析フロー\n\n")
            f.write("### 1. データ前処理\n\n")
            f.write(f"- {len(self.df_raw)}個のCSVファイルからTotal Score抽出\n")
            f.write(f"- Long format DataFrame へ変換（{len(self.df_long)}行）\n")
            f.write(f"- カウンターバランシング検証: {'全参加者で有効' if all(self.counterbalancing_df['valid']) else '一部無効'}\n\n")

            # Descriptive statistics
            f.write("### 2. 記述統計\n\n")
            if 'descriptive' in self.results:
                desc = self.results['descriptive']['by_condition']
                f.write("Condition別:\n\n")
                f.write("| Condition | N | Mean | SD | Median | Min | Max |\n")
                f.write("|-----------|---|------|----|---------|----|-----|\n")
                for _, row in desc.iterrows():
                    f.write(f"| {row['condition']} | {row['count']:.0f} | {row['mean']:.2f} | {row['std']:.2f} | {row['median']:.2f} | {row['min']:.2f} | {row['max']:.2f} |\n")
                f.write("\n")

            # Normality testing
            f.write("### 3. 正規性検定（Shapiro-Wilk）\n\n")
            if 'normality' in self.results:
                norm_df = self.results['normality']
                f.write("| Condition | n | W | p値 | 正規性 |\n")
                f.write("|-----------|---|---|-----|--------|\n")
                for _, row in norm_df.iterrows():
                    f.write(f"| {row['condition']} | {row['n']} | {row['W']:.4f} | {row['p_value']:.4f} | {row['normal']} |\n")
                f.write("\n")

                # Decision
                all_normal = all(norm_df['normal'] == 'Yes')
                f.write(f"**採用した検定手法**: {'パラメトリック（対応のあるt検定）' if all_normal else 'ノンパラメトリック（Wilcoxon符号順位検定）'}\n\n")

            # Paired comparison
            f.write("### 4. 主要分析: Condition a vs b\n\n")
            if 'paired' in self.results:
                paired = self.results['paired']
                test_type = "対応のあるt検定" if paired['parametric'] else "Wilcoxon符号順位検定"
                f.write(f"**検定手法**: {test_type}\n\n")

                f.write(f"- **Condition a**: M = {paired['mean_a']:.2f}, SD = {paired['std_a']:.2f}, Median = {paired['median_a']:.2f}\n")
                f.write(f"- **Condition b**: M = {paired['mean_b']:.2f}, SD = {paired['std_b']:.2f}, Median = {paired['median_b']:.2f}\n")
                f.write(f"- **差（b - a）**: {paired['mean_b'] - paired['mean_a']:.2f}\n\n")

                if paired['parametric']:
                    f.write(f"**統計量**: t({paired['df']}) = {paired['t_statistic']:.4f}, p = {paired['p_value']:.4f}\n\n")
                else:
                    f.write(f"**統計量**: W = {paired['w_statistic']:.4f}, p = {paired['p_value']:.4f}\n\n")

                effect = paired['effect_size']
                f.write(f"**効果量** ({effect['type']}): {effect['value']:.3f} ({effect['interpretation']})\n\n")

                # Interpretation
                if paired['p_value'] < 0.05:
                    if paired['mean_b'] > paired['mean_a']:
                        workload_interp = "Condition b の方がワークロードが高い（スコアが高い）"
                    else:
                        workload_interp = "Condition a の方がワークロードが高い（スコアが高い）"
                    f.write(f"**結論**: 有意差あり（p < 0.05）。{workload_interp}\n\n")
                else:
                    f.write(f"**結論**: 有意差なし（p >= 0.05）\n\n")

            # Results summary
            f.write("## 結果サマリー\n\n")
            f.write("### NASA-TLX Total Score\n\n")
            if 'paired' in self.results:
                paired = self.results['paired']
                if paired['p_value'] < 0.05:
                    if paired['mean_b'] > paired['mean_a']:
                        finding = "Condition b の方がワークロードが有意に高い"
                    else:
                        finding = "Condition a の方がワークロードが有意に高い"
                    f.write(f"- **主要な知見**: {finding}（p = {paired['p_value']:.4f}）\n")
                else:
                    f.write(f"- **主要な知見**: Condition a と b の間にワークロードの有意差は認められなかった（p = {paired['p_value']:.4f}）\n")

                effect = paired['effect_size']
                f.write(f"- **効果量**: {effect['value']:.3f} ({effect['interpretation']})\n")
                f.write(f"- **解釈**: NASA-TLXスコアは低い方が良い（ワークロードが少ない）\n\n")

            # Statistical methods
            f.write("## 統計手法の選択理由\n\n")
            f.write("- 正規性検定（Shapiro-Wilk）の結果に基づき、適切な検定手法を選択\n")
            f.write("- 対応のあるデータのため、対応のあるt検定またはWilcoxon符号順位検定を使用\n")
            f.write("- 効果量としてHedges' g（パラメトリック）またはrank-biserial correlation（ノンパラメトリック）を報告\n\n")

            # Visualizations
            f.write("## 可視化\n\n")
            f.write("- 箱ひげ図: `visualizations/nasatlx_boxplots.png`\n")
            f.write("- 個人別軌跡: `visualizations/nasatlx_trajectories.png`\n")
            f.write("- Q-Qプロット: `visualizations/nasatlx_qq_plots.png`\n")
            f.write("- バイオリンプロット: `visualizations/nasatlx_violin.png`\n\n")

            # Detailed data
            f.write("## 詳細データ\n\n")
            f.write("詳細な統計結果は `statistical_results.xlsx` を参照してください。\n\n")

            # Used libraries
            f.write("## 使用したライブラリ\n\n")
            for lib, ver in versions.items():
                f.write(f"- {lib}: {ver}\n")
            f.write("\n")

            # Execution command
            f.write("## 実行コマンド\n\n")
            f.write("```bash\n")
            f.write("cd /Users/toshiki/Desktop/dev/midAir_display_image\n")
            f.write("source project/.venv/bin/activate\n")
            f.write("python project/src/analysis/nasatlx_main_experiment_analysis.py\n")
            f.write("```\n\n")

            f.write("---\n")
            f.write(f"Generated by nasatlx_main_experiment_analysis.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"\n✓ Markdown report saved: {md_path}")

    def run_full_analysis(self):
        """
        Run complete analysis pipeline
        """
        print("\n" + "=" * 80)
        print("NASA-TLX STATISTICAL ANALYSIS")
        print("2-Condition Within-Subject Design")
        print("=" * 80)

        print("\nExperiment Metadata:")
        print(f"  - Data directory: {self.data_dir}")
        print(f"  - Output directory: {self.output_dir}")
        print(f"  - Dependent variable: NASA-TLX Total Score (0-100, lower is better)")
        print(f"  - Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Load data
        self.build_dataframe()

        # Step 2: Validate counterbalancing
        self.validate_counterbalancing()

        # Step 3: Descriptive statistics
        desc_stats = self.compute_descriptive_stats()
        self.results['descriptive'] = desc_stats

        # Step 4: Normality testing
        normality_df = self.test_normality_shapiro()
        self.results['normality'] = normality_df

        # Create Q-Q plots
        self.create_qq_plots()

        # Step 5: Paired comparison (condition a vs b)
        parametric = self.normality_results['all_normal']
        paired_result = self.run_paired_comparison(parametric=parametric)
        self.results['paired'] = paired_result

        # Create all visualizations
        self.create_all_visualizations()

        # Save results to Excel
        self.save_results_to_excel()

        # Generate Markdown report
        self.save_analysis_report_md()

        print("\n\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print("\nAnalysis summary:")
        print(f"  - Data directory: {self.data_dir}")
        print(f"  - Output directory: {self.output_dir}")
        print(f"  - Visualizations directory: {self.viz_dir}")
        print(f"\n✓ All analysis complete!")
        print("\nGenerated outputs:")
        print("  ✓ Statistical results (Excel): statistical_results.xlsx")
        print("  ✓ Analysis report (Markdown): analysis_report.md")
        print("  ✓ Box plots (condition comparison + reference)")
        print("  ✓ Individual trajectories (participant-level changes)")
        print("  ✓ Q-Q plots for normality assessment")
        print("  ✓ Violin plots (distribution visualization)")
        print(f"\nTotal visualizations: 4 files")
        print(f"\n{'=' * 80}")
        print("You can now review the results in:")
        print(f"  - {self.output_dir}")
        print("=" * 80)


def main():
    """Main execution function"""
    # Path to NASA-TLX data directory
    data_dir = Path(__file__).parent.parent.parent / "output" / "nasa-tlx"

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    # Create analyzer and run analysis
    analyzer = NASATLXAnalyzer(data_dir)
    analyzer.run_full_analysis()

    return 0


if __name__ == "__main__":
    sys.exit(main())
