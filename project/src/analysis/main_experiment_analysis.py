#!/usr/bin/env python3
"""
Main Experiment Statistical Analysis

2×2 within-subject HCI experiment analysis:
- Experimental Design: condition (a, b) × map (0, 1)
- N = 13 participants (within-subject)
- DVs: total_duration_sec, face_gaze_duration_sec
- Analysis: 2-way RM-ANOVA + paired t-test/Wilcoxon test

Author: Generated for midAir_display_image project
Date: 2026-01-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
from pathlib import Path
import warnings
import re
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class MainExperimentAnalyzer:
    """
    Statistical analysis for 2×2 within-subject HCI experiment

    Experimental Design:
    - Within-subject factors: condition (a, b) × map (0, 1)
    - Counterbalanced: each participant uses different map for each condition
    - N = 13 participants
    - DVs: total_duration_sec, face_gaze_duration_sec
    """

    def __init__(self, data_dir):
        """
        Initialize analyzer

        Args:
            data_dir: Path to directory containing summary CSV files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "analysis_results"
        self.viz_dir = self.output_dir / "visualizations"

        # DataFrames
        self.df_raw = None          # Raw parsed data from CSVs
        self.df_long = None         # Long format for analysis

        # Results storage
        self.results = {}
        self.normality_results = {}

        # DVs to analyze
        self.dvs = ['total_duration_sec', 'face_gaze_duration_sec']

    def parse_summary_csv(self, filepath):
        """
        Parse summary CSV file in vertical format (metric, value)

        Extracts from filename: {timestamp}_summary_{name}_{condition}_map{N}.csv
        Example: 20251223_140933_summary_nakayama_a_map0.csv

        Args:
            filepath: Path to summary CSV file

        Returns:
            dict: Parsed data with participant, condition, map, and DV values
        """
        # Parse filename using regex
        # Expected format: {YYYYMMDD_HHMMSS}_summary_{name}_{condition}_map{N}
        filename = filepath.stem
        pattern = r'(\d{8}_\d{6})_summary_(\w+)_([ab])_map([01])'
        match = re.match(pattern, filename)

        if not match:
            raise ValueError(f"Filename does not match expected pattern: {filename}")

        timestamp, participant, condition, map_str = match.groups()
        map_num = int(map_str)

        # Parse CSV (vertical format: metric, value)
        df = pd.read_csv(filepath)
        metrics = dict(zip(df['metric'], df['value']))

        return {
            'participant': participant,
            'condition': condition,
            'map': map_num,
            'total_duration_sec': float(metrics['total_duration_sec']),
            'face_gaze_duration_sec': float(metrics['face_gaze_duration_sec']),
            'session_id': metrics['session_id'],
            'timestamp': timestamp
        }

    def build_dataframe(self):
        """
        Build long-format DataFrame from all summary CSV files

        Returns:
            pd.DataFrame: Long format with columns: participant, condition, map, DVs
        """
        print("=" * 80)
        print("STEP 1: LOADING DATA")
        print("=" * 80)

        # Find all summary files
        summary_files = sorted(self.data_dir.glob('*_summary_*.csv'))

        if len(summary_files) == 0:
            raise FileNotFoundError(f"No summary CSV files found in {self.data_dir}")

        print(f"\nFound {len(summary_files)} summary CSV files")

        # Parse each file
        data_list = []
        failed_files = []

        for filepath in summary_files:
            try:
                row_data = self.parse_summary_csv(filepath)
                data_list.append(row_data)
            except Exception as e:
                print(f"Warning: Failed to parse {filepath.name}: {e}")
                failed_files.append(filepath.name)
                continue

        # Build DataFrame
        self.df_raw = pd.DataFrame(data_list)

        # Create long format
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
        Validate counterbalancing: each participant should use different maps for conditions a and b

        Returns:
            bool: True if counterbalancing is valid for all participants
        """
        print("\n" + "=" * 80)
        print("STEP 2: VALIDATING COUNTERBALANCING")
        print("=" * 80)

        print("\nCounterbalancing Check:")
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

    def compute_descriptive_stats(self, dv):
        """
        Compute descriptive statistics for a dependent variable

        Args:
            dv: Dependent variable name ('total_duration_sec' or 'face_gaze_duration_sec')

        Returns:
            dict: Descriptive statistics tables
        """
        print("\n" + "=" * 80)
        print(f"STEP 3: DESCRIPTIVE STATISTICS - {dv}")
        print("=" * 80)

        # By condition × map
        print("\nBy Condition × Map:")
        desc_cond_map = self.df_long.groupby(['condition', 'map'])[dv].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('median', 'median'),
            ('q1', lambda x: x.quantile(0.25)),
            ('q3', lambda x: x.quantile(0.75)),
            ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25)),
            ('min', 'min'),
            ('max', 'max')
        ]).reset_index()

        print(desc_cond_map.to_string(index=False))

        # By condition (collapsed across map)
        print("\nBy Condition (collapsed across map):")
        df_collapsed_cond = self.df_long.groupby(['participant', 'condition'])[dv].mean().reset_index()
        desc_cond = df_collapsed_cond.groupby('condition')[dv].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('sem', lambda x: x.std() / np.sqrt(len(x))),
            ('median', 'median'),
            ('ci95_lower', lambda x: x.mean() - 1.96 * x.std() / np.sqrt(len(x))),
            ('ci95_upper', lambda x: x.mean() + 1.96 * x.std() / np.sqrt(len(x)))
        ]).reset_index()

        print(desc_cond.to_string(index=False))

        # By map (collapsed across condition)
        print("\nBy Map (collapsed across condition):")
        df_collapsed_map = self.df_long.groupby(['participant', 'map'])[dv].mean().reset_index()
        desc_map = df_collapsed_map.groupby('map')[dv].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('sem', lambda x: x.std() / np.sqrt(len(x))),
            ('median', 'median')
        ]).reset_index()

        print(desc_map.to_string(index=False))

        return {
            'by_cond_map': desc_cond_map,
            'by_condition': desc_cond,
            'by_map': desc_map
        }

    def test_normality_shapiro(self, dv):
        """
        Test normality using Shapiro-Wilk test for each condition × map cell

        Args:
            dv: Dependent variable name

        Returns:
            pd.DataFrame: Normality test results
        """
        print("\n" + "=" * 80)
        print(f"STEP 4: NORMALITY TESTING (Shapiro-Wilk) - {dv}")
        print("=" * 80)
        print("\nH0: Data is normally distributed")
        print("If p < 0.05, reject H0 (data is NOT normally distributed)")

        normality_results = []

        for condition in ['a', 'b']:
            for map_val in [0, 1]:
                data = self.df_long[
                    (self.df_long['condition'] == condition) &
                    (self.df_long['map'] == map_val)
                ][dv]

                if len(data) < 3:
                    print(f"\nCondition {condition}, Map {map_val}:")
                    print(f"  WARNING: Insufficient data (n={len(data)}), skipping test")
                    continue

                statistic, p_value = stats.shapiro(data)
                normal = p_value > 0.05

                result = {
                    'condition': condition,
                    'map': map_val,
                    'n': len(data),
                    'W': statistic,
                    'p_value': p_value,
                    'normal': 'Yes' if normal else 'No'
                }
                normality_results.append(result)

                print(f"\nCondition {condition}, Map {map_val}:")
                print(f"  n = {len(data)}")
                print(f"  W = {statistic:.4f}, p = {p_value:.4f} {'(Normal)' if normal else '(NOT Normal)'}")

        df_normality = pd.DataFrame(normality_results)

        print("\n" + "-" * 80)
        print("Summary:")
        print(df_normality.to_string(index=False))

        all_normal = all(df_normality['normal'] == 'Yes')

        print("\n" + "-" * 80)
        if all_normal:
            print("DECISION: All cells pass normality → Using PARAMETRIC tests")
            print("  ✓ Paired t-test for condition comparison")
            print("  ✓ Hedges' g for effect size")
        else:
            print("DECISION: Some cells violate normality → Using NON-PARAMETRIC tests")
            print("  → Wilcoxon signed-rank test for condition comparison")
            print("  → Rank-biserial correlation for effect size")

        self.normality_results[dv] = {
            'df': df_normality,
            'all_normal': all_normal
        }

        return df_normality

    def create_qq_plots(self, dv):
        """
        Create Q-Q plots for normality assessment

        Args:
            dv: Dependent variable name
        """
        print(f"\n  ✓ Creating Q-Q plots for {dv}...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        cells = [('a', 0), ('a', 1), ('b', 0), ('b', 1)]

        for idx, (condition, map_val) in enumerate(cells):
            data = self.df_long[
                (self.df_long['condition'] == condition) &
                (self.df_long['map'] == map_val)
            ][dv]

            stats.probplot(data, dist="norm", plot=axes[idx])
            axes[idx].set_title(f'Condition {condition}, Map {map_val}', fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"{dv}_qq_plots.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

    def run_2way_rmanova(self, dv):
        """
        Run 2-way repeated measures ANOVA (condition × map)

        Args:
            dv: Dependent variable name

        Returns:
            pd.DataFrame: ANOVA results
        """
        print("\n" + "=" * 80)
        print(f"STEP 5: 2-WAY REPEATED MEASURES ANOVA - {dv}")
        print("=" * 80)
        print("\nWithin-subject factors: condition (a, b) × map (0, 1)")

        # Run 2-way RM-ANOVA using pingouin
        anova_result = pg.rm_anova(
            data=self.df_long,
            dv=dv,
            within=['condition', 'map'],
            subject='participant',
            detailed=True
        )

        print("\nANOVA Results:")
        print(anova_result.to_string(index=False))

        # Extract key statistics
        print("\n" + "-" * 80)
        print("INTERPRETATION:")

        for source in ['condition', 'map', 'condition * map']:
            row = anova_result[anova_result['Source'] == source]
            if len(row) == 0:
                continue

            f_val = row['F'].values[0]
            p_val = row['p-unc'].values[0]
            df1 = row['DF1'].values[0] if 'DF1' in row.columns else row['ddof1'].values[0]
            df2 = row['DF2'].values[0] if 'DF2' in row.columns else row['ddof2'].values[0]

            print(f"\n{source}:")
            print(f"  F({df1:.0f}, {df2:.0f}) = {f_val:.4f}, p = {p_val:.4f}")

            if p_val < 0.001:
                print("  *** Highly significant (p < 0.001)")
            elif p_val < 0.01:
                print("  **  Very significant (p < 0.01)")
            elif p_val < 0.05:
                print("  *   Significant (p < 0.05)")
            else:
                print("      Not significant (p >= 0.05)")

            # Interpretation for each effect
            if source == 'condition':
                if p_val < 0.05:
                    print("  → Condition (a vs b) has a significant effect")
                else:
                    print("  → Condition (a vs b) does not have a significant effect")
            elif source == 'map':
                if p_val < 0.05:
                    print("  → WARNING: Map has a significant effect (counterbalancing issue)")
                else:
                    print("  → Good: Map does not have a significant effect (counterbalancing OK)")
            elif source == 'condition * map':
                if p_val < 0.05:
                    print("  → Interaction exists: condition effect depends on map")
                    print("  → Post-hoc simple effects analysis recommended")
                else:
                    print("  → No interaction: condition effect is consistent across maps")

        return anova_result

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

            # Approximate CI for effect size (simplified)
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
            from scipy.stats import wilcoxon
            _, p = wilcoxon(data1, data2)

            # Calculate rank-biserial correlation
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

    def run_paired_comparison(self, dv, parametric=True):
        """
        Run paired comparison between condition a and b

        Args:
            dv: Dependent variable name
            parametric: If True, use paired t-test; otherwise Wilcoxon

        Returns:
            dict: Test results
        """
        print("\n" + "=" * 80)
        print(f"STEP 6: PAIRED COMPARISON (Condition a vs b) - {dv}")
        print("=" * 80)

        # Collapse across map to get one value per participant per condition
        df_collapsed = self.df_long.groupby(['participant', 'condition'])[dv].mean().reset_index()

        # Pivot to wide format for comparison
        df_wide = df_collapsed.pivot(index='participant', columns='condition', values=dv)

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
            direction = "Condition b > Condition a"
        else:
            direction = "Condition a > Condition b"
        print(f"  {direction}")

        return results

    def plot_boxplots(self, dv):
        """
        Create box plots for the dependent variable

        Args:
            dv: Dependent variable name
        """
        print(f"\n  ✓ Creating box plots for {dv}...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Condition comparison (collapsed across map)
        df_collapsed = self.df_long.groupby(['participant', 'condition'])[dv].mean().reset_index()

        sns.boxplot(data=df_collapsed, x='condition', y=dv, ax=axes[0], palette='Set2')
        sns.stripplot(data=df_collapsed, x='condition', y=dv, ax=axes[0],
                     color='black', alpha=0.5, jitter=True, size=8)
        axes[0].set_title(f'Condition Comparison\n{dv}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Condition', fontsize=11)
        axes[0].set_ylabel(dv.replace('_', ' ').title(), fontsize=11)
        axes[0].grid(True, alpha=0.3, axis='y')

        # Right: 2×2 design (condition × map)
        df_temp = self.df_long.copy()
        df_temp['cond_map'] = df_temp['condition'] + '_map' + df_temp['map'].astype(str)
        order = ['a_map0', 'a_map1', 'b_map0', 'b_map1']

        sns.boxplot(data=df_temp, x='cond_map', y=dv, ax=axes[1], palette='Set3', order=order)
        sns.stripplot(data=df_temp, x='cond_map', y=dv, ax=axes[1],
                     color='black', alpha=0.4, jitter=True, size=6, order=order)
        axes[1].set_title(f'Condition × Map Interaction\n{dv}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Condition × Map', fontsize=11)
        axes[1].set_ylabel(dv.replace('_', ' ').title(), fontsize=11)
        axes[1].set_xticklabels(['a (map0)', 'a (map1)', 'b (map0)', 'b (map1)'])
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = f"{dv}_boxplots.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

    def plot_trajectories(self, dv):
        """
        Create individual participant trajectory plots

        Args:
            dv: Dependent variable name
        """
        print(f"\n  ✓ Creating trajectory plots for {dv}...")

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
            a_val = participant_data[participant_data['condition'] == 'a'][dv].values[0]
            b_val = participant_data[participant_data['condition'] == 'b'][dv].values[0]

            # Color by map used in condition a
            map_a = participant_map_a[participant]
            color = 'steelblue' if map_a == 0 else 'coral'

            # Plot line
            ax.plot([0, 1], [a_val, b_val], marker='o', color=color, alpha=0.6,
                   linewidth=2, markersize=8, label=participant if participant == sorted(self.df_long['participant'].unique())[0] else '')

        # Create custom legend for map colors
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='steelblue', lw=2, marker='o', label='Used map0 in condition a'),
            Line2D([0], [0], color='coral', lw=2, marker='o', label='Used map1 in condition a')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Condition a', 'Condition b'], fontsize=12)
        ax.set_ylabel(dv.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Individual Participant Trajectories\n{dv}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"{dv}_trajectories.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

    def plot_interaction(self, dv):
        """
        Create interaction plot (condition × map)

        Args:
            dv: Dependent variable name
        """
        print(f"\n  ✓ Creating interaction plot for {dv}...")

        # Compute mean and SEM for each condition × map combination
        summary = self.df_long.groupby(['condition', 'map'])[dv].agg(['mean', 'sem']).reset_index()

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot lines for each map
        for map_val in [0, 1]:
            data_map = summary[summary['map'] == map_val].sort_values('condition')
            conditions = data_map['condition'].values
            means = data_map['mean'].values
            sems = data_map['sem'].values

            linestyle = '-' if map_val == 0 else '--'
            marker = 'o' if map_val == 0 else 's'
            color = 'steelblue' if map_val == 0 else 'coral'

            ax.plot(conditions, means, linestyle=linestyle, marker=marker, color=color,
                   linewidth=2.5, markersize=10, label=f'Map {map_val}')
            ax.fill_between(conditions, means - sems, means + sems, alpha=0.2, color=color)

        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel(f'{dv.replace("_", " ").title()} (Mean ± SEM)', fontsize=12)
        ax.set_title(f'Condition × Map Interaction\n{dv}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"{dv}_interaction.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

    def plot_violin(self, dv):
        """
        Create violin plots for the dependent variable

        Args:
            dv: Dependent variable name
        """
        print(f"\n  ✓ Creating violin plots for {dv}...")

        fig, ax = plt.subplots(figsize=(10, 7))

        # Collapse across map
        df_collapsed = self.df_long.groupby(['participant', 'condition'])[dv].mean().reset_index()

        sns.violinplot(data=df_collapsed, x='condition', y=dv, ax=ax, palette='muted', inner='quartile')
        sns.stripplot(data=df_collapsed, x='condition', y=dv, ax=ax,
                     color='black', alpha=0.4, jitter=True, size=7)

        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel(dv.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Distribution by Condition\n{dv}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = f"{dv}_violin.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

    def create_all_visualizations(self, dv):
        """
        Create all visualizations for a dependent variable

        Args:
            dv: Dependent variable name
        """
        print("\n" + "=" * 80)
        print(f"CREATING VISUALIZATIONS - {dv}")
        print("=" * 80)

        self.plot_boxplots(dv)
        self.plot_trajectories(dv)
        self.plot_interaction(dv)
        self.plot_violin(dv)

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
                    'Maps',
                    'Dependent Variables',
                    'Counterbalancing Status'
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    str(self.data_dir),
                    self.df_long['participant'].nunique(),
                    ', '.join(sorted(self.df_long['participant'].unique())),
                    ', '.join(sorted(self.df_long['condition'].unique())),
                    ', '.join([str(x) for x in sorted(self.df_long['map'].unique())]),
                    ', '.join(self.dvs),
                    'Valid' if len(self.counterbalancing_df[self.counterbalancing_df['valid'] == False]) == 0 else 'Invalid'
                ]
            }
            pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadata', index=False)

            # Counterbalancing sheet
            self.counterbalancing_df.to_excel(writer, sheet_name='Counterbalancing', index=False)

            # Results for each DV
            for dv in self.dvs:
                dv_short = dv.split('_')[0].capitalize()[:10]  # Shorten for sheet name

                # Descriptive statistics
                if f'{dv}_descriptive' in self.results:
                    desc_stats = self.results[f'{dv}_descriptive']
                    desc_stats['by_cond_map'].to_excel(
                        writer, sheet_name=f'{dv_short}_Desc_CondMap', index=False)
                    desc_stats['by_condition'].to_excel(
                        writer, sheet_name=f'{dv_short}_Desc_Cond', index=False)
                    desc_stats['by_map'].to_excel(
                        writer, sheet_name=f'{dv_short}_Desc_Map', index=False)

                # Normality testing
                if f'{dv}_normality' in self.results:
                    self.results[f'{dv}_normality'].to_excel(
                        writer, sheet_name=f'{dv_short}_Normality', index=False)

                # ANOVA results
                if f'{dv}_anova' in self.results:
                    self.results[f'{dv}_anova'].to_excel(
                        writer, sheet_name=f'{dv_short}_ANOVA', index=False)

                # Paired comparison results
                if f'{dv}_paired' in self.results:
                    paired_res = self.results[f'{dv}_paired']
                    # Convert to DataFrame
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

                    pd.DataFrame(paired_df_data).to_excel(
                        writer, sheet_name=f'{dv_short}_PairedTest', index=False)

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
        import pandas, numpy, scipy, pingouin, matplotlib, seaborn, sys
        versions = {
            'Python': sys.version.split()[0],
            'pandas': pandas.__version__,
            'numpy': numpy.__version__,
            'scipy': scipy.__version__,
            'pingouin': pingouin.__version__,
            'matplotlib': matplotlib.__version__,
            'seaborn': seaborn.__version__
        }

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 実験分析レポート\n\n")

            # Experiment overview
            f.write("## 実験概要\n\n")
            f.write(f"- **実験デザイン**: 2×2被験者内要因（condition: a, b × map: 0, 1）\n")
            f.write(f"- **参加者数**: {self.df_long['participant'].nunique()}名\n")
            f.write(f"- **従属変数**: {', '.join(self.dvs)}\n")
            f.write(f"- **データ収集日**: {self.df_long['timestamp'].min()} 〜 {self.df_long['timestamp'].max()}\n")
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
            f.write(f"- {len(self.df_raw)}個のCSVファイル読み込み\n")
            f.write(f"- Long format DataFrame へ変換（{len(self.df_long)}行）\n")
            f.write(f"- カウンターバランシング検証: {'全参加者で有効' if all(self.counterbalancing_df['valid']) else '一部無効'}\n\n")

            # Results for each DV
            for dv in self.dvs:
                f.write(f"## 分析結果: {dv}\n\n")

                # Descriptive statistics
                f.write("### 2. 記述統計\n\n")
                if f'{dv}_descriptive' in self.results:
                    desc = self.results[f'{dv}_descriptive']['by_condition']
                    f.write("Condition別（mapをcollapse）:\n\n")
                    f.write("| Condition | N | Mean | SD | Median |\n")
                    f.write("|-----------|---|------|----|---------|\n")
                    for _, row in desc.iterrows():
                        f.write(f"| {row['condition']} | {row['count']:.0f} | {row['mean']:.3f} | {row['std']:.3f} | {row['median']:.3f} |\n")
                    f.write("\n")

                # Normality testing
                f.write("### 3. 正規性検定（Shapiro-Wilk）\n\n")
                if f'{dv}_normality' in self.results:
                    norm_df = self.results[f'{dv}_normality']
                    f.write("| Condition | Map | n | W | p値 | 正規性 |\n")
                    f.write("|-----------|-----|---|---|-----|--------|\n")
                    for _, row in norm_df.iterrows():
                        f.write(f"| {row['condition']} | {row['map']} | {row['n']} | {row['W']:.4f} | {row['p_value']:.4f} | {row['normal']} |\n")
                    f.write("\n")

                    # Decision
                    all_normal = all(norm_df['normal'] == 'Yes')
                    f.write(f"**採用した検定手法**: {'パラメトリック（対応のあるt検定）' if all_normal else 'ノンパラメトリック（Wilcoxon符号順位検定）'}\n\n")

                # 2-way ANOVA results
                f.write("### 4. 2要因反復測定ANOVA\n\n")
                if f'{dv}_anova' in self.results:
                    anova_df = self.results[f'{dv}_anova']
                    f.write("| Source | F | p値 | 解釈 |\n")
                    f.write("|--------|---|-----|------|\n")
                    for source_name in ['condition', 'map', 'condition * map']:
                        row = anova_df[anova_df['Source'] == source_name]
                        if len(row) > 0:
                            f_val = row['F'].values[0]
                            p_val = row['p-unc'].values[0]

                            # Format F value
                            f_val_str = f"{f_val:.3f}" if not pd.isna(f_val) else "N/A"

                            # Format p value
                            p_val_str = f"{p_val:.4f}" if not pd.isna(p_val) else "N/A"

                            # Significance
                            if pd.isna(p_val):
                                sig = "N/A"
                            elif p_val < 0.001:
                                sig = "***"
                            elif p_val < 0.01:
                                sig = "**"
                            elif p_val < 0.05:
                                sig = "*"
                            else:
                                sig = "n.s."

                            f.write(f"| {source_name} | {f_val_str} | {p_val_str} | {sig} |\n")
                    f.write("\n")

                # Paired comparison
                f.write("### 5. 主要分析: Condition a vs b\n\n")
                if f'{dv}_paired' in self.results:
                    paired = self.results[f'{dv}_paired']
                    test_type = "対応のあるt検定" if paired['parametric'] else "Wilcoxon符号順位検定"
                    f.write(f"**検定手法**: {test_type}\n\n")

                    f.write(f"- **Condition a**: M = {paired['mean_a']:.3f}, SD = {paired['std_a']:.3f}, Median = {paired['median_a']:.3f}\n")
                    f.write(f"- **Condition b**: M = {paired['mean_b']:.3f}, SD = {paired['std_b']:.3f}, Median = {paired['median_b']:.3f}\n")
                    f.write(f"- **差（b - a）**: {paired['mean_b'] - paired['mean_a']:.3f}\n\n")

                    if paired['parametric']:
                        f.write(f"**統計量**: t({paired['df']}) = {paired['t_statistic']:.4f}, p = {paired['p_value']:.4f}\n\n")
                    else:
                        f.write(f"**統計量**: W = {paired['w_statistic']:.4f}, p = {paired['p_value']:.4f}\n\n")

                    effect = paired['effect_size']
                    f.write(f"**効果量** ({effect['type']}): {effect['value']:.3f} ({effect['interpretation']})\n\n")

                    # Interpretation
                    if paired['p_value'] < 0.05:
                        direction = "Condition b > Condition a" if paired['mean_b'] > paired['mean_a'] else "Condition a > Condition b"
                        f.write(f"**結論**: 有意差あり（p < 0.05）。{direction}\n\n")
                    else:
                        f.write(f"**結論**: 有意差なし（p >= 0.05）\n\n")

            # Results summary
            f.write("## 結果サマリー\n\n")
            for dv in self.dvs:
                f.write(f"### {dv}\n\n")
                if f'{dv}_paired' in self.results:
                    paired = self.results[f'{dv}_paired']
                    if paired['p_value'] < 0.05:
                        direction = "Condition b の方が有意に大きい" if paired['mean_b'] > paired['mean_a'] else "Condition a の方が有意に大きい"
                        f.write(f"- **主要な知見**: {direction}（p = {paired['p_value']:.4f}）\n")
                    else:
                        f.write(f"- **主要な知見**: Condition a と b の間に有意差は認められなかった（p = {paired['p_value']:.4f}）\n")

                    effect = paired['effect_size']
                    f.write(f"- **効果量**: {effect['value']:.3f} ({effect['interpretation']})\n")

                # Counterbalancing check from ANOVA
                if f'{dv}_anova' in self.results:
                    anova_df = self.results[f'{dv}_anova']
                    map_row = anova_df[anova_df['Source'] == 'map']
                    if len(map_row) > 0:
                        p_map = map_row['p-unc'].values[0]
                        if pd.isna(p_map) or p_map >= 0.05:
                            f.write(f"- **カウンターバランシング**: Map効果は有意でない（良好）\n")
                        else:
                            f.write(f"- **カウンターバランシング**: WARNING - Map効果が有意（p = {p_map:.4f}）\n")
                f.write("\n")

            # Statistical methods
            f.write("## 統計手法の選択理由\n\n")
            f.write("- 正規性検定（Shapiro-Wilk）の結果に基づき、適切な検定手法を選択\n")
            f.write("- 対応のあるデータのため、対応のあるt検定またはWilcoxon符号順位検定を使用\n")
            f.write("- 効果量としてHedges' g（パラメトリック）またはrank-biserial correlation（ノンパラメトリック）を報告\n\n")

            # Visualizations
            f.write("## 可視化\n\n")
            for dv in self.dvs:
                f.write(f"### {dv}\n\n")
                f.write(f"- 箱ひげ図: `visualizations/{dv}_boxplots.png`\n")
                f.write(f"- 個人別軌跡: `visualizations/{dv}_trajectories.png`\n")
                f.write(f"- 交互作用プロット: `visualizations/{dv}_interaction.png`\n")
                f.write(f"- Q-Qプロット: `visualizations/{dv}_qq_plots.png`\n")
                f.write(f"- バイオリンプロット: `visualizations/{dv}_violin.png`\n\n")

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
            f.write("python project/src/analysis/main_experiment_analysis.py\n")
            f.write("```\n\n")

            f.write("---\n")
            f.write(f"Generated by main_experiment_analysis.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"\n✓ Markdown report saved: {md_path}")

    def run_full_analysis(self):
        """
        Run complete analysis pipeline
        """
        print("\n" + "=" * 80)
        print("MAIN EXPERIMENT STATISTICAL ANALYSIS")
        print("HCI Within-Subject Design: Condition (a, b) × Map (0, 1)")
        print("=" * 80)

        print("\nExperiment Metadata:")
        print(f"  - Data directory: {self.data_dir}")
        print(f"  - Output directory: {self.output_dir}")
        print(f"  - Dependent variables: {', '.join(self.dvs)}")
        print(f"  - Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Load data
        self.build_dataframe()

        # Step 2: Validate counterbalancing
        self.validate_counterbalancing()

        # Analyze each DV
        for dv in self.dvs:
            print(f"\n\n{'=' * 80}")
            print(f"ANALYSIS FOR: {dv}")
            print("=" * 80)

            # Step 3: Descriptive statistics
            desc_stats = self.compute_descriptive_stats(dv)
            self.results[f'{dv}_descriptive'] = desc_stats

            # Step 4: Normality testing
            normality_df = self.test_normality_shapiro(dv)
            self.results[f'{dv}_normality'] = normality_df

            # Create Q-Q plots
            self.create_qq_plots(dv)

            # Step 5: 2-way RM-ANOVA
            anova_result = self.run_2way_rmanova(dv)
            self.results[f'{dv}_anova'] = anova_result

            # Step 6: Paired comparison (condition a vs b)
            parametric = self.normality_results[dv]['all_normal']
            paired_result = self.run_paired_comparison(dv, parametric=parametric)
            self.results[f'{dv}_paired'] = paired_result

            # Step 7: Post-hoc simple effects (if interaction significant)
            # Check if interaction is significant
            anova_df = anova_result
            interaction_row = anova_df[anova_df['Source'] == 'condition * map']
            if len(interaction_row) > 0:
                p_interaction = interaction_row['p-unc'].values[0]
                if p_interaction < 0.05:
                    print("\n" + "=" * 80)
                    print(f"STEP 7: POST-HOC SIMPLE EFFECTS - {dv}")
                    print("=" * 80)
                    print(f"\nInteraction is significant (p = {p_interaction:.4f})")
                    print("Performing simple effects analysis...")
                    # TODO: Implement run_posthoc_simple_effects
                    print("(Post-hoc analysis to be implemented)")
                else:
                    print("\n" + "=" * 80)
                    print("STEP 7: POST-HOC SIMPLE EFFECTS - SKIPPED")
                    print("=" * 80)
                    print(f"Interaction not significant (p = {p_interaction:.4f})")

            # Create all visualizations
            self.create_all_visualizations(dv)

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
        print("  ✓ Q-Q plots for normality assessment")
        print("  ✓ Box plots (condition comparison + interaction)")
        print("  ✓ Individual trajectories (participant-level changes)")
        print("  ✓ Interaction plots (condition × map)")
        print("  ✓ Violin plots (distribution visualization)")
        print(f"\nTotal visualizations: {len(self.dvs) * 5} files")
        print(f"\n{'=' * 80}")
        print("You can now review the results in:")
        print(f"  - {self.output_dir}")
        print("=" * 80)


def main():
    """Main execution function"""
    # Path to data directory
    data_dir = Path(__file__).parent.parent.parent / "output" / "csv" / "main"

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    # Create analyzer and run analysis
    analyzer = MainExperimentAnalyzer(data_dir)
    analyzer.run_full_analysis()

    return 0


if __name__ == "__main__":
    sys.exit(main())
