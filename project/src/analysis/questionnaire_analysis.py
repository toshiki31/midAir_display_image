#!/usr/bin/env python3
"""
Post-Experiment Questionnaire Analysis

Analyzes main experiment questionnaire data with:
- 6 Likert scale questions (7-point scale): Task/translation evaluation
- 10 SUS questions (System Usability Scale): Usability assessment

Experimental design: 2 conditions within-subject (A vs B)
Participants: 13
Statistical tests: Paired t-test or Wilcoxon signed-rank test based on normality

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
from datetime import datetime
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment, PatternFill

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Support Japanese fonts
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic']
plt.rcParams['axes.unicode_minus'] = False


def add_significance_brackets(ax, p_value, x1, x2, y_max, height_increment=None):
    """
    Add significance bracket and asterisk to plot

    Parameters:
    - ax: matplotlib axis object
    - p_value: p-value from statistical test
    - x1, x2: x-axis positions for bracket ends
    - y_max: maximum y value for positioning bracket
    - height_increment: spacing above data (auto if None)
    """
    if p_value >= 0.05:
        return

    # Auto-calculate height increment if not provided
    if height_increment is None:
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        height_increment = y_range * 0.08

    # Determine significance level
    if p_value < 0.001:
        sig_symbol = '***'
    elif p_value < 0.01:
        sig_symbol = '**'
    elif p_value < 0.05:
        sig_symbol = '*'
    else:
        return

    # Calculate bracket height
    bracket_height = y_max + height_increment

    # Draw bracket
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


class QuestionnaireAnalyzer:
    """Post-experiment questionnaire analysis (Likert + SUS)"""

    def __init__(self, csv_path):
        """
        Initialize analyzer with path to questionnaire CSV

        Args:
            csv_path: Path to main_expt_questionnaire.csv
        """
        self.csv_path = Path(csv_path)
        self.output_dir = self.csv_path.parent / "main_expt_questionnaire_results"
        self.viz_dir = self.output_dir / "visualizations"

        self.df_raw = None
        self.df_clean = None
        self.df_likert_long = None
        self.df_sus_scores = None

        self.results = {}

        # Likert scale items (7-point scale)
        self.likert_items = {
            'タスクはやりやすかった': 'task_ease',
            'タスクは理解できた': 'task_understanding',
            '相手の顔は見やすかった': 'face_visibility',
            '翻訳は正確だった': 'translation_accuracy',
            '翻訳は読みやすかった': 'translation_readability',
            '翻訳は不自然だった': 'translation_unnaturalness'
        }

        # English to Japanese (for visualization)
        self.likert_items_jp = {v: k for k, v in self.likert_items.items()}

        # SUS items (5-point scale)
        self.sus_items = [
            '1. このシステムを頻繁に使用したい',
            '2. このシステムは必要以上に複雑だと思う',
            '3. このシステムはシンプルで使いやすい',
            '4. このシステムを使うにはテクニカルサポートが必要',
            '5. このシステムはスムーズに機能し、連携がとれていると思う',
            '6. システムには不規則な点が多いと思う',
            '7. ほとんどの人がこのシステムをすぐに習得できると思う',
            '8. このシステムは時間がかかると思う',
            '9. このシステムを使っていると、自信が持てる',
            '10. このシステムを使い始める前に学ぶべきことはたくさんあると思う'
        ]

    def load_data(self):
        """Load and preprocess questionnaire data"""
        print("=" * 80)
        print("LOADING QUESTIONNAIRE DATA")
        print("=" * 80)

        # Read CSV
        self.df_raw = pd.read_csv(self.csv_path, encoding='utf-8')
        print(f"\nRaw data shape: {self.df_raw.shape}")
        print(f"Columns: {self.df_raw.columns.tolist()}")

        # Extract participant identifier from email (more reliable than name)
        self.df_raw['participant'] = self.df_raw['ユーザー名'].apply(
            lambda x: x.split('@')[0].split('.')[0] if isinstance(x, str) and '@' in x else str(x).lower()
        )

        # Standardize condition labels
        self.df_raw['condition'] = self.df_raw['実験タイプ'].str.upper()

        # Select relevant columns
        cols_to_keep = ['participant', 'condition'] + list(self.likert_items.keys()) + self.sus_items
        self.df_clean = self.df_raw[cols_to_keep].copy()

        print(f"\nCleaned data shape: {self.df_clean.shape}")
        print(f"Participants: {sorted(self.df_clean['participant'].unique())}")
        print(f"Number of unique participants: {self.df_clean['participant'].nunique()}")
        print(f"Conditions: {sorted(self.df_clean['condition'].unique())}")

        # Create long format for Likert items
        self._create_likert_long_format()

        return True

    def _create_likert_long_format(self):
        """Convert Likert items to long format"""
        id_vars = ['participant', 'condition']
        value_vars = list(self.likert_items.keys())

        self.df_likert_long = pd.melt(
            self.df_clean,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='item_jp',
            value_name='rating'
        )

        # Add English item names
        self.df_likert_long['item'] = self.df_likert_long['item_jp'].map(self.likert_items)

        print(f"\nLikert long format shape: {self.df_likert_long.shape}")

    def validate_design(self):
        """Verify within-subject design is complete"""
        print("\n" + "=" * 80)
        print("VALIDATING WITHIN-SUBJECT DESIGN")
        print("=" * 80)

        # Check each participant has both conditions
        participant_counts = self.df_clean.groupby('participant')['condition'].apply(list)

        all_valid = True
        for participant, conditions in participant_counts.items():
            if len(conditions) != 2:
                print(f"✗ {participant}: Missing condition (has {len(conditions)})")
                all_valid = False
            elif set(conditions) != {'A', 'B'}:
                print(f"✗ {participant}: Invalid conditions ({conditions})")
                all_valid = False

        if all_valid:
            print(f"✓ All {len(participant_counts)} participants have both conditions A and B")
            print("✓ Within-subject design is valid")
        else:
            print("\n✗ WARNING: Some participants missing conditions")

        return all_valid

    def analyze_likert_items(self):
        """Analyze all 6 Likert scale items"""
        print("\n" + "=" * 80)
        print("LIKERT SCALE ANALYSIS (6 ITEMS)")
        print("=" * 80)

        self.results['likert'] = {}

        for item_jp, item_en in self.likert_items.items():
            print(f"\n{'=' * 80}")
            print(f"Item: {item_en}")
            print(f"      {item_jp}")
            print(f"{'=' * 80}")

            item_data = self.df_likert_long[self.df_likert_long['item'] == item_en]

            # Descriptive statistics
            desc = self._compute_likert_descriptives(item_data)
            print("\nDescriptive Statistics:")
            print(desc.to_string(index=False))

            # Normality testing
            normality = self._test_likert_normality(item_data)
            print(f"\nNormality Test (Shapiro-Wilk):")
            print(f"  Condition A: W = {normality['W_A']:.4f}, p = {normality['p_A']:.4f} {normality['interpretation_A']}")
            print(f"  Condition B: W = {normality['W_B']:.4f}, p = {normality['p_B']:.4f} {normality['interpretation_B']}")
            print(f"  → Using {normality['test_type']} test")

            # Paired comparison
            comparison = self._run_likert_paired_comparison(item_data, normality['parametric'])
            print(f"\nPaired Comparison ({comparison['test_name']}):")
            print(f"  Condition A: M = {comparison['mean_A']:.2f}, SD = {comparison['sd_A']:.2f}")
            print(f"  Condition B: M = {comparison['mean_B']:.2f}, SD = {comparison['sd_B']:.2f}")
            print(f"  Difference (B - A): {comparison['diff']:.2f}")
            print(f"  Test statistic: {comparison['statistic_name']} = {comparison['statistic']:.4f}, p = {comparison['p_value']:.4f}")
            print(f"  Effect size: {comparison['effect_size_name']} = {comparison['effect_size']:.3f} ({comparison['effect_magnitude']})")
            print(f"  Result: {comparison['interpretation']}")

            # Store results
            self.results['likert'][item_en] = {
                'item_jp': item_jp,
                'descriptive': desc,
                'normality': normality,
                'comparison': comparison
            }

        print("\n" + "=" * 80)
        print("LIKERT ANALYSIS SUMMARY")
        print("=" * 80)

        sig_items = [item for item, res in self.results['likert'].items()
                     if res['comparison']['p_value'] < 0.05]

        if sig_items:
            print(f"\n{len(sig_items)} items showed significant differences:")
            for item in sig_items:
                res = self.results['likert'][item]
                direction = "B > A" if res['comparison']['diff'] > 0 else "A > B"
                print(f"  - {item}: p = {res['comparison']['p_value']:.4f}, {direction}, "
                      f"effect size = {res['comparison']['effect_size']:.3f}")
        else:
            print("\nNo significant differences found across any Likert items")

    def _compute_likert_descriptives(self, item_data):
        """Compute descriptive statistics for a Likert item"""
        desc = item_data.groupby('condition')['rating'].agg([
            'count',
            ('mean', 'mean'),
            ('std', 'std'),
            ('median', 'median'),
            ('q1', lambda x: x.quantile(0.25)),
            ('q3', lambda x: x.quantile(0.75)),
            'min',
            'max'
        ]).reset_index()

        return desc

    def _test_likert_normality(self, item_data):
        """Test normality for both conditions of a Likert item"""
        data_a = item_data[item_data['condition'] == 'A']['rating']
        data_b = item_data[item_data['condition'] == 'B']['rating']

        W_a, p_a = stats.shapiro(data_a)
        W_b, p_b = stats.shapiro(data_b)

        interp_a = "(Normal)" if p_a > 0.05 else "(NOT Normal)"
        interp_b = "(Normal)" if p_b > 0.05 else "(NOT Normal)"

        parametric = (p_a > 0.05) and (p_b > 0.05)
        test_type = "Parametric (paired t-test)" if parametric else "Non-parametric (Wilcoxon)"

        return {
            'W_A': W_a,
            'p_A': p_a,
            'interpretation_A': interp_a,
            'W_B': W_b,
            'p_B': p_b,
            'interpretation_B': interp_b,
            'parametric': parametric,
            'test_type': test_type
        }

    def _run_likert_paired_comparison(self, item_data, parametric):
        """Run paired comparison test for a Likert item"""
        # Get paired data (sorted by participant to ensure alignment)
        data_a = item_data[item_data['condition'] == 'A'].sort_values('participant')['rating'].values
        data_b = item_data[item_data['condition'] == 'B'].sort_values('participant')['rating'].values

        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        sd_a = np.std(data_a, ddof=1)
        sd_b = np.std(data_b, ddof=1)
        diff = mean_b - mean_a

        if parametric:
            # Paired t-test
            statistic, p_value = stats.ttest_rel(data_a, data_b)

            # Effect size: Hedges' g
            effect_size = pg.compute_effsize(data_a, data_b, paired=True, eftype='hedges')
            effect_size_name = "Hedges' g"
            statistic_name = "t"
            test_name = "Paired t-test"
        else:
            # Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(data_a, data_b)

            # Effect size: rank-biserial correlation
            n = len(data_a)
            effect_size = 1 - (2 * statistic) / (n * (n + 1))
            effect_size_name = "Rank-biserial"
            statistic_name = "W"
            test_name = "Wilcoxon signed-rank"

        # Effect magnitude
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            magnitude = "negligible"
        elif abs_effect < 0.5:
            magnitude = "small"
        elif abs_effect < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        # Interpretation
        if p_value < 0.05:
            if diff > 0:
                interp = f"Significant difference: Condition B > Condition A (p < 0.05)"
            else:
                interp = f"Significant difference: Condition A > Condition B (p < 0.05)"
        else:
            interp = "No significant difference (p >= 0.05)"

        return {
            'mean_A': mean_a,
            'mean_B': mean_b,
            'sd_A': sd_a,
            'sd_B': sd_b,
            'diff': diff,
            'statistic': statistic,
            'statistic_name': statistic_name,
            'p_value': p_value,
            'test_name': test_name,
            'effect_size': effect_size,
            'effect_size_name': effect_size_name,
            'effect_magnitude': magnitude,
            'interpretation': interp
        }

    def calculate_sus_scores(self):
        """Calculate SUS scores for each participant × condition"""
        print("\n" + "=" * 80)
        print("CALCULATING SUS SCORES")
        print("=" * 80)

        sus_data = []

        for _, row in self.df_clean.iterrows():
            participant = row['participant']
            condition = row['condition']

            # Extract SUS responses
            responses = [row[item] for item in self.sus_items]

            # Calculate SUS score
            # Odd questions (1,3,5,7,9): contribution = response - 1
            # Even questions (2,4,6,8,10): contribution = 5 - response
            contributions = []
            for i, response in enumerate(responses, start=1):
                if i % 2 == 1:  # Odd
                    contributions.append(response - 1)
                else:  # Even
                    contributions.append(5 - response)

            sus_score = sum(contributions) * 2.5

            sus_data.append({
                'participant': participant,
                'condition': condition,
                'sus_score': sus_score,
                **{f'Q{i+1}': responses[i] for i in range(10)}
            })

        self.df_sus_scores = pd.DataFrame(sus_data)

        print(f"\nSUS scores calculated for {len(self.df_sus_scores)} responses")
        print(f"\nSUS Score Range: {self.df_sus_scores['sus_score'].min():.1f} - {self.df_sus_scores['sus_score'].max():.1f}")
        print(f"Overall Mean: {self.df_sus_scores['sus_score'].mean():.1f}")
        print(f"\nSample SUS Scores:")
        print(self.df_sus_scores[['participant', 'condition', 'sus_score']].head(10).to_string(index=False))

        return self.df_sus_scores

    def analyze_sus_scores(self):
        """Analyze SUS scores between conditions"""
        print("\n" + "=" * 80)
        print("SUS SCORE ANALYSIS")
        print("=" * 80)

        # Descriptive statistics
        desc = self._compute_sus_descriptives()
        print("\nDescriptive Statistics:")
        print(desc.to_string(index=False))

        # Compare to benchmark
        print(f"\nComparison to SUS Benchmark (68 points):")
        for condition in ['A', 'B']:
            mean_score = desc[desc['condition'] == condition]['mean'].values[0]
            if mean_score >= 70:
                assessment = "Good usability (≥70)"
            elif mean_score >= 68:
                assessment = "Average usability (≥68)"
            elif mean_score >= 50:
                assessment = "Below average (50-68)"
            else:
                assessment = "Poor usability (<50)"
            print(f"  Condition {condition}: {mean_score:.1f} - {assessment}")

        # Normality testing
        normality = self._test_sus_normality()
        print(f"\nNormality Test (Shapiro-Wilk):")
        print(f"  Condition A: W = {normality['W_A']:.4f}, p = {normality['p_A']:.4f} {normality['interpretation_A']}")
        print(f"  Condition B: W = {normality['W_B']:.4f}, p = {normality['p_B']:.4f} {normality['interpretation_B']}")
        print(f"  → Using {normality['test_type']} test")

        # Paired comparison
        comparison = self._run_sus_paired_comparison(normality['parametric'])
        print(f"\nPaired Comparison ({comparison['test_name']}):")
        print(f"  Condition A: M = {comparison['mean_A']:.2f}, SD = {comparison['sd_A']:.2f}")
        print(f"  Condition B: M = {comparison['mean_B']:.2f}, SD = {comparison['sd_B']:.2f}")
        print(f"  Difference (B - A): {comparison['diff']:.2f}")
        print(f"  Test statistic: {comparison['statistic_name']} = {comparison['statistic']:.4f}, p = {comparison['p_value']:.4f}")
        print(f"  Effect size: {comparison['effect_size_name']} = {comparison['effect_size']:.3f} ({comparison['effect_magnitude']})")
        print(f"  Result: {comparison['interpretation']}")

        # Store results
        self.results['sus'] = {
            'descriptive': desc,
            'normality': normality,
            'comparison': comparison
        }

    def _compute_sus_descriptives(self):
        """Compute descriptive statistics for SUS scores"""
        desc = self.df_sus_scores.groupby('condition')['sus_score'].agg([
            'count',
            ('mean', 'mean'),
            ('std', 'std'),
            ('median', 'median'),
            ('q1', lambda x: x.quantile(0.25)),
            ('q3', lambda x: x.quantile(0.75)),
            'min',
            'max'
        ]).reset_index()

        return desc

    def _test_sus_normality(self):
        """Test normality for SUS scores in both conditions"""
        data_a = self.df_sus_scores[self.df_sus_scores['condition'] == 'A']['sus_score']
        data_b = self.df_sus_scores[self.df_sus_scores['condition'] == 'B']['sus_score']

        W_a, p_a = stats.shapiro(data_a)
        W_b, p_b = stats.shapiro(data_b)

        interp_a = "(Normal)" if p_a > 0.05 else "(NOT Normal)"
        interp_b = "(Normal)" if p_b > 0.05 else "(NOT Normal)"

        parametric = (p_a > 0.05) and (p_b > 0.05)
        test_type = "Parametric (paired t-test)" if parametric else "Non-parametric (Wilcoxon)"

        return {
            'W_A': W_a,
            'p_A': p_a,
            'interpretation_A': interp_a,
            'W_B': W_b,
            'p_B': p_b,
            'interpretation_B': interp_b,
            'parametric': parametric,
            'test_type': test_type
        }

    def _run_sus_paired_comparison(self, parametric):
        """Run paired comparison test for SUS scores"""
        # Get paired data (sorted by participant)
        data_a = self.df_sus_scores[self.df_sus_scores['condition'] == 'A'].sort_values('participant')['sus_score'].values
        data_b = self.df_sus_scores[self.df_sus_scores['condition'] == 'B'].sort_values('participant')['sus_score'].values

        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        sd_a = np.std(data_a, ddof=1)
        sd_b = np.std(data_b, ddof=1)
        diff = mean_b - mean_a

        if parametric:
            # Paired t-test
            statistic, p_value = stats.ttest_rel(data_a, data_b)

            # Effect size: Hedges' g
            effect_size = pg.compute_effsize(data_a, data_b, paired=True, eftype='hedges')
            effect_size_name = "Hedges' g"
            statistic_name = "t"
            test_name = "Paired t-test"
        else:
            # Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(data_a, data_b)

            # Effect size: rank-biserial correlation
            n = len(data_a)
            effect_size = 1 - (2 * statistic) / (n * (n + 1))
            effect_size_name = "Rank-biserial"
            statistic_name = "W"
            test_name = "Wilcoxon signed-rank"

        # Effect magnitude
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            magnitude = "negligible"
        elif abs_effect < 0.5:
            magnitude = "small"
        elif abs_effect < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        # Interpretation
        if p_value < 0.05:
            if diff > 0:
                interp = f"Significant difference: Condition B has higher usability (p < 0.05)"
            else:
                interp = f"Significant difference: Condition A has higher usability (p < 0.05)"
        else:
            interp = "No significant difference in usability (p >= 0.05)"

        return {
            'mean_A': mean_a,
            'mean_B': mean_b,
            'sd_A': sd_a,
            'sd_B': sd_b,
            'diff': diff,
            'statistic': statistic,
            'statistic_name': statistic_name,
            'p_value': p_value,
            'test_name': test_name,
            'effect_size': effect_size,
            'effect_size_name': effect_size_name,
            'effect_magnitude': magnitude,
            'interpretation': interp
        }

    def plot_likert_boxplots(self):
        """Create boxplots for all 6 Likert items"""
        print("\n" + "=" * 80)
        print("CREATING LIKERT BOXPLOTS")
        print("=" * 80)

        self.viz_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (item_en, item_jp) in enumerate(self.likert_items_jp.items()):
            ax = axes[idx]
            item_data = self.df_likert_long[self.df_likert_long['item'] == item_en]

            # Boxplot
            bp = ax.boxplot(
                [item_data[item_data['condition'] == cond]['rating'].values for cond in ['A', 'B']],
                positions=[0, 1],
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5)
            )

            # Add individual points
            for i, cond in enumerate(['A', 'B']):
                cond_data = item_data[item_data['condition'] == cond]['rating'].values
                x = np.random.normal(i, 0.04, size=len(cond_data))
                ax.scatter(x, cond_data, alpha=0.5, s=40, color='darkblue')

            # Add significance brackets if significant
            p_value = self.results['likert'][item_en]['comparison']['p_value']
            y_max = item_data['rating'].max()
            add_significance_brackets(ax, p_value, 0, 1, y_max, height_increment=0.5)

            # Labels and formatting
            ax.set_title(f"{item_jp}\n(1: 否定的, 7: 肯定的)", fontsize=11, fontweight='bold')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Condition A', 'Condition B'])
            ax.set_ylabel('Rating', fontsize=10)
            ax.set_ylim(0.5, 7.5)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = "likert_boxplots.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

        # Create individual plots for each Likert item
        print("\n" + "=" * 80)
        print("CREATING INDIVIDUAL LIKERT BOXPLOTS")
        print("=" * 80)

        for item_en, item_jp in self.likert_items_jp.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            item_data = self.df_likert_long[self.df_likert_long['item'] == item_en]

            # Boxplot
            bp = ax.boxplot(
                [item_data[item_data['condition'] == cond]['rating'].values for cond in ['A', 'B']],
                positions=[0, 1],
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5)
            )

            # Add individual points
            for i, cond in enumerate(['A', 'B']):
                cond_data = item_data[item_data['condition'] == cond]['rating'].values
                x = np.random.normal(i, 0.04, size=len(cond_data))
                ax.scatter(x, cond_data, alpha=0.5, s=40, color='darkblue')

            # Add significance brackets if significant
            p_value = self.results['likert'][item_en]['comparison']['p_value']
            y_max = item_data['rating'].max()
            add_significance_brackets(ax, p_value, 0, 1, y_max, height_increment=0.5)

            # Labels and formatting
            ax.set_title(f"{item_jp}\n(1: 否定的, 7: 肯定的)", fontsize=11, fontweight='bold')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Condition A', 'Condition B'])
            ax.set_ylabel('Rating', fontsize=10)
            ax.set_ylim(0.5, 7.5)
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            filename = f"likert_{item_en}_boxplot.png"
            plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
            plt.close()

    def plot_likert_heatmap(self):
        """Create heatmap of median Likert ratings"""
        print("\n" + "=" * 80)
        print("CREATING LIKERT HEATMAP")
        print("=" * 80)

        # Calculate median ratings
        pivot_data = self.df_likert_long.pivot_table(
            values='rating',
            index='item',
            columns='condition',
            aggfunc='median'
        )

        # Reorder rows by item order
        pivot_data = pivot_data.reindex([item for item in self.likert_items.values()])

        # Rename index for display
        pivot_data.index = [f"{item}\n({self.likert_items_jp[item]})"
                           for item in pivot_data.index]

        fig, ax = plt.subplots(figsize=(8, 10))

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            vmin=1,
            vmax=7,
            cbar_kws={'label': 'Median Rating'},
            linewidths=0.5,
            ax=ax
        )

        ax.set_title('Median Likert Ratings by Condition and Item',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('Evaluation Item', fontsize=12)

        plt.tight_layout()
        filename = "likert_heatmap.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

    def plot_sus_boxplot(self):
        """Create boxplot for SUS scores"""
        print("\n" + "=" * 80)
        print("CREATING SUS BOXPLOT")
        print("=" * 80)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Boxplot
        bp = ax.boxplot(
            [self.df_sus_scores[self.df_sus_scores['condition'] == cond]['sus_score'].values
             for cond in ['A', 'B']],
            positions=[0, 1],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor='lightgreen', alpha=0.7),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5)
        )

        # Add connected individual points (showing pairing)
        for participant in self.df_sus_scores['participant'].unique():
            part_data = self.df_sus_scores[self.df_sus_scores['participant'] == participant]
            score_a = part_data[part_data['condition'] == 'A']['sus_score'].values[0]
            score_b = part_data[part_data['condition'] == 'B']['sus_score'].values[0]
            ax.plot([0, 1], [score_a, score_b], 'o-', alpha=0.4, color='gray', linewidth=1)

        # Add benchmark line
        ax.axhline(y=68, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Benchmark (68)')
        ax.axhline(y=70, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Good (≥70)')

        # Add significance brackets
        p_value = self.results['sus']['comparison']['p_value']
        y_max = self.df_sus_scores['sus_score'].max()
        add_significance_brackets(ax, p_value, 0, 1, y_max, height_increment=5)

        # Labels and formatting
        ax.set_title('SUS Scores by Condition\n(System Usability Scale: 0-100)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Condition A\n(Smartphone)', 'Condition B\n(Half-mirror)'], fontsize=11)
        ax.set_ylabel('SUS Score', fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = "sus_boxplot.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

    def plot_sus_trajectories(self):
        """Create line plot showing individual SUS score changes"""
        print("\n" + "=" * 80)
        print("CREATING SUS TRAJECTORIES")
        print("=" * 80)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot each participant's trajectory
        for participant in self.df_sus_scores['participant'].unique():
            part_data = self.df_sus_scores[self.df_sus_scores['participant'] == participant].sort_values('condition')
            scores = part_data['sus_score'].values

            # Determine color based on direction
            if scores[0] < scores[1]:  # A < B (improvement)
                color = 'green'
                alpha = 0.6
            else:  # A >= B (decrease or no change)
                color = 'red'
                alpha = 0.6

            ax.plot([0, 1], scores, 'o-', color=color, alpha=alpha, linewidth=2, markersize=8)

        # Add benchmark lines
        ax.axhline(y=68, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Benchmark (68)')
        ax.axhline(y=70, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (≥70)')

        # Labels and formatting
        ax.set_title('Individual SUS Score Changes (A → B)\n'
                    'Green = Improvement, Red = Decrease',
                    fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Condition A\n(Smartphone)', 'Condition B\n(Half-mirror)'], fontsize=11)
        ax.set_ylabel('SUS Score', fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = "sus_trajectories.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

    def plot_sus_comparison(self):
        """Create bar chart comparing mean SUS scores"""
        print("\n" + "=" * 80)
        print("CREATING SUS COMPARISON BAR CHART")
        print("=" * 80)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Calculate means and SDs
        means = self.df_sus_scores.groupby('condition')['sus_score'].mean()
        stds = self.df_sus_scores.groupby('condition')['sus_score'].std()

        # Bar chart
        bars = ax.bar([0, 1], means, width=0.6, color=['lightblue', 'lightgreen'],
                      edgecolor='black', linewidth=1.5, alpha=0.7)

        # Error bars
        ax.errorbar([0, 1], means, yerr=stds, fmt='none', ecolor='black',
                   capsize=10, capthick=2, linewidth=2)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 2, f'{mean:.1f}', ha='center', va='bottom',
                   fontsize=12, fontweight='bold')

        # Add benchmark lines
        ax.axhline(y=68, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Benchmark (68)')
        ax.axhline(y=70, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (≥70)')
        ax.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Poor (<50)')

        # Labels and formatting
        ax.set_title('Mean SUS Scores by Condition\n(Error bars = ±1 SD)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Condition A\n(Smartphone)', 'Condition B\n(Half-mirror)'], fontsize=11)
        ax.set_ylabel('Mean SUS Score', fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = "sus_comparison.png"
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

    def save_results_to_excel(self):
        """Save all results to Excel file"""
        print("\n" + "=" * 80)
        print("SAVING RESULTS TO EXCEL")
        print("=" * 80)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "statistical_results.xlsx"

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Metadata
            metadata = pd.DataFrame({
                'Parameter': ['Analysis Date', 'Participants', 'Conditions', 'Design', 'Likert Items', 'SUS Items'],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    self.df_clean['participant'].nunique(),
                    'A, B',
                    'Within-subject (paired)',
                    6,
                    10
                ]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)

            # Likert: Descriptive stats (combined)
            likert_desc_all = []
            for item_en, res in self.results['likert'].items():
                desc = res['descriptive'].copy()
                desc.insert(0, 'item', item_en)
                desc.insert(1, 'item_jp', res['item_jp'])
                likert_desc_all.append(desc)
            pd.concat(likert_desc_all, ignore_index=True).to_excel(
                writer, sheet_name='Likert_Descriptive', index=False)

            # Likert: Normality tests
            likert_norm_all = []
            for item_en, res in self.results['likert'].items():
                norm = res['normality']
                likert_norm_all.append({
                    'item': item_en,
                    'W_A': norm['W_A'],
                    'p_A': norm['p_A'],
                    'normal_A': 'Yes' if norm['p_A'] > 0.05 else 'No',
                    'W_B': norm['W_B'],
                    'p_B': norm['p_B'],
                    'normal_B': 'Yes' if norm['p_B'] > 0.05 else 'No',
                    'test_type': norm['test_type']
                })
            pd.DataFrame(likert_norm_all).to_excel(
                writer, sheet_name='Likert_Normality', index=False)

            # Likert: Paired tests
            likert_test_all = []
            for item_en, res in self.results['likert'].items():
                comp = res['comparison']
                likert_test_all.append({
                    'item': item_en,
                    'mean_A': comp['mean_A'],
                    'sd_A': comp['sd_A'],
                    'mean_B': comp['mean_B'],
                    'sd_B': comp['sd_B'],
                    'diff_B_minus_A': comp['diff'],
                    'test_name': comp['test_name'],
                    'statistic': comp['statistic'],
                    'p_value': comp['p_value'],
                    'effect_size': comp['effect_size'],
                    'effect_size_type': comp['effect_size_name'],
                    'effect_magnitude': comp['effect_magnitude'],
                    'significant': 'Yes' if comp['p_value'] < 0.05 else 'No'
                })
            pd.DataFrame(likert_test_all).to_excel(
                writer, sheet_name='Likert_Paired_Tests', index=False)

            # SUS: Raw responses
            self.df_sus_scores.to_excel(writer, sheet_name='SUS_Raw_Scores', index=False)

            # SUS: Descriptive
            self.results['sus']['descriptive'].to_excel(
                writer, sheet_name='SUS_Descriptive', index=False)

            # SUS: Normality
            norm = self.results['sus']['normality']
            sus_norm = pd.DataFrame([{
                'condition': 'A',
                'W': norm['W_A'],
                'p_value': norm['p_A'],
                'normal': 'Yes' if norm['p_A'] > 0.05 else 'No'
            }, {
                'condition': 'B',
                'W': norm['W_B'],
                'p_value': norm['p_B'],
                'normal': 'Yes' if norm['p_B'] > 0.05 else 'No'
            }])
            sus_norm.to_excel(writer, sheet_name='SUS_Normality', index=False)

            # SUS: Paired test
            comp = self.results['sus']['comparison']
            sus_test = pd.DataFrame([{
                'mean_A': comp['mean_A'],
                'sd_A': comp['sd_A'],
                'mean_B': comp['mean_B'],
                'sd_B': comp['sd_B'],
                'diff_B_minus_A': comp['diff'],
                'test_name': comp['test_name'],
                'statistic': comp['statistic'],
                'p_value': comp['p_value'],
                'effect_size': comp['effect_size'],
                'effect_size_type': comp['effect_size_name'],
                'effect_magnitude': comp['effect_magnitude'],
                'significant': 'Yes' if comp['p_value'] < 0.05 else 'No',
                'interpretation': comp['interpretation']
            }])
            sus_test.to_excel(writer, sheet_name='SUS_Paired_Test', index=False)

        print(f"✓ Results saved to: {output_path}")

    def save_analysis_report_md(self):
        """Generate comprehensive Markdown analysis report"""
        print("\n" + "=" * 80)
        print("GENERATING MARKDOWN REPORT")
        print("=" * 80)

        report_path = self.output_dir / "analysis_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Post-Experiment Questionnaire Analysis\n\n")

            # Experiment Overview
            f.write("## Experiment Overview\n\n")
            f.write(f"- **Design**: 2 conditions within-subject (A vs B)\n")
            f.write(f"- **Participants**: {self.df_clean['participant'].nunique()}\n")
            f.write(f"- **Questionnaire Components**:\n")
            f.write(f"  - 6 Likert scale items (7-point scale): Task/translation evaluation\n")
            f.write(f"  - 10 SUS questions (5-point scale): System usability\n")
            f.write(f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Part 1: Likert Scale Analysis
            f.write("## Part 1: Task/Translation Evaluation (Likert Scale)\n\n")

            f.write("### 1. Descriptive Statistics\n\n")
            f.write("| Item | Condition | N | Mean | SD | Median | Min | Max |\n")
            f.write("|------|-----------|---|------|----|---------|----|-----|\n")
            for item_en, res in self.results['likert'].items():
                desc = res['descriptive']
                for _, row in desc.iterrows():
                    f.write(f"| {item_en} | {row['condition']} | {row['count']:.0f} | "
                           f"{row['mean']:.2f} | {row['std']:.2f} | {row['median']:.2f} | "
                           f"{row['min']:.0f} | {row['max']:.0f} |\n")
            f.write("\n")

            f.write("### 2. Normality Testing (Shapiro-Wilk)\n\n")
            f.write("| Item | Condition | W | p-value | Normal | Test Used |\n")
            f.write("|------|-----------|---|---------|--------|------------|\n")
            for item_en, res in self.results['likert'].items():
                norm = res['normality']
                f.write(f"| {item_en} | A | {norm['W_A']:.4f} | {norm['p_A']:.4f} | "
                       f"{'Yes' if norm['p_A'] > 0.05 else 'No'} | {norm['test_type']} |\n")
                f.write(f"| {item_en} | B | {norm['W_B']:.4f} | {norm['p_B']:.4f} | "
                       f"{'Yes' if norm['p_B'] > 0.05 else 'No'} | |\n")
            f.write("\n")

            f.write("### 3. Paired Comparisons\n\n")
            for item_en, res in self.results['likert'].items():
                comp = res['comparison']
                f.write(f"#### {item_en}\n")
                f.write(f"**{res['item_jp']}**\n\n")
                f.write(f"- **Test**: {comp['test_name']}\n")
                f.write(f"- **Condition A**: M = {comp['mean_A']:.2f}, SD = {comp['sd_A']:.2f}\n")
                f.write(f"- **Condition B**: M = {comp['mean_B']:.2f}, SD = {comp['sd_B']:.2f}\n")
                f.write(f"- **Difference (B - A)**: {comp['diff']:.2f}\n")
                f.write(f"- **Statistic**: {comp['statistic_name']} = {comp['statistic']:.4f}, p = {comp['p_value']:.4f}\n")
                f.write(f"- **Effect Size**: {comp['effect_size_name']} = {comp['effect_size']:.3f} ({comp['effect_magnitude']})\n")
                f.write(f"- **Result**: {comp['interpretation']}\n\n")

            f.write("### Summary: Likert Items\n\n")
            sig_items = [(item, res) for item, res in self.results['likert'].items()
                        if res['comparison']['p_value'] < 0.05]
            if sig_items:
                f.write(f"**{len(sig_items)} items showed significant differences:**\n\n")
                for item, res in sig_items:
                    direction = "B > A" if res['comparison']['diff'] > 0 else "A > B"
                    f.write(f"- **{item}**: p = {res['comparison']['p_value']:.4f}, {direction}, "
                           f"effect size = {res['comparison']['effect_size']:.3f}\n")
            else:
                f.write("No significant differences found across any Likert items.\n")
            f.write("\n")

            # Part 2: SUS Analysis
            f.write("## Part 2: System Usability Scale (SUS) Analysis\n\n")

            f.write("### 1. SUS Score Calculation\n\n")
            f.write("**Methodology:**\n")
            f.write("- Odd questions (1,3,5,7,9): contribution = response - 1\n")
            f.write("- Even questions (2,4,6,8,10): contribution = 5 - response\n")
            f.write("- SUS Score = sum(contributions) × 2.5\n")
            f.write("- Score range: 0-100\n")
            f.write("- Benchmark: 68 points (average), ≥70 (good), <50 (poor)\n\n")

            f.write("### 2. SUS Scores by Participant\n\n")
            f.write("| Participant | Condition A | Condition B | Change (B-A) |\n")
            f.write("|-------------|-------------|-------------|---------------|\n")
            for participant in sorted(self.df_sus_scores['participant'].unique()):
                part_data = self.df_sus_scores[self.df_sus_scores['participant'] == participant]
                score_a = part_data[part_data['condition'] == 'A']['sus_score'].values[0]
                score_b = part_data[part_data['condition'] == 'B']['sus_score'].values[0]
                change = score_b - score_a
                f.write(f"| {participant} | {score_a:.1f} | {score_b:.1f} | {change:+.1f} |\n")
            f.write("\n")

            f.write("### 3. Descriptive Statistics\n\n")
            desc = self.results['sus']['descriptive']
            for _, row in desc.iterrows():
                f.write(f"**Condition {row['condition']}:**\n")
                f.write(f"- Mean: {row['mean']:.2f}\n")
                f.write(f"- SD: {row['std']:.2f}\n")
                f.write(f"- Median: {row['median']:.2f}\n")
                f.write(f"- Range: {row['min']:.1f} - {row['max']:.1f}\n")

                # Comparison to benchmark
                if row['mean'] >= 70:
                    assessment = "Good usability (≥70)"
                elif row['mean'] >= 68:
                    assessment = "Average usability (≥68)"
                elif row['mean'] >= 50:
                    assessment = "Below average (50-68)"
                else:
                    assessment = "Poor usability (<50)"
                f.write(f"- **Assessment**: {assessment}\n\n")

            f.write("### 4. Statistical Comparison\n\n")
            norm = self.results['sus']['normality']
            comp = self.results['sus']['comparison']

            f.write(f"**Normality Test (Shapiro-Wilk):**\n")
            f.write(f"- Condition A: W = {norm['W_A']:.4f}, p = {norm['p_A']:.4f} {norm['interpretation_A']}\n")
            f.write(f"- Condition B: W = {norm['W_B']:.4f}, p = {norm['p_B']:.4f} {norm['interpretation_B']}\n")
            f.write(f"- Test used: {norm['test_type']}\n\n")

            f.write(f"**Paired Comparison ({comp['test_name']}):**\n")
            f.write(f"- Statistic: {comp['statistic_name']} = {comp['statistic']:.4f}\n")
            f.write(f"- p-value: {comp['p_value']:.4f}\n")
            f.write(f"- Effect size: {comp['effect_size_name']} = {comp['effect_size']:.3f} ({comp['effect_magnitude']})\n")
            f.write(f"- **Result**: {comp['interpretation']}\n\n")

            f.write("### Interpretation\n\n")
            if comp['diff'] > 0:
                better = "B (Half-mirror)"
                worse = "A (Smartphone)"
            else:
                better = "A (Smartphone)"
                worse = "B (Half-mirror)"

            if comp['p_value'] < 0.05:
                f.write(f"Condition {better} showed significantly higher usability than Condition {worse}. ")
            else:
                f.write(f"No significant difference in usability between conditions was found. ")

            mean_a = comp['mean_A']
            mean_b = comp['mean_B']
            if mean_a >= 70 and mean_b >= 70:
                f.write(f"Both conditions achieved good usability scores (≥70).\n")
            elif mean_a >= 68 or mean_b >= 68:
                f.write(f"At least one condition met the average benchmark (68).\n")
            else:
                f.write(f"Both conditions scored below the average benchmark (68), suggesting room for improvement.\n")
            f.write("\n")

            # Visualizations
            f.write("## Visualizations\n\n")
            f.write("- Likert boxplots: `visualizations/likert_boxplots.png`\n")
            f.write("- Likert heatmap: `visualizations/likert_heatmap.png`\n")
            f.write("- SUS boxplot: `visualizations/sus_boxplot.png`\n")
            f.write("- SUS trajectories: `visualizations/sus_trajectories.png`\n")
            f.write("- SUS comparison: `visualizations/sus_comparison.png`\n\n")

            # Statistical Methods
            f.write("## Statistical Methods\n\n")
            f.write("- **Design**: Within-subject (paired comparisons)\n")
            f.write("- **Normality testing**: Shapiro-Wilk test\n")
            f.write("- **Paired tests**: t-test (parametric) or Wilcoxon signed-rank (non-parametric)\n")
            f.write("- **Effect sizes**: Hedges' g (parametric) or rank-biserial correlation (non-parametric)\n")
            f.write("- **Significance level**: α = 0.05\n\n")

            # Overall Conclusions
            f.write("## Overall Conclusions\n\n")

            # Count significant Likert items
            n_sig_likert = sum(1 for res in self.results['likert'].values()
                             if res['comparison']['p_value'] < 0.05)

            f.write(f"**Likert Scale Evaluation:**\n")
            f.write(f"- {n_sig_likert} out of 6 task/translation evaluation items showed significant differences\n")

            if n_sig_likert > 0:
                f.write(f"- Significant differences indicate distinct user experiences between conditions\n")

            f.write(f"\n**System Usability (SUS):**\n")
            f.write(f"- Condition A mean: {comp['mean_A']:.1f}\n")
            f.write(f"- Condition B mean: {comp['mean_B']:.1f}\n")

            if comp['p_value'] < 0.05:
                f.write(f"- **Significant difference** in overall system usability (p = {comp['p_value']:.4f})\n")
            else:
                f.write(f"- No significant difference in overall system usability (p = {comp['p_value']:.4f})\n")

            f.write("\n---\n")
            f.write(f"Generated by questionnaire_analysis.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"✓ Report saved to: {report_path}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 80)
        print("POST-EXPERIMENT QUESTIONNAIRE ANALYSIS")
        print("=" * 80)
        print("\nAnalyzing:")
        print("  - 6 Likert scale items (7-point scale)")
        print("  - 10 SUS questions (System Usability Scale)")
        print(f"  - Design: Within-subject (A vs B)")
        print(f"  - Participants: 13")

        # Load data
        if not self.load_data():
            print("Error loading data")
            return None

        # Validate design
        if not self.validate_design():
            print("Warning: Design validation failed")

        # Analyze Likert items
        self.analyze_likert_items()

        # Calculate and analyze SUS scores
        self.calculate_sus_scores()
        self.analyze_sus_scores()

        # Create visualizations
        self.plot_likert_boxplots()
        self.plot_likert_heatmap()
        self.plot_sus_boxplot()
        self.plot_sus_trajectories()
        self.plot_sus_comparison()

        # Save results
        self.save_results_to_excel()
        self.save_analysis_report_md()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\n✓ Results directory: {self.output_dir}")
        print("✓ Statistical results: statistical_results.xlsx")
        print("✓ Analysis report: analysis_report.md")
        print("✓ Visualizations: visualizations/")

        return self.results


def main():
    """Main execution function"""
    import sys

    # Parse command-line arguments
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path(__file__).parent.parent.parent / "output" / "main_expt_questionnaire.csv"

    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        print("Please ensure the CSV file exists")
        return

    print(f"Analyzing: {csv_path}")

    # Create analyzer and run full analysis
    analyzer = QuestionnaireAnalyzer(csv_path)
    results = analyzer.run_full_analysis()

    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Likert summary
        print("\nLikert Scale Results:")
        for item, res in results['likert'].items():
            comp = res['comparison']
            sig = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else "n.s."
            print(f"  {item}: p = {comp['p_value']:.4f} {sig}, effect = {comp['effect_size']:.3f}")

        # SUS summary
        print("\nSUS Score Results:")
        comp = results['sus']['comparison']
        sig = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else "n.s."
        print(f"  Mean A: {comp['mean_A']:.1f}, Mean B: {comp['mean_B']:.1f}")
        print(f"  p = {comp['p_value']:.4f} {sig}, effect = {comp['effect_size']:.3f}")


if __name__ == "__main__":
    main()
