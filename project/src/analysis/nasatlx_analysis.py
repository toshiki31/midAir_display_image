#!/usr/bin/env python3
"""
NASA-TLX Analysis Script

This script analyzes NASA Task Load Index (NASA-TLX) data across
5 distance conditions (a, b, c, d, e → 0cm, 15cm, 30cm, 45cm, 60cm).

NASA-TLX measures subjective workload with 6 subscales:
1. Mental Demand (知的・知覚的要求)
2. Physical Demand (身体的要求)
3. Temporal Demand (タイムプレッシャー)
4. Performance (作業成績)
5. Effort (努力)
6. Frustration (フラストレーション)

Total Score: 0-100 (lower is better - less workload)

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
import re

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Japanese font support
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio']
plt.rcParams['axes.unicode_minus'] = False


class NASATLXAnalysis:
    """Class to analyze NASA-TLX data"""

    def __init__(self, data_dir):
        """
        Initialize with path to NASA-TLX data directory

        Args:
            data_dir: Path to directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.df_raw = None
        self.df_long = None

        # Condition mapping
        self.condition_labels = {
            'a': '0cm',
            'b': '15cm',
            'c': '30cm',
            'd': '45cm',
            'e': '60cm'
        }
        self.distance_order = ['0cm', '15cm', '30cm', '45cm', '60cm']

    def load_data(self):
        """Load all CSV files and extract total NASA-TLX scores"""
        print("=" * 80)
        print("LOADING NASA-TLX DATA")
        print("=" * 80)

        csv_files = list(self.data_dir.glob("*.csv"))
        print(f"\nFound {len(csv_files)} CSV files")

        data_list = []

        for csv_file in csv_files:
            try:
                # Parse filename: participant_condition - displayname.csv
                filename = csv_file.stem
                match = re.match(r'([a-zA-Z]+)_([a-e])', filename)

                if not match:
                    print(f"Skipping {csv_file.name}: filename format not recognized")
                    continue

                participant = match.group(1)
                condition = match.group(2)

                # Exclude taninaka (outlier)
                if 'taninaka' in participant.lower() or '谷中' in participant.lower():
                    print(f"Excluding {csv_file.name}: taninaka is an outlier")
                    continue

                # Fix typo: hodna -> honda
                if participant.lower() == 'hodna':
                    participant = 'honda'

                # Read CSV and extract total score (last row, last column)
                df_csv = pd.read_csv(csv_file)

                # Total score is in the last row, last column (スコア)
                # CSV has 8 lines (with header), pandas loads 7 rows (0-6)
                if len(df_csv) >= 7:
                    total_score = df_csv.iloc[-1, -1]  # Last row, last column

                    # Convert to numeric
                    try:
                        total_score = float(total_score)
                    except (ValueError, TypeError):
                        print(f"Warning: Could not parse score from {csv_file.name}")
                        continue

                    data_list.append({
                        'participant': participant,
                        'condition': condition,
                        'distance': self.condition_labels[condition],
                        'total_score': total_score,
                        'filename': csv_file.name
                    })

                    print(f"  Loaded: {participant}_{condition} → Score: {total_score}")

                else:
                    print(f"Warning: {csv_file.name} has insufficient rows")

            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")

        self.df_raw = pd.DataFrame(data_list)

        if len(self.df_raw) == 0:
            print("\nError: No data loaded!")
            return False

        print(f"\n{'-' * 80}")
        print(f"Successfully loaded {len(self.df_raw)} records")
        print(f"Participants: {sorted(self.df_raw['participant'].unique())}")
        print(f"Conditions: {sorted(self.df_raw['condition'].unique())}")

        print("\nData summary:")
        print(self.df_raw.groupby(['participant', 'distance'])['total_score'].first().unstack())

        return True

    def descriptive_statistics(self):
        """Calculate descriptive statistics"""
        print("\n" + "=" * 80)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 80)

        desc = self.df_raw.groupby('distance')['total_score'].agg([
            'count',
            ('median', 'median'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('q1', lambda x: x.quantile(0.25)),
            ('q3', lambda x: x.quantile(0.75)),
            'min',
            'max'
        ]).reset_index()

        # Ensure correct order
        desc['distance'] = pd.Categorical(desc['distance'],
                                         categories=self.distance_order, ordered=True)
        desc = desc.sort_values('distance')

        print("\nNASA-TLX Total Score by Distance Condition:")
        print("(Score range: 0-100, lower is better)")
        print(desc.to_string(index=False))

        # By participant
        print("\nBy Participant:")
        desc_participant = self.df_raw.groupby('participant')['total_score'].agg([
            'count', 'median', 'mean', 'min', 'max'
        ])
        print(desc_participant.to_string())

        return desc

    def test_normality(self):
        """Test normality for each distance condition"""
        print("\n" + "=" * 80)
        print("NORMALITY TESTS (Shapiro-Wilk)")
        print("=" * 80)
        print("H0: Data is normally distributed")
        print("If p < 0.05, reject H0 (data is NOT normally distributed)")

        normality_results = []

        for distance in self.distance_order:
            data = self.df_raw[self.df_raw['distance'] == distance]['total_score']

            if len(data) >= 3:
                statistic, p_value = stats.shapiro(data)
                normal = 'Yes' if p_value > 0.05 else 'No'

                normality_results.append({
                    'distance': distance,
                    'n': len(data),
                    'W': statistic,
                    'p_value': p_value,
                    'normal': normal
                })

                print(f"\n{distance}: W = {statistic:.4f}, p = {p_value:.4f} "
                      f"{'(Normal)' if p_value > 0.05 else '(NOT Normal)'}")
            else:
                print(f"\n{distance}: Insufficient data (n={len(data)})")

        df_normality = pd.DataFrame(normality_results)

        if len(df_normality) > 0:
            print("\n" + "-" * 80)
            print("Summary:")
            print(df_normality.to_string(index=False))

            all_normal = all(df_normality['normal'] == 'Yes')
            if all_normal:
                print("\n✓ All conditions meet normality assumption")
                print("  → Parametric tests (ANOVA) can be used")
            else:
                print("\n✗ Some conditions violate normality assumption")
                print("  → Non-parametric tests (Friedman) are RECOMMENDED")

        return df_normality

    def friedman_test(self):
        """Perform Friedman test"""
        print("\n" + "=" * 80)
        print("FRIEDMAN TEST (NON-PARAMETRIC)")
        print("=" * 80)
        print("Within-subject factor: Distance (5 levels)")
        print("H0: All distance conditions have identical distributions")

        # Friedman test
        friedman = pg.friedman(
            data=self.df_raw,
            dv='total_score',
            within='distance',
            subject='participant'
        )

        print("\nFriedman Test Results:")
        print(friedman.to_string(index=False))

        q_value = friedman['Q'].values[0]
        p_value = friedman['p-unc'].values[0]
        w_value = friedman['W'].values[0] if 'W' in friedman.columns else np.nan

        print(f"\n{'=' * 80}")
        print("INTERPRETATION:")
        print(f"χ²(4) = {q_value:.4f}, p = {p_value:.4f}")

        if w_value and not np.isnan(w_value):
            print(f"Kendall's W = {w_value:.4f} (effect size)")

        if p_value < 0.001:
            print("*** Highly significant effect (p < 0.001)")
            print("    → Distance significantly affects NASA-TLX workload")
            sig_level = '***'
        elif p_value < 0.01:
            print("**  Very significant effect (p < 0.01)")
            print("    → Distance significantly affects NASA-TLX workload")
            sig_level = '**'
        elif p_value < 0.05:
            print("*   Significant effect (p < 0.05)")
            print("    → Distance affects NASA-TLX workload")
            sig_level = '*'
        else:
            print("    No significant effect (p >= 0.05)")
            print("    → Distance does not significantly affect NASA-TLX workload")
            sig_level = 'n.s.'

        return friedman, p_value, sig_level

    def posthoc_wilcoxon(self):
        """Perform post-hoc pairwise Wilcoxon tests"""
        print("\n" + "=" * 80)
        print("POST-HOC PAIRWISE COMPARISONS")
        print("=" * 80)
        print("Wilcoxon signed-rank tests with Holm-Bonferroni correction")

        posthoc = pg.pairwise_tests(
            data=self.df_raw,
            dv='total_score',
            within='distance',
            subject='participant',
            parametric=False,
            padjust='holm',
            effsize='CLES'
        )

        print("\nPairwise Comparison Results:")
        display_cols = ['A', 'B', 'W-val', 'p-unc', 'p-corr', 'CLES']
        posthoc_display = posthoc[display_cols].copy()
        print(posthoc_display.to_string(index=False))

        print("\n" + "-" * 80)
        print("SIGNIFICANT COMPARISONS (p-corrected < 0.05):")

        sig_comparisons = posthoc_display[posthoc_display['p-corr'] < 0.05]

        if len(sig_comparisons) > 0:
            for idx, row in sig_comparisons.iterrows():
                direction = "higher" if row['CLES'] < 0.5 else "lower"
                print(f"\n  {row['A']} has {direction} workload than {row['B']}")
                print(f"    W = {row['W-val']:.0f}, p = {row['p-corr']:.4f}, CLES = {row['CLES']:.3f}")
        else:
            print("\n  No significant pairwise differences found after correction")

        return posthoc

    def visualize_boxplot(self, output_dir):
        """Create box plot"""
        print("\n" + "=" * 80)
        print("CREATING BOX PLOT")
        print("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.boxplot(
            data=self.df_raw,
            x='distance',
            y='total_score',
            order=self.distance_order,
            palette='RdYlGn',  # Red=high (bad), Green=low (good)
            ax=ax
        )

        ax.set_title('NASA-TLX Total Score by Distance\n(Lower is better - less workload)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('NASA-TLX Total Score (0-100)', fontsize=12)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = "boxplot_nasatlx.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

    def visualize_barplot(self, output_dir):
        """Create bar plot with error bars"""
        print("\n" + "=" * 80)
        print("CREATING BAR PLOT")
        print("=" * 80)

        output_dir = Path(output_dir)

        summary = self.df_raw.groupby('distance')['total_score'].agg(['mean', 'sem']).reset_index()
        summary['distance'] = pd.Categorical(summary['distance'],
                                            categories=self.distance_order, ordered=True)
        summary = summary.sort_values('distance')

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(summary['distance'], summary['mean'],
                     yerr=summary['sem'], capsize=8,
                     alpha=0.7, edgecolor='navy', linewidth=1.5,
                     color=plt.cm.RdYlGn_r(summary['mean'] / 100))

        ax.set_title('Mean NASA-TLX Score by Distance\n(Lower is better)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('NASA-TLX Score (Mean ± SEM)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, summary['mean']):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = "barplot_nasatlx.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

    def visualize_trajectories(self, output_dir):
        """Create individual participant trajectories"""
        print("\n" + "=" * 80)
        print("CREATING TRAJECTORY PLOT")
        print("=" * 80)

        output_dir = Path(output_dir)

        fig, ax = plt.subplots(figsize=(12, 8))

        for participant in sorted(self.df_raw['participant'].unique()):
            participant_data = self.df_raw[self.df_raw['participant'] == participant].copy()
            participant_data['distance'] = pd.Categorical(
                participant_data['distance'],
                categories=self.distance_order, ordered=True
            )
            participant_data = participant_data.sort_values('distance')

            ax.plot(participant_data['distance'], participant_data['total_score'],
                   marker='o', label=participant, linewidth=2, markersize=8)

        ax.set_title('Individual Participant NASA-TLX Trajectories\n(Lower is better)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('NASA-TLX Total Score', fontsize=12)
        ax.legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        filename = "trajectories_nasatlx.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

    def visualize_heatmap(self, output_dir):
        """Create heatmap of scores"""
        print("\n" + "=" * 80)
        print("CREATING HEATMAP")
        print("=" * 80)

        output_dir = Path(output_dir)

        # Pivot data
        pivot = self.df_raw.pivot(index='participant', columns='distance', values='total_score')
        pivot = pivot[self.distance_order]

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',  # Red=high (bad), Green=low (good)
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'NASA-TLX Score'},
            linewidths=0.5,
            ax=ax
        )

        ax.set_title('NASA-TLX Scores: Participant × Distance\n(Lower is better)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Participant', fontsize=12)

        plt.tight_layout()
        filename = "heatmap_nasatlx.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

    def save_results(self, desc, normality, friedman, posthoc, output_path):
        """Save results to Excel"""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Raw data
            self.df_raw[['participant', 'distance', 'total_score']].to_excel(
                writer, sheet_name='Raw_Data', index=False
            )

            # Descriptive statistics
            desc.to_excel(writer, sheet_name='Descriptive_Stats', index=False)

            # Normality tests
            normality.to_excel(writer, sheet_name='Normality_Tests', index=False)

            # Friedman test
            friedman.to_excel(writer, sheet_name='Friedman_Test', index=False)

            # Post-hoc tests
            if posthoc is not None:
                posthoc.to_excel(writer, sheet_name='PostHoc_Tests', index=False)

        print(f"✓ Results saved to: {output_path}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 80)
        print("NASA-TLX ANALYSIS")
        print("=" * 80)
        print("\nWithin-subject design:")
        print("  Factor: Distance (5 levels: 0cm, 15cm, 30cm, 45cm, 60cm)")
        print("  Dependent variable: NASA-TLX Total Score (0-100, lower is better)")
        print("  Method: Friedman test (non-parametric)")

        # Load data
        if not self.load_data():
            return None

        # Descriptive statistics
        desc = self.descriptive_statistics()

        # Normality tests
        normality = self.test_normality()

        # Friedman test
        friedman, p_value, sig_level = self.friedman_test()

        # Post-hoc if significant
        posthoc = None
        if p_value < 0.05:
            posthoc = self.posthoc_wilcoxon()
        else:
            print("\n" + "=" * 80)
            print("POST-HOC TESTS SKIPPED")
            print("=" * 80)
            print("Friedman test was not significant (p >= 0.05)")

        # Visualizations
        output_dir = Path(self.data_dir).parent / "nasatlx_results" / "visualizations"
        self.visualize_boxplot(output_dir)
        self.visualize_barplot(output_dir)
        self.visualize_trajectories(output_dir)
        self.visualize_heatmap(output_dir)

        # Save results
        output_path = Path(self.data_dir).parent / "nasatlx_results" / "nasatlx_results.xlsx"
        self.save_results(desc, normality, friedman, posthoc, output_path)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\n✓ Results directory: {Path(self.data_dir).parent / 'nasatlx_results'}")
        print("✓ Statistical results: nasatlx_results.xlsx")
        print("✓ Visualizations: visualizations/")

        return {
            'descriptive': desc,
            'normality': normality,
            'friedman': friedman,
            'posthoc': posthoc,
            'p_value': p_value,
            'significance': sig_level
        }


def main():
    """Main execution function"""
    import sys

    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path(__file__).parent.parent.parent / "files" / "nasa-tlx"

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return

    print(f"Analyzing NASA-TLX data from: {data_dir}")

    analyzer = NASATLXAnalysis(data_dir)
    results = analyzer.run_full_analysis()

    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nFriedman Test: p = {results['p_value']:.4f} {results['significance']}")
        print(f"\nMean NASA-TLX Scores by Distance:")
        print(results['descriptive'][['distance', 'mean', 'median']].to_string(index=False))


if __name__ == "__main__":
    main()
