#!/usr/bin/env python3
"""
NASA-TLX Repeated Measures ANOVA Analysis

This script performs parametric repeated measures ANOVA analysis on NASA-TLX data
with 5 distance conditions (a, b, c, d, e → 0cm, 15cm, 30cm, 45cm, 60cm).

Use this script when normality assumptions are met (verified by Shapiro-Wilk tests).

Author: Generated for comic-effect project
Date: 2025-11-02
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
import sys

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Japanese font support
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio']
plt.rcParams['axes.unicode_minus'] = False


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


class NASATLXAnova:
    """Class to perform repeated measures ANOVA analysis on NASA-TLX data"""

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
                        'total_score': total_score
                    })

                    print(f"  Loaded: {participant}_{condition} → Score: {total_score}")

                else:
                    print(f"Warning: {csv_file.name} has insufficient rows")

            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
                continue

        print("-" * 80)
        print(f"Successfully loaded {len(data_list)} records")

        self.df_raw = pd.DataFrame(data_list)

        # Ensure distance is categorical with correct order
        self.df_raw['distance'] = pd.Categorical(
            self.df_raw['distance'],
            categories=self.distance_order,
            ordered=True
        )

        # Sort by participant and distance
        self.df_raw = self.df_raw.sort_values(['participant', 'distance'])

        print(f"Participants: {sorted(self.df_raw['participant'].unique())}")
        print(f"Conditions: {sorted(self.df_raw['condition'].unique())}")

        print("\nData summary:")
        print(self.df_raw.pivot(index='participant', columns='distance', values='total_score'))

    def descriptive_statistics(self):
        """Calculate and display descriptive statistics"""
        print("\n" + "=" * 80)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 80)

        print("\nNASA-TLX Total Score by Distance Condition:")
        print("(Score range: 0-100, lower is better)")

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

        print(desc.to_string(index=False))

        print("\nBy Participant:")
        desc_participant = self.df_raw.groupby('participant')['total_score'].agg([
            'count',
            ('median', 'median'),
            ('mean', 'mean'),
            'min',
            'max'
        ])
        print(desc_participant.to_string())

        return desc

    def test_normality(self):
        """Test normality using Shapiro-Wilk test for each distance condition"""
        print("\n" + "=" * 80)
        print("NORMALITY TESTS (Shapiro-Wilk)")
        print("=" * 80)
        print("H0: Data is normally distributed")
        print("If p < 0.05, reject H0 (data is NOT normally distributed)")

        normality_results = []

        for distance in self.distance_order:
            data = self.df_raw[self.df_raw['distance'] == distance]['total_score']
            statistic, p_value = stats.shapiro(data)

            result = {
                'distance': distance,
                'n': len(data),
                'W': statistic,
                'p_value': p_value,
                'normal': 'Yes' if p_value > 0.05 else 'No'
            }
            normality_results.append(result)

            print(f"\n{distance}: W = {statistic:.4f}, p = {p_value:.4f} {'(Normal)' if p_value > 0.05 else '(NOT Normal)'}")

        df_normality = pd.DataFrame(normality_results)
        print("\n" + "-" * 80)
        print("Summary:")
        print(df_normality.to_string(index=False))

        all_normal = all(df_normality['normal'] == 'Yes')
        if all_normal:
            print("\n✓ All conditions meet normality assumption")
            print("  → Parametric tests (ANOVA) can be used")
        else:
            print("\n✗ Some conditions violate normality assumption")
            print("  → Consider using non-parametric tests (Friedman test)")

        return df_normality

    def test_sphericity_and_anova(self):
        """Perform repeated measures ANOVA with sphericity test"""
        print("\n" + "=" * 80)
        print("REPEATED MEASURES ANOVA (PARAMETRIC)")
        print("=" * 80)

        print("\nPerforming repeated measures ANOVA...")
        print("Within-subject factor: distance (5 levels)")
        print("Dependent variable: NASA-TLX Total Score")
        print("Error term: participant (random effect)")

        # Perform repeated measures ANOVA
        anova_result = pg.rm_anova(
            data=self.df_raw,
            dv='total_score',
            within='distance',
            subject='participant',
            detailed=True
        )

        print("\n" + "-" * 80)
        print("ANOVA Results:")
        print(anova_result.to_string(index=False))

        # Extract key statistics
        f_value = anova_result['F'].values[0]
        p_value = anova_result['p-unc'].values[0]

        # Check for sphericity columns
        if 'sphericity' in anova_result.columns:
            sphericity = anova_result['sphericity'].values[0]
            print(f"\nSphericity assumption met: {sphericity}")

            if not sphericity and 'p-GG-corr' in anova_result.columns:
                p_gg = anova_result['p-GG-corr'].values[0]
                epsilon_gg = anova_result['eps'].values[0] if 'eps' in anova_result.columns else None

                print("\n✗ Sphericity violated - Using Greenhouse-Geisser correction")
                if epsilon_gg:
                    print(f"  Epsilon: {epsilon_gg:.4f}")
                print(f"  GG-corrected p-value: {p_gg:.4f}")
                p_value = p_gg  # Use corrected p-value

        print("\n" + "-" * 80)
        print("INTERPRETATION:")

        # Get degrees of freedom
        df_effect = anova_result.loc[anova_result['Source'] == 'distance', 'DF'].values[0]
        df_error = anova_result.loc[anova_result['Source'] == 'Error', 'DF'].values[0]

        print(f"F({df_effect:.0f}, {df_error:.0f}) = {f_value:.4f}, p = {p_value:.4f}")

        if p_value < 0.001:
            print("*** Highly significant effect (p < 0.001)")
            print("    → Distance has a strong effect on NASA-TLX workload")
        elif p_value < 0.01:
            print("**  Very significant effect (p < 0.01)")
            print("    → Distance significantly affects workload")
        elif p_value < 0.05:
            print("*   Significant effect (p < 0.05)")
            print("    → Distance affects workload")
        else:
            print("    No significant effect (p >= 0.05)")
            print("    → Distance does not significantly affect workload")

        return anova_result, p_value

    def posthoc_pairwise(self):
        """Perform post-hoc pairwise comparisons with multiple comparison correction"""
        print("\n" + "=" * 80)
        print("POST-HOC PAIRWISE COMPARISONS")
        print("=" * 80)
        print("\nPaired t-tests for all distance comparisons")
        print("Multiple comparison correction: Holm-Bonferroni method")

        # Perform pairwise t-tests with correction
        posthoc = pg.pairwise_tests(
            data=self.df_raw,
            dv='total_score',
            within='distance',
            subject='participant',
            parametric=True,  # Use paired t-test
            padjust='holm',  # Holm-Bonferroni correction
            effsize='hedges'  # Effect size (Hedges' g for paired samples)
        )

        print("\n" + "-" * 80)
        print("Pairwise comparison results:")

        # Select relevant columns for display
        display_cols = ['A', 'B', 'T', 'dof', 'p-unc', 'p-corr', 'hedges']
        posthoc_display = posthoc[display_cols].copy()
        posthoc_display.columns = ['Distance_1', 'Distance_2', 't-statistic', 'df',
                                    'p-uncorrected', 'p-corrected', 'Effect_size']

        print(posthoc_display.to_string(index=False))

        # Highlight significant comparisons
        print("\n" + "-" * 80)
        print("SIGNIFICANT COMPARISONS (p-corrected < 0.05):")

        sig_comparisons = posthoc_display[posthoc_display['p-corrected'] < 0.05]

        if len(sig_comparisons) > 0:
            for idx, row in sig_comparisons.iterrows():
                direction = ">" if row['t-statistic'] > 0 else "<"
                print(f"\n  {row['Distance_1']} {direction} {row['Distance_2']}")
                print(f"    t({row['df']:.0f}) = {row['t-statistic']:.3f}, p = {row['p-corrected']:.4f}")
                print(f"    Effect size (Hedges' g) = {row['Effect_size']:.3f}")
        else:
            print("\n  No significant pairwise differences found after correction")

        return posthoc

    def visualize_data(self, output_dir, posthoc=None):
        """Create visualizations for the data"""
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Box plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Box plot
        sns.boxplot(data=self.df_raw, x='distance', y='total_score', ax=ax,
                   palette='Set2', order=self.distance_order)
        ax.set_title('NASA-TLX',
                         fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance Condition', fontsize=12)
        ax.set_ylabel('NASA-TLX Total Score (0-100)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add significance brackets if post-hoc results provided
        if posthoc is not None:
            y_max = self.df_raw.groupby('distance')['total_score'].max().max()
            x_positions = {dist: i for i, dist in enumerate(self.distance_order)}
            add_significance_brackets(ax, posthoc, x_positions, y_max,
                                     self.distance_order, height_increment=5.0)

        plt.tight_layout()
        filename = "boxplot_anova.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {filename}")
        plt.close()

        # 2. Bar plot with error bars (mean ± SEM)
        fig, ax = plt.subplots(figsize=(10, 6))

        summary = self.df_raw.groupby('distance')['total_score'].agg(['mean', 'sem']).reset_index()

        bars = ax.bar(summary['distance'], summary['mean'],
                     yerr=summary['sem'], capsize=8,
                     alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1.5)

        ax.set_title('Mean NASA-TLX Score by Distance\n(Lower = better)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance Condition', fontsize=12)
        ax.set_ylabel('NASA-TLX Total Score (Mean ± SEM)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        filename = "barplot_mean_anova.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

        # 3. Individual participant trajectories
        fig, ax = plt.subplots(figsize=(12, 8))

        for participant in sorted(self.df_raw['participant'].unique()):
            participant_data = self.df_raw[self.df_raw['participant'] == participant]
            ax.plot(participant_data['distance'], participant_data['total_score'],
                   marker='o', label=participant, linewidth=2, markersize=8)

        ax.set_title('Individual Participant Trajectories - NASA-TLX',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance Condition', fontsize=12)
        ax.set_ylabel('NASA-TLX Total Score', fontsize=12)
        ax.legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = "trajectories_anova.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

        # 4. Q-Q plots for normality assessment
        n_distances = len(self.distance_order)
        n_cols = 3
        n_rows = int(np.ceil(n_distances / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_distances > 1 else [axes]

        for idx, distance in enumerate(self.distance_order):
            data = self.df_raw[self.df_raw['distance'] == distance]['total_score']
            stats.probplot(data, dist="norm", plot=axes[idx])
            axes[idx].set_title(f'Q-Q Plot: {distance}', fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_distances, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        filename = "qq_plots_anova.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

        print(f"\n✓ All plots saved to: {output_dir}")

    def save_results(self, results_dict, output_path):
        """Save statistical results to Excel file"""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in results_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"\n✓ Results saved to: {output_path}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 80)
        print("NASA-TLX REPEATED MEASURES ANOVA ANALYSIS")
        print("=" * 80)
        print("\nWithin-subject design:")
        print("  Factor: Distance (5 levels: 0cm, 15cm, 30cm, 45cm, 60cm)")
        print("  Dependent variable: NASA-TLX Total Score (0-100, lower is better)")
        print("  Method: Repeated Measures ANOVA (parametric)")
        print("\nNote: Use this analysis when normality assumption is met")

        # Load data
        self.load_data()

        # Descriptive statistics
        desc = self.descriptive_statistics()

        # Test normality
        normality = self.test_normality()

        # Repeated measures ANOVA with sphericity test
        anova_result, p_value = self.test_sphericity_and_anova()

        # Post-hoc tests if significant
        posthoc = None
        if p_value < 0.05:
            posthoc = self.posthoc_pairwise()
        else:
            print("\n" + "=" * 80)
            print("POST-HOC TESTS SKIPPED")
            print("=" * 80)
            print("ANOVA was not significant (p >= 0.05)")
            print("No need for pairwise comparisons")

        # Visualizations
        output_dir = self.data_dir.parent / "nasatlx_results" / "anova"
        self.visualize_data(output_dir, posthoc=posthoc)

        # Save all results
        output_path = self.data_dir.parent / "nasatlx_results" / "anova" / "anova_results.xlsx"

        results_to_save = {
            'Descriptive_Stats': desc,
            'Normality_Tests': normality,
            'ANOVA_Results': anova_result,
        }

        # Add posthoc results if they exist
        if posthoc is not None:
            results_to_save['PostHoc_Tests'] = posthoc

        self.save_results(results_to_save, output_path)

        print("\n\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\n✓ Results directory: {self.data_dir.parent / 'nasatlx_results' / 'anova'}")
        print("✓ Statistical results: anova_results.xlsx")
        print("✓ Visualizations: *.png files")

        return {
            'descriptive': desc,
            'normality': normality,
            'anova': anova_result,
            'posthoc': posthoc
        }


def main():
    """Main execution function"""
    # Parse command-line arguments
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        # Default path
        data_dir = Path(__file__).parent.parent.parent / "files" / "nasa-tlx"

    if not data_dir.exists():
        print(f"Error: Could not find {data_dir}")
        print("Please ensure the NASA-TLX data directory exists")
        print("\nUsage:")
        print("  python nasatlx_anova.py [path/to/nasa-tlx-dir]")
        return

    print(f"Analyzing NASA-TLX data from: {data_dir}")

    # Create analyzer and run full analysis
    analyzer = NASATLXAnova(data_dir)
    results = analyzer.run_full_analysis()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nANOVA p-value: {results['anova']['p-unc'].values[0]:.4f}")

    # Check sphericity
    if 'sphericity' in results['anova'].columns:
        sphericity = results['anova']['sphericity'].values[0]
        print(f"Sphericity met: {sphericity}")
        if not sphericity and 'p-GG-corr' in results['anova'].columns:
            print(f"GG-corrected p-value: {results['anova']['p-GG-corr'].values[0]:.4f}")


if __name__ == "__main__":
    main()
