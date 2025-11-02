#!/usr/bin/env python3
"""
Subjective Evaluation Analysis with Likert Scale Data

This script performs statistical analysis on subjective evaluation data
using 7-point Likert scales across 5 distance conditions (a, b, c, d, e).

Evaluation items:
1. Readability of speech bubbles (吹き出しの読みやすさ)
2. Eye strain - blepharospasm (目のしょぼつき)
3. Eye fatigue (目の疲れ)
4. Eye pain (目の痛み)
5. Eye dryness (目の乾き)
6. Blurred vision (ものがぼやけて見える)

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

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Support Japanese fonts
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic']
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
                ha='center', va='bottom', fontsize=10, fontweight='bold')


class SubjectiveAnalysis:
    """Class to analyze subjective evaluation data with Likert scales"""

    def __init__(self, csv_path):
        """
        Initialize with path to subjective CSV file

        Args:
            csv_path: Path to subjective.csv file
        """
        self.csv_path = csv_path
        self.df_raw = None
        self.df_clean = None
        self.df_long = None

        # Evaluation items (Japanese to English mapping)
        self.eval_items = {
            '吹き出しの文字は読みやすかったですか？': 'Readability',
            '目はしょぼつきましたか？': 'Eye_strain',
            '目が疲れましたか': 'Eye_fatigue',
            '目はいたいですか？': 'Eye_pain',
            '目はかわきましたか？': 'Eye_dryness',
            'ものがぼやけて見えますか？': 'Blurred_vision'
        }

        # English to Japanese (for visualization)
        self.eval_items_jp = {v: k.replace('？', '').replace('ですか', '')
                             for k, v in self.eval_items.items()}

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
        """Load and clean the data"""
        print("=" * 80)
        print("LOADING SUBJECTIVE EVALUATION DATA")
        print("=" * 80)

        # Read CSV
        self.df_raw = pd.read_csv(self.csv_path)

        # Exclude taninaka (outlier) - filter by name column '氏名'
        if '氏名' in self.df_raw.columns:
            original_shape = self.df_raw.shape
            self.df_raw = self.df_raw[~self.df_raw['氏名'].str.contains('taninaka|谷中', case=False, na=False)]
            excluded_count = original_shape[0] - self.df_raw.shape[0]
            print(f"\nRaw data shape: {self.df_raw.shape} (excluded {excluded_count} rows for taninaka/谷中)")
        else:
            print(f"\nRaw data shape: {self.df_raw.shape}")

        # Extract relevant columns
        # Column 3: condition, Columns 4-9: evaluation items
        relevant_cols = ['実験タイプを教えてください'] + list(self.eval_items.keys())

        # Check if all columns exist
        missing_cols = [col for col in relevant_cols if col not in self.df_raw.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            return False

        self.df_clean = self.df_raw[relevant_cols].copy()

        # Rename columns
        col_rename = {'実験タイプを教えてください': 'condition'}
        col_rename.update(self.eval_items)
        self.df_clean.columns = [col_rename.get(col, col) for col in self.df_clean.columns]

        # Map condition labels
        self.df_clean['distance'] = self.df_clean['condition'].map(self.condition_labels)

        # Identify participants (by order of appearance, as there's no explicit ID)
        # Group by unique combinations to assign participant IDs
        self.df_clean['participant_order'] = range(len(self.df_clean))

        # Assign participant based on groups of 5 (each participant does all 5 conditions)
        self.df_clean['participant'] = 'P' + (self.df_clean['participant_order'] // 5 + 1).astype(str)

        print(f"\nCleaned data shape: {self.df_clean.shape}")
        print(f"Number of participants: {self.df_clean['participant'].nunique()}")
        print(f"Participants: {sorted(self.df_clean['participant'].unique())}")
        print(f"Conditions: {sorted(self.df_clean['distance'].unique())}")

        print("\nFirst few rows:")
        display_cols = ['participant', 'distance'] + list(self.eval_items.values())
        print(self.df_clean[display_cols].head(10))

        return True

    def create_long_format(self):
        """Convert to long format for analysis"""
        print("\n" + "=" * 80)
        print("CREATING LONG FORMAT DATA")
        print("=" * 80)

        # Melt to long format
        id_vars = ['participant', 'condition', 'distance']
        value_vars = list(self.eval_items.values())

        self.df_long = pd.melt(
            self.df_clean,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='evaluation_item',
            value_name='rating'
        )

        print(f"Long format shape: {self.df_long.shape}")
        print("\nSample:")
        print(self.df_long.head(10))

    def descriptive_statistics(self):
        """Calculate descriptive statistics for each evaluation item"""
        print("\n" + "=" * 80)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 80)

        results = {}

        for item in self.eval_items.values():
            print(f"\n{'-' * 80}")
            print(f"Evaluation Item: {item} ({self.eval_items_jp[item]})")
            print(f"{'-' * 80}")

            item_data = self.df_long[self.df_long['evaluation_item'] == item]

            desc = item_data.groupby('distance')['rating'].agg([
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

            print(desc.to_string(index=False))
            results[item] = desc

        return results

    def test_normality(self):
        """Test normality for each evaluation item and condition"""
        print("\n" + "=" * 80)
        print("NORMALITY TESTS (Shapiro-Wilk)")
        print("=" * 80)
        print("H0: Data is normally distributed")
        print("If p < 0.05, reject H0 (data is NOT normally distributed)")

        normality_results = []

        for item in self.eval_items.values():
            print(f"\n{'-' * 80}")
            print(f"{item} ({self.eval_items_jp[item]})")
            print(f"{'-' * 80}")

            item_data = self.df_long[self.df_long['evaluation_item'] == item]

            for distance in self.distance_order:
                data = item_data[item_data['distance'] == distance]['rating']

                if len(data) >= 3:
                    statistic, p_value = stats.shapiro(data)
                    normal = 'Yes' if p_value > 0.05 else 'No'

                    normality_results.append({
                        'evaluation_item': item,
                        'distance': distance,
                        'W': statistic,
                        'p_value': p_value,
                        'normal': normal
                    })

                    print(f"  {distance}: W = {statistic:.4f}, p = {p_value:.4f} {'(Normal)' if p_value > 0.05 else '(NOT Normal)'}")

        df_normality = pd.DataFrame(normality_results)

        print("\n" + "=" * 80)
        print("NORMALITY SUMMARY")
        print("=" * 80)

        # Count normal vs not normal for each item
        summary = df_normality.groupby('evaluation_item')['normal'].value_counts().unstack(fill_value=0)
        print(summary)

        # Overall assessment
        all_normal = all(df_normality['normal'] == 'Yes')
        if all_normal:
            print("\n✓ All conditions meet normality assumption")
            print("  → Parametric tests (Repeated Measures ANOVA) can be used")
        else:
            print("\n✗ Some conditions violate normality assumption")
            print("  → Non-parametric tests (Friedman test) are RECOMMENDED")

        return df_normality

    def friedman_test_per_item(self):
        """Perform Friedman test for each evaluation item"""
        print("\n" + "=" * 80)
        print("FRIEDMAN TESTS (NON-PARAMETRIC)")
        print("=" * 80)

        friedman_results = []
        posthoc_results = {}

        for item in self.eval_items.values():
            print(f"\n{'=' * 80}")
            print(f"Item: {item} ({self.eval_items_jp[item]})")
            print(f"{'=' * 80}")

            item_data = self.df_long[self.df_long['evaluation_item'] == item]

            # Friedman test
            friedman = pg.friedman(
                data=item_data,
                dv='rating',
                within='distance',
                subject='participant'
            )

            print("\nFriedman Test Results:")
            print(friedman.to_string(index=False))

            q_value = friedman['Q'].values[0]
            p_value = friedman['p-unc'].values[0]
            w_value = friedman['W'].values[0] if 'W' in friedman.columns else np.nan

            friedman_results.append({
                'evaluation_item': item,
                'Q': q_value,
                'p_value': p_value,
                'W': w_value
            })

            # Interpretation
            print(f"\nχ²(4) = {q_value:.4f}, p = {p_value:.4f}")
            if p_value < 0.001:
                print("*** Highly significant effect (p < 0.001)")
                sig_level = '***'
            elif p_value < 0.01:
                print("**  Very significant effect (p < 0.01)")
                sig_level = '**'
            elif p_value < 0.05:
                print("*   Significant effect (p < 0.05)")
                sig_level = '*'
            else:
                print("    No significant effect (p >= 0.05)")
                sig_level = 'n.s.'

            friedman_results[-1]['significance'] = sig_level

            # Post-hoc if significant
            if p_value < 0.05:
                print("\nPerforming post-hoc pairwise comparisons (Wilcoxon signed-rank tests)...")

                posthoc = pg.pairwise_tests(
                    data=item_data,
                    dv='rating',
                    within='distance',
                    subject='participant',
                    parametric=False,
                    padjust='holm',
                    effsize='CLES'
                )

                # Filter significant comparisons
                posthoc_sig = posthoc[posthoc['p-corr'] < 0.05].copy()

                if len(posthoc_sig) > 0:
                    print(f"\nSignificant pairwise differences (p < 0.05, Holm-corrected):")
                    for idx, row in posthoc_sig.iterrows():
                        print(f"  {row['A']} vs {row['B']}: p = {row['p-corr']:.4f}, CLES = {row['CLES']:.3f}")
                else:
                    print("\nNo significant pairwise differences after correction")

                posthoc_results[item] = posthoc
            else:
                print("\nPost-hoc tests skipped (not significant)")
                posthoc_results[item] = None

        df_friedman = pd.DataFrame(friedman_results)

        print("\n" + "=" * 80)
        print("FRIEDMAN TEST SUMMARY")
        print("=" * 80)
        print(df_friedman.to_string(index=False))

        # Multiple comparison correction (Bonferroni)
        print("\n" + "=" * 80)
        print("MULTIPLE COMPARISON CORRECTION (Bonferroni)")
        print("=" * 80)
        print(f"Number of tests: {len(df_friedman)}")
        print(f"Bonferroni-corrected α = 0.05 / {len(df_friedman)} = {0.05/len(df_friedman):.4f}")

        df_friedman['significant_bonferroni'] = df_friedman['p_value'] < (0.05 / len(df_friedman))
        print("\nSignificant after Bonferroni correction:")
        print(df_friedman[df_friedman['significant_bonferroni']][['evaluation_item', 'p_value']])

        return df_friedman, posthoc_results

    def visualize_boxplots(self, output_dir, posthoc_results=None):
        """Create box plots for each evaluation item

        Args:
            output_dir: Directory to save plots
            posthoc_results: Dictionary mapping item names to posthoc DataFrames (optional)
        """
        print("\n" + "=" * 80)
        print("CREATING BOX PLOTS")
        print("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subplots for all items
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, item in enumerate(self.eval_items.values()):
            item_data = self.df_long[self.df_long['evaluation_item'] == item]

            sns.boxplot(
                data=item_data,
                x='distance',
                y='rating',
                order=self.distance_order,
                palette='Set2',
                ax=axes[idx]
            )

            axes[idx].set_title(f'{item}\n({self.eval_items_jp[item]})',
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Distance', fontsize=10)
            axes[idx].set_ylabel('Rating (1-7)', fontsize=10)
            axes[idx].set_ylim(0.5, 7.5)
            axes[idx].grid(True, alpha=0.3, axis='y')

            # Add significance brackets if post-hoc results provided
            if posthoc_results is not None and item in posthoc_results and posthoc_results[item] is not None:
                y_max = item_data.groupby('distance')['rating'].max().max()
                x_positions = {dist: i for i, dist in enumerate(self.distance_order)}
                add_significance_brackets(axes[idx], posthoc_results[item], x_positions, y_max,
                                         self.distance_order, height_increment=0.4)

        plt.tight_layout()
        filename = "boxplots_all_items.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

    def visualize_heatmap(self, output_dir):
        """Create heatmap of median ratings"""
        print("\n" + "=" * 80)
        print("CREATING HEATMAP")
        print("=" * 80)

        output_dir = Path(output_dir)

        # Calculate median for each item x distance
        pivot_data = self.df_long.pivot_table(
            values='rating',
            index='evaluation_item',
            columns='distance',
            aggfunc='median'
        )

        # Reorder columns
        pivot_data = pivot_data[self.distance_order]

        # Reorder rows by item order
        pivot_data = pivot_data.reindex([item for item in self.eval_items.values()])

        # Rename index for display
        pivot_data.index = [f"{item}\n({self.eval_items_jp[item]})"
                           for item in pivot_data.index]

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',  # Red=high (bad for most items), Green=low (good)
            vmin=1,
            vmax=7,
            cbar_kws={'label': 'Median Rating'},
            linewidths=0.5,
            ax=ax
        )

        ax.set_title('Median Ratings by Distance and Evaluation Item',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Evaluation Item', fontsize=12)

        plt.tight_layout()
        filename = "heatmap_median_ratings.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

    def visualize_stacked_bar(self, output_dir):
        """Create stacked bar charts showing distribution of ratings"""
        print("\n" + "=" * 80)
        print("CREATING STACKED BAR CHARTS")
        print("=" * 80)

        output_dir = Path(output_dir)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, item in enumerate(self.eval_items.values()):
            item_data = self.df_long[self.df_long['evaluation_item'] == item]

            # Calculate percentage distribution
            crosstab = pd.crosstab(
                item_data['distance'],
                item_data['rating'],
                normalize='index'
            ) * 100

            # Ensure all ratings 1-7 are present
            for rating in range(1, 8):
                if rating not in crosstab.columns:
                    crosstab[rating] = 0

            crosstab = crosstab[[r for r in range(1, 8) if r in crosstab.columns]]

            # Reorder by distance
            crosstab = crosstab.reindex(self.distance_order)

            # Plot
            crosstab.plot(
                kind='barh',
                stacked=True,
                ax=axes[idx],
                colormap='RdYlGn_r',
                legend=False
            )

            axes[idx].set_title(f'{item}\n({self.eval_items_jp[item]})',
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Percentage (%)', fontsize=10)
            axes[idx].set_ylabel('Distance', fontsize=10)
            axes[idx].invert_yaxis()

        # Add legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Rating', loc='center right',
                  bbox_to_anchor=(1.02, 0.5))

        plt.tight_layout()
        plt.subplots_adjust(right=0.95)
        filename = "stacked_bar_distribution.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

    def visualize_profile(self, output_dir):
        """Create radar chart showing profile for each distance"""
        print("\n" + "=" * 80)
        print("CREATING PROFILE CHARTS")
        print("=" * 80)

        output_dir = Path(output_dir)

        # Calculate median for each distance x item
        profile_data = self.df_long.pivot_table(
            values='rating',
            index='distance',
            columns='evaluation_item',
            aggfunc='median'
        )

        profile_data = profile_data.reindex(self.distance_order)
        profile_data = profile_data[[item for item in self.eval_items.values()]]

        # Radar chart
        categories = [self.eval_items_jp[item].replace('吹き出しの文字は', '').replace('目は', '').replace('ものが', '')
                     for item in profile_data.columns]
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set2(np.linspace(0, 1, len(self.distance_order)))

        for idx, (distance, row) in enumerate(profile_data.iterrows()):
            values = row.values.tolist()
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=distance,
                   color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 7)
        ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
        ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7'])
        ax.grid(True)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Evaluation Profile by Distance\n(Radar Chart)',
                 size=14, fontweight='bold', pad=20)

        plt.tight_layout()
        filename = "radar_profile.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

    def save_results(self, desc_stats, normality, friedman, posthoc, output_path):
        """Save all results to Excel"""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Descriptive statistics (one sheet per item)
            for item, df in desc_stats.items():
                sheet_name = f"Desc_{item[:20]}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Normality tests
            normality.to_excel(writer, sheet_name='Normality_Tests', index=False)

            # Friedman tests
            friedman.to_excel(writer, sheet_name='Friedman_Tests', index=False)

            # Post-hoc tests (one sheet per significant item)
            for item, df in posthoc.items():
                if df is not None:
                    sheet_name = f"PostHoc_{item[:15]}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"✓ Results saved to: {output_path}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 80)
        print("SUBJECTIVE EVALUATION ANALYSIS - LIKERT SCALE DATA")
        print("=" * 80)
        print("\nWithin-subject design:")
        print("  Factor: Distance (5 levels: 0cm, 15cm, 30cm, 45cm, 60cm)")
        print("  Evaluation items: 6 items (7-point Likert scale)")
        print("  Method: Friedman test (non-parametric)")

        # Load and preprocess
        if not self.load_data():
            print("Error loading data")
            return None

        self.create_long_format()

        # Descriptive statistics
        desc_stats = self.descriptive_statistics()

        # Normality tests
        normality = self.test_normality()

        # Friedman tests
        friedman, posthoc = self.friedman_test_per_item()

        # Visualizations
        output_dir = Path(self.csv_path).parent / "subjective_results" / "visualizations"
        self.visualize_boxplots(output_dir, posthoc_results=posthoc)
        self.visualize_heatmap(output_dir)
        self.visualize_stacked_bar(output_dir)
        self.visualize_profile(output_dir)

        # Save results
        output_path = Path(self.csv_path).parent / "subjective_results" / "subjective_results.xlsx"
        self.save_results(desc_stats, normality, friedman, posthoc, output_path)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\n✓ Results directory: {Path(self.csv_path).parent / 'subjective_results'}")
        print("✓ Statistical results: subjective_results.xlsx")
        print("✓ Visualizations: visualizations/")

        return {
            'descriptive': desc_stats,
            'normality': normality,
            'friedman': friedman,
            'posthoc': posthoc
        }


def main():
    """Main execution function"""
    import sys

    # Parse command-line arguments
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path(__file__).parent.parent.parent / "files" / "subjective.csv"

    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        print("Please ensure the CSV file exists")
        return

    print(f"Analyzing: {csv_path}")

    # Create analyzer and run full analysis
    analyzer = SubjectiveAnalysis(csv_path)
    results = analyzer.run_full_analysis()

    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("\nFriedman Test Results:")
        print(results['friedman'][['evaluation_item', 'p_value', 'significance']].to_string(index=False))


if __name__ == "__main__":
    main()
