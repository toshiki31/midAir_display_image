#!/usr/bin/env python3
"""
Repeated Measures ANOVA Analysis for Performance Data

This script performs within-subject one-way ANOVA analysis on performance data
with 5 distance conditions (a, b, c, d, e).

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

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class PerformanceANOVA:
    """Class to perform repeated measures ANOVA analysis on performance data"""

    def __init__(self, csv_path):
        """
        Initialize with path to performance CSV file

        Args:
            csv_path: Path to performance.csv file
        """
        self.csv_path = csv_path
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
        print(f"\nRaw data shape: {self.df_raw.shape}")
        print(f"Columns: {list(self.df_raw.columns)}")
        print(f"\nFirst few rows:")
        print(self.df_raw.head(10))

        # Basic info
        print(f"\nNumber of participants: {self.df_raw['name'].nunique()}")
        print(f"Participants: {sorted(self.df_raw['name'].unique())}")
        print(f"Number of distance conditions: {self.df_raw['task'].nunique()}")
        print(f"Distance conditions: {sorted(self.df_raw['task'].unique())}")

    def preprocess_data(self):
        """Prepare data for analysis"""
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING")
        print("=" * 80)

        # Calculate mean of 3 trials for each participant x distance
        self.df_averaged = self.df_raw.copy()
        self.df_averaged['mean_performance'] = self.df_averaged[['no1', 'no2', 'no3']].mean(axis=1)

        print("\nAveraged data (3 trials → mean):")
        print(self.df_averaged[['name', 'task', 'mean_performance']].head(10))

        # Create long format for Analysis 1 (averaged data)
        self.df_long_averaged = self.df_averaged[['name', 'task', 'mean_performance']].copy()
        self.df_long_averaged.columns = ['participant', 'distance', 'performance']

        # Create long format for Analysis 2 (all trials as separate data)
        df_trial1 = self.df_raw[['name', 'task', 'no1']].copy()
        df_trial1.columns = ['participant', 'distance', 'performance']
        df_trial1['trial'] = 1

        df_trial2 = self.df_raw[['name', 'task', 'no2']].copy()
        df_trial2.columns = ['participant', 'distance', 'performance']
        df_trial2['trial'] = 2

        df_trial3 = self.df_raw[['name', 'task', 'no3']].copy()
        df_trial3.columns = ['participant', 'distance', 'performance']
        df_trial3['trial'] = 3

        self.df_long_all = pd.concat([df_trial1, df_trial2, df_trial3], ignore_index=True)

        print(f"\nLong format (averaged): {self.df_long_averaged.shape}")
        print(f"Long format (all trials): {self.df_long_all.shape}")

    def descriptive_statistics(self, df, analysis_name):
        """Calculate and display descriptive statistics"""
        print("\n" + "=" * 80)
        print(f"DESCRIPTIVE STATISTICS - {analysis_name}")
        print("=" * 80)

        desc = df.groupby('distance')['performance'].describe()
        print("\nDescriptive statistics by distance condition:")
        print(desc)

        # Calculate statistics by participant
        print("\nDescriptive statistics by participant:")
        desc_participant = df.groupby('participant')['performance'].describe()
        print(desc_participant)

        return desc

    def test_normality(self, df, analysis_name):
        """
        Test normality using Shapiro-Wilk test for each distance condition

        Args:
            df: DataFrame in long format
            analysis_name: Name of the analysis for reporting
        """
        print("\n" + "=" * 80)
        print(f"NORMALITY TEST (Shapiro-Wilk) - {analysis_name}")
        print("=" * 80)
        print("\nH0: Data is normally distributed")
        print("If p < 0.05, reject H0 (data is NOT normally distributed)")

        normality_results = []

        for distance in sorted(df['distance'].unique()):
            data = df[df['distance'] == distance]['performance']
            statistic, p_value = stats.shapiro(data)

            result = {
                'distance': distance,
                'statistic': statistic,
                'p_value': p_value,
                'normal': 'Yes' if p_value > 0.05 else 'No'
            }
            normality_results.append(result)

            print(f"\nDistance {distance}:")
            print(f"  W = {statistic:.4f}, p = {p_value:.4f} {'(Normal)' if p_value > 0.05 else '(NOT Normal)'}")

        df_normality = pd.DataFrame(normality_results)
        print("\n" + "-" * 80)
        print("Summary:")
        print(df_normality.to_string(index=False))

        all_normal = all(df_normality['normal'] == 'Yes')
        if all_normal:
            print("\n✓ All conditions meet normality assumption")
        else:
            print("\n✗ Some conditions violate normality assumption")
            print("  Consider using non-parametric tests (Friedman test) as alternative")

        return df_normality

    def test_sphericity_and_anova(self, df, analysis_name):
        """
        Perform repeated measures ANOVA with sphericity test

        Args:
            df: DataFrame in long format
            analysis_name: Name of the analysis for reporting
        """
        print("\n" + "=" * 80)
        print(f"REPEATED MEASURES ANOVA - {analysis_name}")
        print("=" * 80)

        # Perform repeated measures ANOVA using pingouin
        # pingouin automatically performs sphericity test (Mauchly's test)
        print("\nPerforming repeated measures ANOVA...")
        print("Within-subject factor: distance (5 levels: a, b, c, d, e)")
        print("Error term: participant (random effect)")

        anova_result = pg.rm_anova(
            data=df,
            dv='performance',
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
                print(f"  Epsilon: {epsilon_gg:.4f}" if epsilon_gg else "")
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
            print("    → Distance condition has a strong effect on performance")
        elif p_value < 0.01:
            print("**  Very significant effect (p < 0.01)")
            print("    → Distance condition significantly affects performance")
        elif p_value < 0.05:
            print("*   Significant effect (p < 0.05)")
            print("    → Distance condition affects performance")
        else:
            print("    No significant effect (p >= 0.05)")
            print("    → Distance condition does not significantly affect performance")

        return anova_result, p_value

    def posthoc_pairwise(self, df, analysis_name):
        """
        Perform post-hoc pairwise comparisons with multiple comparison correction

        Args:
            df: DataFrame in long format
            analysis_name: Name of the analysis for reporting
        """
        print("\n" + "=" * 80)
        print(f"POST-HOC PAIRWISE COMPARISONS - {analysis_name}")
        print("=" * 80)
        print("\nPaired t-tests for all distance comparisons")
        print("Multiple comparison correction: Holm-Bonferroni method")

        # Perform pairwise t-tests with correction
        posthoc = pg.pairwise_tests(
            data=df,
            dv='performance',
            within='distance',
            subject='participant',
            parametric=True,
            padjust='holm',  # Holm-Bonferroni correction
            effsize='hedges'  # Effect size (Hedges' g for paired samples)
        )

        print("\n" + "-" * 80)
        print("Pairwise comparison results:")

        # Select relevant columns for display
        display_cols = ['A', 'B', 'T', 'dof', 'p-unc', 'p-corr', 'hedges']
        posthoc_display = posthoc[display_cols].copy()
        posthoc_display.columns = ['Distance_1', 'Distance_2', 't-statistic', 'df', 'p-uncorrected', 'p-corrected', 'Effect_size']

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

    def visualize_data(self, df, analysis_name, output_dir):
        """
        Create visualizations for the data

        Args:
            df: DataFrame in long format
            analysis_name: Name of the analysis for file naming
            output_dir: Directory to save plots
        """
        print("\n" + "=" * 80)
        print(f"CREATING VISUALIZATIONS - {analysis_name}")
        print("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Box plot and violin plot combined
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Box plot
        sns.boxplot(data=df, x='distance', y='performance', ax=axes[0], palette='Set2')
        axes[0].set_title(f'Box Plot - {analysis_name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Distance Condition', fontsize=12)
        axes[0].set_ylabel('Performance', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Violin plot
        sns.violinplot(data=df, x='distance', y='performance', ax=axes[1], palette='Set2')
        axes[1].set_title(f'Violin Plot - {analysis_name}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Distance Condition', fontsize=12)
        axes[1].set_ylabel('Performance', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"boxplot_violin_{analysis_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {filename}")
        plt.close()

        # 2. Individual participant trajectories
        fig, ax = plt.subplots(figsize=(12, 8))

        for participant in sorted(df['participant'].unique()):
            participant_data = df[df['participant'] == participant].sort_values('distance')
            ax.plot(participant_data['distance'], participant_data['performance'],
                   marker='o', label=participant, linewidth=2, markersize=8)

        ax.set_title(f'Individual Participant Trajectories - {analysis_name}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance Condition', fontsize=12)
        ax.set_ylabel('Performance', fontsize=12)
        ax.legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"trajectories_{analysis_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

        # 3. Q-Q plots for normality assessment
        distances = sorted(df['distance'].unique())
        n_distances = len(distances)
        n_cols = 3
        n_rows = int(np.ceil(n_distances / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_distances > 1 else [axes]

        for idx, distance in enumerate(distances):
            data = df[df['distance'] == distance]['performance']
            stats.probplot(data, dist="norm", plot=axes[idx])
            axes[idx].set_title(f'Q-Q Plot: Distance {distance}', fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_distances, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        filename = f"qq_plots_{analysis_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

        # 4. Bar plot with error bars (mean ± SEM)
        fig, ax = plt.subplots(figsize=(10, 6))

        summary = df.groupby('distance')['performance'].agg(['mean', 'sem']).reset_index()

        bars = ax.bar(summary['distance'], summary['mean'],
                     yerr=summary['sem'], capsize=8,
                     alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1.5)

        ax.set_title(f'Mean Performance by Distance - {analysis_name}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance Condition', fontsize=12)
        ax.set_ylabel('Performance (Mean ± SEM)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        filename = f"barplot_mean_{analysis_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

        print(f"\n✓ All plots saved to: {output_dir}")

    def save_results(self, results_dict, output_path):
        """
        Save statistical results to CSV file

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
        print("ANALYSIS 1: AVERAGED TRIALS (3 trials → mean per participant×condition)")
        print("=" * 80)

        # Descriptive statistics
        desc = self.descriptive_statistics(self.df_long_averaged, "Analysis 1")

        # Test normality
        normality = self.test_normality(self.df_long_averaged, "Analysis 1")

        # Repeated measures ANOVA with sphericity test
        anova_result, p_value = self.test_sphericity_and_anova(self.df_long_averaged, "Analysis 1")

        # Post-hoc tests if significant
        posthoc = None
        if p_value < 0.05:
            posthoc = self.posthoc_pairwise(self.df_long_averaged, "Analysis 1")
        else:
            print("\n" + "=" * 80)
            print("POST-HOC TESTS SKIPPED")
            print("=" * 80)
            print("ANOVA was not significant (p >= 0.05)")
            print("No need for pairwise comparisons")

        # Visualizations
        output_dir = Path(self.csv_path).parent / "anova_results" / "analysis1_averaged"
        self.visualize_data(self.df_long_averaged, "Analysis 1 Averaged", output_dir)

        return {
            'descriptive': desc,
            'normality': normality,
            'anova': anova_result,
            'posthoc': posthoc
        }

    def run_analysis_2(self):
        """Run Analysis 2: All trials as separate data"""
        print("\n\n" + "=" * 80)
        print("ANALYSIS 2: ALL TRIALS SEPARATE (3 trials × 6 participants × 5 distances)")
        print("=" * 80)

        # Descriptive statistics
        desc = self.descriptive_statistics(self.df_long_all, "Analysis 2")

        # Test normality
        normality = self.test_normality(self.df_long_all, "Analysis 2")

        # Repeated measures ANOVA with sphericity test
        anova_result, p_value = self.test_sphericity_and_anova(self.df_long_all, "Analysis 2")

        # Post-hoc tests if significant
        posthoc = None
        if p_value < 0.05:
            posthoc = self.posthoc_pairwise(self.df_long_all, "Analysis 2")
        else:
            print("\n" + "=" * 80)
            print("POST-HOC TESTS SKIPPED")
            print("=" * 80)
            print("ANOVA was not significant (p >= 0.05)")
            print("No need for pairwise comparisons")

        # Visualizations
        output_dir = Path(self.csv_path).parent / "anova_results" / "analysis2_all_trials"
        self.visualize_data(self.df_long_all, "Analysis 2 All Trials", output_dir)

        return {
            'descriptive': desc,
            'normality': normality,
            'anova': anova_result,
            'posthoc': posthoc
        }

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 80)
        print("REPEATED MEASURES ANOVA - PERFORMANCE DATA ANALYSIS")
        print("=" * 80)
        print("\nWithin-subject design:")
        print("  Factor: Distance (5 levels: a, b, c, d, e)")
        print("  Dependent variable: Performance")
        print("  Error term: Participant (random effect)")
        print("\nTwo analyses:")
        print("  1. Averaged trials: Mean of 3 trials per participant×condition")
        print("  2. All trials: Each of 3 trials treated separately")

        # Load and preprocess
        self.load_data()
        self.preprocess_data()

        # Run both analyses
        results_1 = self.run_analysis_1()
        results_2 = self.run_analysis_2()

        # Save all results
        output_path = Path(self.csv_path).parent / "anova_results" / "statistical_results.xlsx"

        results_to_save = {
            'Analysis1_Descriptive': results_1['descriptive'],
            'Analysis1_Normality': results_1['normality'],
            'Analysis1_ANOVA': results_1['anova'],
            'Analysis2_Descriptive': results_2['descriptive'],
            'Analysis2_Normality': results_2['normality'],
            'Analysis2_ANOVA': results_2['anova'],
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
        print(f"\n✓ Results directory: {Path(self.csv_path).parent / 'anova_results'}")
        print("✓ Statistical results: statistical_results.xlsx")
        print("✓ Visualizations: analysis1_averaged/ and analysis2_all_trials/")

        return results_1, results_2


def main():
    """Main execution function"""
    # Path to performance.csv
    csv_path = Path(__file__).parent.parent.parent / "files" / "performance.csv"

    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        print("Please ensure performance.csv exists in project/files/ directory")
        return

    # Create analyzer and run full analysis
    analyzer = PerformanceANOVA(csv_path)
    results_1, results_2 = analyzer.run_full_analysis()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nAnalysis 1 (Averaged Trials):")
    print(f"  ANOVA p-value: {results_1['anova']['p-unc'].values[0]:.4f}")

    print("\nAnalysis 2 (All Trials Separate):")
    print(f"  ANOVA p-value: {results_2['anova']['p-unc'].values[0]:.4f}")


if __name__ == "__main__":
    main()
