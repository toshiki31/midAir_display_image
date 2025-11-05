#!/usr/bin/env python3
"""
Performance Visualization - All Participants (Including Outliers)

This script visualizes task time data for ALL participants without excluding outliers.
Generates bar plot showing mean task times across 5 distance conditions.

Author: Generated for comic-effect project
Date: 2025-11-04
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class PerformanceVisualizerAllParticipants:
    """Visualize performance data for all participants including outliers"""

    def __init__(self, csv_path, output_dir):
        """
        Initialize with paths

        Args:
            csv_path: Path to performance.csv file
            output_dir: Directory to save output graphs
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.df_raw = None
        self.df_averaged = None

    def load_data(self):
        """Load data WITHOUT excluding any participants"""
        print("=" * 80)
        print("LOADING DATA - ALL PARTICIPANTS (NO OUTLIER EXCLUSION)")
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

    def calculate_averaged_trials(self):
        """Calculate mean of 3 trials for each participant x distance"""
        print("\n" + "=" * 80)
        print("CALCULATING AVERAGED TRIALS")
        print("=" * 80)

        # Calculate mean of 3 trials for each participant x distance
        self.df_averaged = self.df_raw.copy()
        self.df_averaged['mean_performance'] = self.df_averaged[['no1', 'no2', 'no3']].mean(axis=1)

        print("\nAveraged data (3 trials → mean):")
        print(self.df_averaged[['name', 'task', 'no1', 'no2', 'no3', 'mean_performance']].head(10))

        # Create summary statistics by distance
        summary = self.df_averaged.groupby('task')['mean_performance'].agg(['count', 'mean', 'std', 'sem'])
        print("\nSummary by distance condition:")
        print(summary)

    def create_barplot(self):
        """Create bar plot with mean ± SEM"""
        print("\n" + "=" * 80)
        print("GENERATING BAR PLOT")
        print("=" * 80)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Map task codes to distance labels
        condition_labels = {
            'a': '0cm',
            'b': '15cm',
            'c': '30cm',
            'd': '45cm',
            'e': '60cm'
        }
        distance_order = ['0cm', '15cm', '30cm', '45cm', '60cm']

        # Create a copy with mapped labels
        df_plot = self.df_averaged.copy()
        df_plot['distance'] = df_plot['task'].map(condition_labels)

        # Calculate summary statistics
        summary = df_plot.groupby('distance')['mean_performance'].agg(['mean', 'sem']).reset_index()

        # Ensure correct order
        summary['distance'] = pd.Categorical(summary['distance'],
                                            categories=distance_order, ordered=True)
        summary = summary.sort_values('distance')

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(summary['distance'], summary['mean'],
                     yerr=summary['sem'], capsize=8,
                     alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1.5)

        ax.set_title('Mean Task Time by Distance - All Participants (Including Outliers)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance Condition', fontsize=12)
        ax.set_ylabel('Task Time [seconds] (Mean ± SEM)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)

        # Add note about including all participants
        ax.text(0.02, 0.98, 'Note: All 9 participants included (no outlier exclusion)',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        # Save the plot
        filename = "barplot_all_participants_averaged.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Bar plot saved to: {output_path}")
        plt.close()

        # Print summary statistics
        print("\nSummary Statistics by Distance:")
        print(summary)

        return output_path

    def run(self):
        """Run the complete visualization workflow"""
        print("\n" + "🎨" * 40)
        print("PERFORMANCE VISUALIZATION - ALL PARTICIPANTS")
        print("🎨" * 40 + "\n")

        # Step 1: Load data (no exclusions)
        self.load_data()

        # Step 2: Calculate averaged trials
        self.calculate_averaged_trials()

        # Step 3: Create bar plot
        output_path = self.create_barplot()

        print("\n" + "=" * 80)
        print("✅ VISUALIZATION COMPLETE!")
        print("=" * 80)
        print(f"\nOutput saved to: {output_path}")


def main():
    """Main execution function"""
    # Define paths
    csv_path = Path(__file__).parent.parent.parent / "files" / "performance.csv"
    output_dir = Path(__file__).parent.parent.parent / "files" / "anova_results"

    # Create visualizer and run
    visualizer = PerformanceVisualizerAllParticipants(csv_path, output_dir)
    visualizer.run()


if __name__ == "__main__":
    main()
