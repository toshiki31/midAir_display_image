#!/usr/bin/env python3
"""
Communication Count Statistical Analysis

Analyzes communication counts for each condition (A vs B).
- Experimental Design: Paired comparison (condition A vs condition B)
- N = 13 participants (within-subject)
- DVs: question_count, nod_count
- Analysis: Normality test → Paired t-test/Wilcoxon test

Author: Generated for midAir_display_image project
Date: 2026-02-09
"""

from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "Yu Gothic", "Meirio", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class CommunicationCountAnalyzer:
    """
    Statistical analysis for communication counts

    Experimental Design:
    - Within-subject comparison: condition A vs condition B
    - N = 13 participants
    - DVs: question_count, nod_count
    """

    def __init__(self, csv_path, output_dir=None):
        """
        Initialize the analyzer

        Args:
            csv_path: Path to communication_count.csv
            output_dir: Directory to save output files (default: same as csv)
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir) if output_dir else self.csv_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.df = self._load_data()
        self.n_participants = len(self.df)

        print("=" * 80)
        print("COMMUNICATION COUNT STATISTICAL ANALYSIS")
        print("=" * 80)
        print(f"\nData loaded from: {self.csv_path}")
        print(f"Number of participants: {self.n_participants}")
        print(f"Output directory: {self.output_dir}")

        self.results = {}

    def _load_data(self):
        """Load and validate the CSV data"""
        df = pd.read_csv(self.csv_path)

        expected_cols = [
            "名前",
            "question_count_A",
            "question_count_B",
            "nod_count_A",
            "nod_count_B",
        ]
        if list(df.columns) != expected_cols:
            raise ValueError(f"CSV columns must be: {expected_cols}")

        df = df.rename(columns={"名前": "participant"})

        # Validate data
        for col in expected_cols[1:]:
            if df[col].isna().any():
                raise ValueError(f"Missing values in {col}")

        return df

    def display_data(self):
        """Display the loaded data"""
        print("\n" + "=" * 80)
        print("RAW DATA")
        print("=" * 80)
        print(self.df.to_string(index=False))

    def descriptive_statistics(self, measure_key):
        """Calculate descriptive statistics for a measure"""
        col_a = f"{measure_key}_A"
        col_b = f"{measure_key}_B"

        stats_a = {
            "Mean": self.df[col_a].mean(),
            "SD": self.df[col_a].std(ddof=1),
            "Median": self.df[col_a].median(),
            "Min": self.df[col_a].min(),
            "Max": self.df[col_a].max(),
            "Q1": self.df[col_a].quantile(0.25),
            "Q3": self.df[col_a].quantile(0.75),
        }

        stats_b = {
            "Mean": self.df[col_b].mean(),
            "SD": self.df[col_b].std(ddof=1),
            "Median": self.df[col_b].median(),
            "Min": self.df[col_b].min(),
            "Max": self.df[col_b].max(),
            "Q1": self.df[col_b].quantile(0.25),
            "Q3": self.df[col_b].quantile(0.75),
        }

        df_stats = pd.DataFrame({"Condition A": stats_a, "Condition B": stats_b}).T
        return df_stats

    def test_normality(self, measure_key):
        """
        Test normality using Shapiro-Wilk test for each condition

        Returns:
            dict: Normality test results
        """
        col_a = f"{measure_key}_A"
        col_b = f"{measure_key}_B"

        w_a, p_a = stats.shapiro(self.df[col_a])
        w_b, p_b = stats.shapiro(self.df[col_b])

        return {
            "W_A": w_a,
            "p_A": p_a,
            "W_B": w_b,
            "p_B": p_b,
            "all_normal": (p_a > 0.05) and (p_b > 0.05),
        }

    def compute_effect_size(self, data_a, data_b, parametric=True):
        """
        Compute effect size

        Args:
            data_a: Data for condition A
            data_b: Data for condition B
            parametric: If True, use Cohen's d; otherwise rank-biserial
        """
        if parametric:
            diff = data_a - data_b
            d = np.mean(diff) / np.std(diff, ddof=1)

            abs_d = abs(d)
            if abs_d < 0.2:
                interpretation = "negligible"
            elif abs_d < 0.5:
                interpretation = "small"
            elif abs_d < 0.8:
                interpretation = "medium"
            else:
                interpretation = "large"

            return {"measure": "Cohen's d", "value": d, "interpretation": interpretation}

        w_stat, _ = stats.wilcoxon(data_a, data_b)
        n = len(data_a)

        mu_w = n * (n + 1) / 4
        sigma_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (w_stat - mu_w) / sigma_w
        r = z / np.sqrt(n)

        abs_r = abs(r)
        if abs_r < 0.1:
            interpretation = "negligible"
        elif abs_r < 0.3:
            interpretation = "small"
        elif abs_r < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"

        return {
            "measure": "Rank-biserial correlation (r)",
            "value": r,
            "interpretation": interpretation,
        }

    def run_paired_comparison(self, measure_key, parametric=True):
        """Run paired comparison between condition A and B"""
        col_a = f"{measure_key}_A"
        col_b = f"{measure_key}_B"

        condition_a = self.df[col_a].values
        condition_b = self.df[col_b].values

        results = {
            "test_type": "paired_t_test" if parametric else "wilcoxon",
            "parametric": parametric,
            "n": len(condition_a),
            "mean_a": np.mean(condition_a),
            "std_a": np.std(condition_a, ddof=1),
            "mean_b": np.mean(condition_b),
            "std_b": np.std(condition_b, ddof=1),
            "median_a": np.median(condition_a),
            "median_b": np.median(condition_b),
        }

        if parametric:
            t_stat, p_value = stats.ttest_rel(condition_a, condition_b)
            effect_size_result = self.compute_effect_size(condition_a, condition_b, parametric=True)
            results.update(
                {
                    "statistic_name": "t",
                    "statistic": t_stat,
                    "p_value": p_value,
                    "df": len(condition_a) - 1,
                    "effect_size": effect_size_result,
                }
            )
        else:
            w_stat, p_value = stats.wilcoxon(condition_a, condition_b)
            effect_size_result = self.compute_effect_size(condition_a, condition_b, parametric=False)
            results.update(
                {
                    "statistic_name": "W",
                    "statistic": w_stat,
                    "p_value": p_value,
                    "effect_size": effect_size_result,
                }
            )

        return results

    def analyze_measure(self, measure_key, measure_label):
        """Run full analysis for a single measure"""
        print("\n" + "=" * 80)
        print(f"ANALYZING: {measure_label}")
        print("=" * 80)

        desc = self.descriptive_statistics(measure_key)
        print("\nDESCRIPTIVE STATISTICS")
        print(desc.to_string())

        norm = self.test_normality(measure_key)
        print("\nNORMALITY TESTING (Shapiro-Wilk)")
        print("H0: Data is normally distributed")
        print(f"Condition A: W = {norm['W_A']:.4f}, p = {norm['p_A']:.4f}")
        print(f"Condition B: W = {norm['W_B']:.4f}, p = {norm['p_B']:.4f}")

        parametric = norm["all_normal"]
        if parametric:
            print("DECISION: Both conditions pass normality → Paired t-test")
        else:
            print("DECISION: Normality violated → Wilcoxon signed-rank test")

        comp = self.run_paired_comparison(measure_key, parametric=parametric)
        effect = comp["effect_size"]

        print("\nPAIRED COMPARISON RESULTS")
        if parametric:
            print(
                f"t({comp['df']}) = {comp['statistic']:.4f}, p = {comp['p_value']:.4f}"
            )
        else:
            print(f"W = {comp['statistic']:.4f}, p = {comp['p_value']:.4f}")

        print(
            f"Condition A: M = {comp['mean_a']:.3f}, SD = {comp['std_a']:.3f}, "
            f"Median = {comp['median_a']:.3f}"
        )
        print(
            f"Condition B: M = {comp['mean_b']:.3f}, SD = {comp['std_b']:.3f}, "
            f"Median = {comp['median_b']:.3f}"
        )
        print(
            f"Effect size: {effect['measure']} = {effect['value']:.4f} ({effect['interpretation']})"
        )

        self.results[measure_key] = {
            "label": measure_label,
            "descriptive": desc,
            "normality": norm,
            "comparison": comp,
        }

    def plot_boxplots(self):
        """Create separate box plots for question and nod counts"""
        print("\n  ✓ Creating box plots...")

        df_long = pd.melt(
            self.df,
            id_vars=["participant"],
            value_vars=[
                "question_count_A",
                "question_count_B",
                "nod_count_A",
                "nod_count_B",
            ],
            var_name="measure",
            value_name="count",
        )

        df_long["condition"] = df_long["measure"].apply(lambda x: "A" if x.endswith("_A") else "B")
        df_long["metric"] = df_long["measure"].apply(
            lambda x: "質問・確認回数" if x.startswith("question") else "頷き回数"
        )

        metrics = [
            ("質問・確認回数", "communication_question_count_boxplot.png"),
            ("頷き回数", "communication_nod_count_boxplot.png"),
        ]

        for metric_label, filename in metrics:
            subset = df_long[df_long["metric"] == metric_label]
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            sns.boxplot(
                data=subset,
                x="condition",
                y="count",
                ax=ax,
                palette="Set2",
            )
            sns.stripplot(
                data=subset,
                x="condition",
                y="count",
                ax=ax,
                color="black",
                alpha=0.5,
                jitter=True,
                size=8,
            )

            ax.set_title(f"{metric_label}\n(条件A vs 条件B)", fontsize=12, fontweight="bold")
            ax.set_xlabel("条件", fontsize=11)
            ax.set_ylabel("回数", fontsize=11)
            ax.set_xticklabels(
                ["スマートフォン\n(条件A)", "椅子型空中像インタフェース\n(条件B)"], fontsize=11
            )
            ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
            print(f"    Saved: {filename}")
            plt.close()

    def save_analysis_report_md(self):
        """Generate Markdown analysis report"""
        print("\n" + "=" * 80)
        print("GENERATING MARKDOWN REPORT")
        print("=" * 80)

        report_path = self.output_dir / "communication_count_analysis_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# コミュニケーション回数の分析\n\n")
            f.write("## 実験概要\n\n")
            f.write("- **デザイン**: 条件Aと条件Bの対応のある比較（被験者内）\n")
            f.write(f"- **参加者数**: {self.n_participants}\n")
            f.write("- **指標**:\n")
            f.write("  - 質問・確認回数\n")
            f.write("  - 頷き回数\n")
            f.write(f"- **分析日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for measure_key, res in self.results.items():
                label_map = {
                    "Question Count": "質問・確認回数",
                    "Nod Count": "頷き回数",
                }
                section_title = label_map.get(res["label"], res["label"])
                f.write(f"## {section_title}\n\n")

                f.write("### 記述統計\n\n")
                f.write("| 条件 | 平均 | SD | 中央値 | 最小 | 最大 |\n")
                f.write("|------|------|----|--------|------|------|\n")
                desc = res["descriptive"]
                for cond in ["Condition A", "Condition B"]:
                    row = desc.loc[cond]
                    f.write(
                        f"| {cond[-1]} | {row['Mean']:.2f} | {row['SD']:.2f} | "
                        f"{row['Median']:.2f} | {row['Min']:.2f} | {row['Max']:.2f} |\n"
                    )
                f.write("\n")

                f.write("### 正規性の検定（Shapiro-Wilk）\n\n")
                norm = res["normality"]
                f.write("| 条件 | W | p値 | 正規性 |\n")
                f.write("|------|---|-----|--------|\n")
                f.write(
                    f"| A | {norm['W_A']:.4f} | {norm['p_A']:.4f} | "
                    f"{'はい' if norm['p_A'] > 0.05 else 'いいえ'} |\n"
                )
                f.write(
                    f"| B | {norm['W_B']:.4f} | {norm['p_B']:.4f} | "
                    f"{'はい' if norm['p_B'] > 0.05 else 'いいえ'} |\n"
                )
                f.write("\n")

                comp = res["comparison"]
                f.write("### 対応のある比較\n\n")
                test_name = "対応のあるt検定" if comp["parametric"] else "Wilcoxon符号順位検定"
                f.write(f"- **検定手法**: {test_name}\n")
                f.write(
                    f"- **条件A**: M = {comp['mean_a']:.2f}, SD = {comp['std_a']:.2f}\n"
                )
                f.write(
                    f"- **条件B**: M = {comp['mean_b']:.2f}, SD = {comp['std_b']:.2f}\n"
                )
                f.write(
                    f"- **統計量**: {comp['statistic_name']} = {comp['statistic']:.4f}, "
                    f"p = {comp['p_value']:.4f}\n"
                )
                effect = comp["effect_size"]
                f.write(
                    f"- **効果量**: {effect['measure']} = {effect['value']:.3f} "
                    f"({effect['interpretation']})\n"
                )
                significant = comp["p_value"] < 0.05
                f.write(f"- **有意差**: {'あり' if significant else 'なし'}\n")

                diff = comp["mean_b"] - comp["mean_a"]
                direction = "条件Bが多い" if diff > 0 else "条件Aが多い"
                f.write(f"- **方向**: {direction}（B - A = {diff:.2f}）\n\n")

                f.write("### 考察\n\n")
                if significant:
                    f.write(
                        "本指標では条件間に有意差が認められた。平均差の方向から、"
                        f"{direction}傾向が示唆される。\n\n"
                    )
                else:
                    f.write(
                        "本指標では条件間に有意差は認められなかった。平均差は存在するが、"
                        "個人差やサンプル数の影響で統計的に有意とは言えない可能性がある。\n\n"
                    )

        print(f"  ✓ Saved report: {report_path.name}")

    def run_full_analysis(self):
        """Run full analysis pipeline"""
        print("\n" + "=" * 80)
        print("RUNNING FULL ANALYSIS PIPELINE")
        print("=" * 80)

        self.display_data()

        self.analyze_measure("question_count", "Question Count")
        self.analyze_measure("nod_count", "Nod Count")

        self.plot_boxplots()
        self.save_analysis_report_md()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nAll outputs saved to: {self.output_dir}")

        return self.results


def main():
    """Main function to run the analysis"""
    csv_path = Path(__file__).parent.parent.parent / "output" / "communication_count.csv"
    output_dir = Path(__file__).parent.parent.parent / "output"

    analyzer = CommunicationCountAnalyzer(csv_path, output_dir)
    results = analyzer.run_full_analysis()
    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()
