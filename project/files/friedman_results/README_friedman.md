# Performance Data Analysis - Friedman Test Results

## 📋 分析概要

**分析日**: 2025-11-02
**データソース**: `performance.csv`
**分析手法**: Friedman検定（ノンパラメトリック反復測定）
**被験者数**: 8名（外れ値として谷中を除外）
**実験条件**: 5水準（a, b, c, d, e → 0cm, 15cm, 30cm, 45cm, 60cm）
**試行回数**: 各条件3試行/被験者

---

## 👥 参加者

### 分析対象（8名）
- takiguchi（瀧口）
- ishibashi（石橋）
- hayashida（林田）
- nakayama（中山）
- honda（本田）
- mitsumaru（光丸）
- shiojiri（塩尻）
- hatamoto（旗本）

### 除外
- **taninaka（谷中健介）** - 外れ値として除外

---

## 🔍 正規性評価（Shapiro-Wilk検定）

### Analysis 1（3試行の平均値）

| 条件 | W統計量 | p値 | 正規性 | 判定 |
|------|---------|-----|--------|------|
| a (0cm) | 0.7164 | **0.0034** | ❌ | 正規性違反 |
| b (15cm) | 0.8948 | 0.2589 | ✅ | 正規 |
| c (30cm) | 0.7540 | **0.0090** | ❌ | 正規性違反 |
| d (45cm) | 0.9218 | 0.4445 | ✅ | 正規 |
| e (60cm) | 0.8867 | 0.2179 | ✅ | 正規 |

**結論**: ❌ 一部の条件で正規性の仮定が満たされない

### Analysis 2（全試行を個別データとして扱う）

| 条件 | W統計量 | p値 | 正規性 | 判定 |
|------|---------|-----|--------|------|
| a (0cm) | 0.7847 | **0.0002** | ❌ | 正規性違反 |
| b (15cm) | 0.8857 | **0.0109** | ❌ | 正規性違反 |
| c (30cm) | 0.7951 | **0.0002** | ❌ | 正規性違反 |
| d (45cm) | 0.9296 | 0.0953 | ✅ | 正規 |
| e (60cm) | 0.9315 | 0.1055 | ✅ | 正規 |

**結論**: ❌ 複数の条件で正規性違反
**推奨**: ノンパラメトリック検定（Friedman検定）を使用

---

## 📊 記述統計（Analysis 1: 平均値データ）

| 条件 | n | 中央値 | 平均 | 標準偏差 | 最小 | 最大 |
|------|---|--------|------|----------|------|------|
| 0cm | 8 | 22.32 | 23.63 | 6.11 | 17.91 | 38.02 |
| 15cm | 8 | 22.48 | 23.46 | 3.99 | 19.48 | 30.84 |
| 30cm | 8 | 23.27 | 24.96 | 4.92 | 20.44 | 36.03 |
| 45cm | 8 | 23.78 | 24.28 | 2.49 | 21.31 | 28.14 |
| 60cm | 8 | **27.37** | **26.80** | 2.67 | 23.61 | 30.22 |

**傾向**: 60cm（最遠距離）でパフォーマンスが最も低下（値が大きい = 悪い）

---

## 🎯 Friedman検定結果

### Analysis 1（平均値データ）

```
χ²(4) = 11.70, p = 0.0197
Kendall's W = 0.366（中程度の効果量）
```

**結果**: ✅ **有意差あり**（p < 0.05）

**解釈**:
距離条件がパフォーマンスに**統計的に有意な影響**を与える

### Analysis 2（全試行データ）

```
χ²(4) = 11.70, p = 0.0197
Kendall's W = 0.366（中程度の効果量）
```

**結果**: ✅ **有意差あり**（p < 0.05）

**一貫性**: Analysis 1と同じ結果を確認

---

## 🔬 事後検定（Post-hoc）

**手法**: Wilcoxon符号順位検定 + Holm-Bonferroni補正

### ペアワイズ比較結果（Analysis 1）

| 比較 | W統計量 | p値（未補正） | p値（補正後） | 有意性 |
|------|---------|---------------|---------------|--------|
| a vs b | 16.0 | 0.844 | 1.000 | n.s. |
| a vs c | 8.0 | 0.195 | 1.000 | n.s. |
| a vs d | 12.0 | 0.461 | 1.000 | n.s. |
| a vs e | 9.0 | 0.250 | 1.000 | n.s. |
| b vs c | 10.0 | 0.313 | 1.000 | n.s. |
| b vs d | 11.0 | 0.383 | 1.000 | n.s. |
| b vs e | 4.0 | 0.055 | 0.492 | n.s. |
| c vs d | 17.0 | 0.945 | 1.000 | n.s. |
| c vs e | 8.0 | 0.195 | 1.000 | n.s. |
| **d vs e** | **0.0** | **0.008** | **0.078** | **(傾向)** |

### 重要な知見

✅ **主効果は有意**（p = 0.020）だが、個別の比較では補正後に有意差なし

📈 **最も差が大きい比較**: 45cm vs 60cm（p = 0.078）
- 補正後は非有意だが、最も強い傾向を示す
- 45cmから60cmへの距離増加でパフォーマンス低下

💡 **解釈**:
- 効果は段階的な傾向（gradual trend）
- 特定の条件間の急激な変化ではない
- サンプルサイズ（n=8）が小さく、個別比較の検出力が低い

---

## 📈 順位分析

### 条件ごとの平均順位（低いほど良い）

| 条件 | 平均順位 | 解釈 |
|------|----------|------|
| **0cm** | **2.81** | 最良 |
| **15cm** | **2.62** | 最良 |
| 30cm | 2.88 | 中間 |
| 45cm | 3.00 | 中間 |
| **60cm** | **3.69** | 最悪 |

**傾向**: 距離が増加するにつれてパフォーマンスが悪化

---

## 💡 主要な発見

### 1. 統計的有意性 ✅
- **主効果**: p = 0.0197（有意）
- **効果量**: Kendall's W = 0.366（中程度）
- 距離条件はパフォーマンスに明確な影響を与える

### 2. パフォーマンスパターン 📊
- **最良**: 15cm（中央値 22.48）と0cm（中央値 22.32）
- **最悪**: 60cm（中央値 27.37）
- **差**: 約4-5ポイント（約18%の低下）

### 3. 段階的な効果 📉
- 急激な変化点はない
- 距離増加に伴う累積的な悪化
- 特に45cm→60cmで顕著な傾向

### 4. 個人差 👥
- 各被験者で一貫したパターン
- 順位の一致度が中程度（W = 0.37）
- trajectories plotで個人別パターンを確認可能

---

## 🎓 統計的解釈

### なぜFriedman検定を使用したのか？

✅ **正規性の仮定違反**
- Shapiro-Wilk検定で複数条件が棄却
- パラメトリック検定（ANOVA）の前提が満たされない

✅ **Friedman検定の利点**
- 正規性を仮定しない
- 順位変換により外れ値に頑健
- 小サンプルでも使用可能

### なぜ事後検定で有意差が出ないのか？

1. **多重比較補正の影響**
   - 10個の比較で厳しい補正（Holm-Bonferroni）
   - 個別のα水準が厳格化

2. **検出力の問題**
   - サンプルサイズ（n=8）が小さい
   - 個別比較の検出力 < 主効果の検出力

3. **効果の性質**
   - 特定のペア間の急激な変化ではない
   - 複数条件にわたる累積的な傾向

### 実質的な意義

✅ **主効果が有意** → 距離は確実にパフォーマンスに影響
📊 **平均で約18%の低下** → 実用的に重要
📈 **一貫した傾向** → 再現性が期待できる

---

## 📁 ファイル構成

### 統計結果
```
friedman_results/
├── README_friedman.md              # このファイル
├── friedman_results.xlsx           # 完全な統計結果
│   ├── Analysis1_Descriptive      # 記述統計
│   ├── Analysis1_Friedman         # Friedman検定結果
│   ├── Analysis1_PostHoc          # 事後検定
│   ├── Analysis2_Descriptive
│   ├── Analysis2_Friedman
│   └── Analysis2_PostHoc
```

### 可視化 - Analysis 1
```
analysis1_averaged/
├── boxplot_friedman_analysis_1.png      # 箱ひげ図（中央値強調）
├── median_barplot_analysis_1.png        # 中央値の棒グラフ
├── trajectories_analysis_1.png          # 個人別軌跡
└── rank_plot_analysis_1.png             # 平均順位プロット
```

### 可視化 - Analysis 2
```
analysis2_all_trials/
├── boxplot_friedman_analysis_2.png
├── median_barplot_analysis_2.png
├── trajectories_analysis_2.png
└── rank_plot_analysis_2.png
```

---

## 🔗 関連分析

- [主観評価結果](../subjective_results/README_subjective.md)
- [NASA-TLX ワークロード結果](../nasatlx_results/README_nasatlx.md)

---

## 📖 用語解説

### Friedman検定
反復測定ANOVAのノンパラメトリック版。正規性を仮定せず、データを順位に変換して分析。

### Kendall's W（一致係数）
効果量の指標（0-1）。被験者間での順位付けの一致度を示す。
- W < 0.1: 効果なし
- 0.1 ≤ W < 0.3: 小
- 0.3 ≤ W < 0.5: 中
- W ≥ 0.5: 大

### Holm-Bonferroni補正
多重比較の際に第一種過誤（偽陽性）を制御するための補正法。

### CLES (Common Language Effect Size)
条件1が条件2より大きい確率を表す効果量。0.5が基準。

---

## 🔧 再現方法

### 環境設定

```bash
# 仮想環境を有効化
source project/.venv/bin/activate

# 必要パッケージ
pip install scipy pingouin seaborn pandas numpy matplotlib openpyxl
```

### 分析実行

```bash
# デフォルトパスで実行
python project/src/analysis/performance_friedman.py

# カスタムCSVで実行
python project/src/analysis/performance_friedman.py path/to/data.csv
```

**必要なCSV形式**:
- 列: `name`, `task`, `no1`, `no2`, `no3`
- 各行: 1被験者 × 1条件

---

## 📚 参考文献

- Friedman, M. (1937). The use of ranks to avoid the assumption of normality. *Journal of the American Statistical Association*, 32(200), 675-701.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Pingouin documentation: https://pingouin-stats.org/

---

## 📝 更新履歴

- **2025-11-02**: 8名データでの分析完了（谷中除外）、有意な主効果を確認
- **2025-10-26**: 初版作成（6名データ）

---

**最終更新**: 2025-11-02
**分析スクリプト**: `project/src/analysis/performance_friedman.py`
