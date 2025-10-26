# 反復測定ANOVA分析結果 - 説明書

## 概要

このディレクトリには、performance.csvデータに対する被験者内一要因（距離=5水準）の反復測定ANOVA統計解析結果が含まれています。

### データ構造
- **被験者数**: 6名（takiguchi, ishibashi, taninaka, hayashida, nakayama, honda）
- **距離条件**: 5水準（a, b, c, d, e）
- **繰り返し測定**: 各条件で3試行
- **合計データ点**: 90データ点（6被験者 × 5距離 × 3試行）

### 分析の種類

#### **Analysis 1: 平均化データ分析**
- 3試行の平均値を使用（各被験者×距離条件で1つのデータ点）
- データ点数: 30（6被験者 × 5距離）
- より保守的なアプローチ

#### **Analysis 2: 全試行データ分析**
- 3試行すべてを別々のデータとして扱う
- データ点数: 90（6被験者 × 5距離 × 3試行）
- より多くの情報を活用

---

## ファイル構造

```
anova_results/
├── README.md                          # このファイル
├── statistical_results.xlsx           # すべての統計結果（Excel形式）
├── analysis1_averaged/                # Analysis 1の可視化
│   ├── boxplot_violin_analysis_1_averaged.png
│   ├── trajectories_analysis_1_averaged.png
│   ├── qq_plots_analysis_1_averaged.png
│   └── barplot_mean_analysis_1_averaged.png
└── analysis2_all_trials/              # Analysis 2の可視化
    ├── boxplot_violin_analysis_2_all_trials.png
    ├── trajectories_analysis_2_all_trials.png
    ├── qq_plots_analysis_2_all_trials.png
    └── barplot_mean_analysis_2_all_trials.png
```

---

## statistical_results.xlsx の内容

Excelファイルには以下のシートが含まれています：

### シート一覧

1. **Analysis1_Descriptive** - Analysis 1の記述統計
2. **Analysis1_Normality** - Analysis 1の正規性検定
3. **Analysis1_ANOVA** - Analysis 1の反復測定ANOVA結果
4. **Analysis1_PostHoc** - Analysis 1の事後検定（有意差がある場合のみ）
5. **Analysis2_Descriptive** - Analysis 2の記述統計
6. **Analysis2_Normality** - Analysis 2の正規性検定
7. **Analysis2_ANOVA** - Analysis 2の反復測定ANOVA結果
8. **Analysis2_PostHoc** - Analysis 2の事後検定（有意差がある場合のみ）

---

## 各シートのカラム説明

### 1. Descriptive（記述統計）シート

距離条件ごとの基本統計量を示します。

| カラム名 | 説明 | 単位 |
|---------|------|------|
| **distance** | 距離条件（a, b, c, d, e） | - |
| **count** | データ点数 | 個 |
| **mean** | 平均値 | 秒（または該当単位） |
| **std** | 標準偏差（データのばらつき） | 秒 |
| **min** | 最小値 | 秒 |
| **25%** | 第1四分位数（下位25%の値） | 秒 |
| **50%** | 中央値（メディアン） | 秒 |
| **75%** | 第3四分位数（上位25%の値） | 秒 |
| **max** | 最大値 | 秒 |

**解釈のポイント:**
- meanが条件間で大きく異なる場合、条件効果の可能性
- stdが大きい場合、個人差が大きいことを示す
- 中央値(50%)と平均値(mean)が大きく異なる場合、外れ値の影響がある可能性

---

### 2. Normality（正規性検定）シート

各距離条件のデータが正規分布に従うかを検定した結果です。

| カラム名 | 説明 | 値の範囲 |
|---------|------|---------|
| **distance** | 距離条件 | a, b, c, d, e |
| **statistic** | Shapiro-Wilk検定のW統計量 | 0～1（1に近いほど正規分布に近い） |
| **p_value** | 有意確率 | 0～1 |
| **normal** | 正規性の判定 | Yes（正規分布）/ No（非正規分布） |

**解釈のポイント:**
- **帰無仮説（H0）**: データは正規分布に従う
- **p_value < 0.05**: H0を棄却 → データは正規分布に従わない
- **p_value ≥ 0.05**: H0を採択 → データは正規分布に従う
- すべての条件がNoの場合、ノンパラメトリック検定（Friedman検定）の使用を検討

---

### 3. ANOVA（反復測定分散分析）シート

距離条件による効果を検定した結果です。最も重要なシートです。

| カラム名 | 説明 |
|---------|------|
| **Source** | 変動の要因（distance = 条件間、Error = 誤差項） |
| **SS** | 平方和（Sum of Squares）- 変動の大きさ |
| **DF** | 自由度（Degrees of Freedom） |
| **MS** | 平均平方（Mean Square）= SS / DF |
| **F** | F統計量（条件効果の大きさ） |
| **p-unc** | p値（未補正） |
| **p-GG-corr** | Greenhouse-Geisser補正後のp値（球面性違反時） |
| **ng2** | 偏イータ二乗（効果量の指標） |
| **eps** | イプシロン（球面性の程度、ε）|
| **sphericity** | 球面性仮定の判定（True/False） |
| **W-spher** | Mauchlyの球面性検定のW統計量 |
| **p-spher** | 球面性検定のp値 |

#### 重要なカラムの詳細解説

**F統計量（F）**
- 条件間変動と誤差変動の比
- 大きいほど条件効果が強い
- 通常は1以上の値

**p値（p-unc / p-GG-corr）**
- 統計的有意性を示す確率
- **p < 0.001**: 非常に有意（***）
- **p < 0.01**: 非常に有意（**）
- **p < 0.05**: 有意（*）
- **p ≥ 0.05**: 有意差なし（n.s.）
- 球面性が破れている場合は**p-GG-corr**を使用

**偏イータ二乗（ng2）**
効果量の指標。条件要因が説明する分散の割合。
- **0.01**: 小さい効果
- **0.06**: 中程度の効果
- **0.14**: 大きい効果

**球面性（sphericity）**
- 反復測定ANOVAの前提条件
- **False**: 球面性仮定が破れている → GG補正を使用
- **True**: 球面性仮定が満たされている → 通常のp値を使用

**イプシロン（eps）**
- 球面性の程度を示す指標（0～1）
- **ε = 1**: 完全な球面性
- **ε < 0.75**: 球面性の大きな違反 → GG補正推奨
- **ε ≥ 0.75**: 軽微な違反 → Huynh-Feldt補正も可

---

### 4. PostHoc（事後検定）シート

ANOVAで有意差が認められた場合のみ作成されます。
どの距離条件間に差があるかを特定します。

| カラム名 | 説明 |
|---------|------|
| **Distance_1** | 比較する距離条件1 |
| **Distance_2** | 比較する距離条件2 |
| **t-statistic** | t統計量（差の大きさ） |
| **df** | 自由度 |
| **p-uncorrected** | 補正前のp値 |
| **p-corrected** | Holm-Bonferroni法で補正後のp値 |
| **Effect_size** | 効果量（Hedges' g） |

**解釈のポイント:**
- **p-corrected < 0.05**: 2つの条件間に有意差あり
- **t-statistic > 0**: Distance_1の方が大きい
- **t-statistic < 0**: Distance_2の方が大きい

**効果量（Hedges' g）の解釈:**
- **|g| < 0.2**: 小さい効果
- **0.2 ≤ |g| < 0.5**: 中程度の効果
- **0.5 ≤ |g| < 0.8**: 大きい効果
- **|g| ≥ 0.8**: 非常に大きい効果

---

## 可視化ファイルの説明

### 1. boxplot_violin_*.png
- **左側**: 箱ひげ図（中央値、四分位数、外れ値を表示）
- **右側**: バイオリンプロット（データ分布の形状を表示）
- **用途**: 各距離条件のデータ分布とばらつきを視覚的に比較

### 2. trajectories_*.png
- 各被験者の距離条件ごとのパフォーマンスの変化を線グラフで表示
- **用途**: 個人差や条件間のパターンを確認

### 3. qq_plots_*.png
- 各距離条件のQ-Qプロット（正規確率プロット）
- **用途**: 正規性仮定の視覚的確認
- データ点が直線上にあれば正規分布に近い

### 4. barplot_mean_*.png
- 各距離条件の平均値と標準誤差（SEM）を棒グラフで表示
- **用途**: 条件間の平均値の差を視覚的に比較

---

## 統計分析の流れ

```
1. データ読み込み
   ↓
2. 記述統計（平均、標準偏差など）
   ↓
3. 前提条件のチェック
   ├─ 正規性検定（Shapiro-Wilk）
   └─ 球面性検定（Mauchly）
   ↓
4. 反復測定ANOVA
   ↓
5. 結果判定
   ├─ p < 0.05 → 有意差あり → 事後検定へ
   └─ p ≥ 0.05 → 有意差なし → 終了
   ↓
6. 事後検定（ペアごとの比較）
   └─ Holm-Bonferroni補正
```

---

## 現在のデータの分析結果まとめ

### Analysis 1（平均化データ）
- **F(4, 20) = 0.606, p = 0.521（GG補正後）**
- **結論**: 距離条件による有意な効果は認められませんでした（p > 0.05）
- **正規性**: すべての条件で棄却（Friedman検定の使用推奨）
- **球面性**: 棄却（ε = 0.36 → GG補正適用済み）

### Analysis 2（全試行データ）
- **F(4, 20) = 0.606, p = 0.521（GG補正後）**
- **結論**: 距離条件による有意な効果は認められませんでした（p > 0.05）
- **正規性**: すべての条件で棄却（Friedman検定の使用推奨）
- **球面性**: 棄却（ε = 0.36 → GG補正適用済み）

**注**: 両分析とも同じ結果となりました。

---

## スクリプトの実行方法

```bash
# 仮想環境の有効化
source project/.tobiienv/bin/activate

# 分析の実行
python project/src/analysis/performance_anova.py
```

---

## 使用した統計パッケージ

- **pandas**: データ処理
- **scipy.stats**: Shapiro-Wilk検定
- **pingouin**: 反復測定ANOVA、球面性検定、事後検定
- **matplotlib & seaborn**: 可視化
- **numpy**: 数値計算

---

## 参考文献・リンク

### 反復測定ANOVA
- [Pythonでの分散分析ANOVA](https://statisticsschool.com/%E3%80%90python%E3%80%91%E5%88%86%E6%95%A3%E5%88%86%E6%9E%90anova%E3%81%AE%E5%9F%BA%E7%A4%8E%E3%81%8B%E3%82%89%E5%BF%9C%E7%94%A8%E3%81%BE%E3%81%A7%E7%B5%B1%E8%A8%88%E7%9A%84%E4%BB%AE%E8%AA%AC/)
- Pingouin documentation: https://pingouin-stats.org/

### 統計的検定の基礎
- 有意水準: α = 0.05（5%）
- 帰無仮説（H0）: 条件間に差がない
- 対立仮説（H1）: 条件間に差がある

### 多重比較補正
- Holm-Bonferroni法: 段階的に棄却水準を調整する方法（Bonferroni法より検出力が高い）

---

## トラブルシューティング

### Q1: 正規性が満たされない場合はどうすればよいですか？
**A**: ノンパラメトリック検定である**Friedman検定**の使用を検討してください。Friedman検定は反復測定ANOVAのノンパラメトリック版です。

### Q2: 球面性が破れている場合はどうなりますか？
**A**: 自動的にGreenhouse-Geisser（GG）補正が適用されます。p-GG-corrの値を使用してください。

### Q3: 効果量が小さいが有意差がある場合は？
**A**: 統計的には有意でも、実質的な意味（practical significance）は小さい可能性があります。効果量と合わせて解釈してください。

### Q4: サンプルサイズが小さい場合（n=6）の注意点は？
**A**:
- 検出力（power）が低く、本当は差があっても検出できない可能性
- 外れ値の影響を受けやすい
- 正規性の検定も信頼性が低下
- 結果の解釈には慎重さが必要

---

## 更新履歴

- 2025-10-26: 初版作成

---

## お問い合わせ

分析に関する質問や追加の統計解析が必要な場合は、スクリプト作成者にお問い合わせください。

**生成スクリプト**: `project/src/analysis/performance_anova.py`
