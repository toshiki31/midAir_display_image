# Friedman検定分析結果 - 説明書

## 概要

このディレクトリには、performance.csvデータに対する**Friedman検定（フリードマン検定）**による統計解析結果が含まれています。

### Friedman検定とは

**Friedman検定**は、**反復測定ANOVAのノンパラメトリック版**です。

| 項目 | 反復測定ANOVA | Friedman検定 |
|------|---------------|--------------|
| データの種類 | パラメトリック | ノンパラメトリック |
| 正規性の仮定 | **必要** | **不要** |
| 使用するデータ | 実測値 | 順位（ランク） |
| 外れ値の影響 | 受けやすい | 受けにくい |
| 検出力 | 高い（正規分布の場合） | やや低い |

### いつFriedman検定を使うべきか

以下の条件を満たす場合、Friedman検定が適しています：

✓ **正規性の仮定が満たされない**（Shapiro-Wilk検定で棄却）
✓ **被験者内計画**（同じ参加者が複数条件を経験）
✓ **3つ以上の条件**を比較したい
✓ **外れ値**が多く含まれている
✓ **サンプルサイズが小さい**

---

## データ構造

- **被験者数**: 6名
- **条件数**: 5水準（a, b, c, d, e）
- **繰り返し測定**: 各条件で3試行
- **合計データ点**: 90データ点（6被験者 × 5条件 × 3試行）

### 分析の種類

#### **Analysis 1: 平均化データ分析**
- 3試行の平均値を使用
- データ点数: 30（6被験者 × 5条件）

#### **Analysis 2: 全試行データ分析**
- 3試行すべてを別々のデータとして扱う
- データ点数: 90（6被験者 × 5条件 × 3試行）

---

## ファイル構造

```
friedman_results/
├── README_friedman.md                 # このファイル
├── friedman_results.xlsx              # すべての統計結果
├── analysis1_averaged/                # Analysis 1の可視化
│   ├── boxplot_friedman_analysis_1.png
│   ├── median_barplot_analysis_1.png
│   ├── rank_plot_analysis_1.png
│   └── trajectories_analysis_1.png
└── analysis2_all_trials/              # Analysis 2の可視化
    ├── boxplot_friedman_analysis_2.png
    ├── median_barplot_analysis_2.png
    ├── rank_plot_analysis_2.png
    └── trajectories_analysis_2.png
```

---

## friedman_results.xlsx の内容

### シート一覧

1. **Analysis1_Descriptive** - Analysis 1の記述統計（ノンパラメトリック）
2. **Analysis1_Friedman** - Analysis 1のFriedman検定結果
3. **Analysis1_PostHoc** - Analysis 1の事後検定（有意差がある場合のみ）
4. **Analysis2_Descriptive** - Analysis 2の記述統計
5. **Analysis2_Friedman** - Analysis 2のFriedman検定結果
6. **Analysis2_PostHoc** - Analysis 2の事後検定（有意差がある場合のみ）

---

## 各シートのカラム説明

### 1. Descriptive（記述統計）シート

ノンパラメトリック分析では、**平均値よりも中央値**を重視します。

| カラム名 | 説明 | 重要度 |
|---------|------|--------|
| **condition** | 条件（a, b, c, d, e） | - |
| **count** | データ点数 | ★ |
| **median** | **中央値**（最も重要な代表値） | ★★★ |
| **mean** | 平均値（参考値） | ★ |
| **q1** | 第1四分位数（25%タイル） | ★★ |
| **q3** | 第3四分位数（75%タイル） | ★★ |
| **iqr** | 四分位範囲（q3 - q1）= ばらつきの指標 | ★★ |
| **min** | 最小値 | ★ |
| **max** | 最大値 | ★ |

**解釈のポイント:**
- **median**（中央値）：データの中央の値。外れ値の影響を受けにくい
- **iqr**（四分位範囲）：データの中央50%の広がり。標準偏差より外れ値に強い
- medianが条件間で大きく異なる場合、条件効果の可能性
- iqrが大きい場合、個人差が大きいことを示す

---

### 2. Friedman（Friedman検定）シート

**最も重要なシート**です。条件による効果を検定します。

| カラム名 | 説明 |
|---------|------|
| **Source** | 変動の要因（condition = 条件効果） |
| **W** | Kendallの一致係数（0～1、1に近いほど条件間に差がある） |
| **ddof1** | 自由度（条件数 - 1） |
| **Q** | Friedman検定統計量（カイ二乗分布に従う） |
| **p-unc** | p値（有意確率） |

#### 重要なカラムの詳細解説

**Q統計量（Friedman統計量）**
- 各条件の順位和の差に基づく統計量
- 大きいほど条件間に差がある
- カイ二乗分布（χ²分布）に従う

**p値（p-unc）**
- 統計的有意性を示す確率
- **p < 0.001**: 非常に有意（***）
- **p < 0.01**: 非常に有意（**）
- **p < 0.05**: 有意（*）
- **p ≥ 0.05**: 有意差なし（n.s.）

**Kendallの一致係数（W）**
- 効果量の指標（0～1）
- **W < 0.1**: 効果なし
- **0.1 ≤ W < 0.3**: 小さい効果
- **0.3 ≤ W < 0.5**: 中程度の効果
- **W ≥ 0.5**: 大きい効果

---

### 3. PostHoc（事後検定）シート

Friedman検定で有意差が認められた場合のみ作成されます。
どの条件間に差があるかを特定します。

| カラム名 | 説明 |
|---------|------|
| **Condition_1** | 比較する条件1 |
| **Condition_2** | 比較する条件2 |
| **W-statistic** | Wilcoxon検定のW統計量 |
| **p-uncorrected** | 補正前のp値 |
| **p-corrected** | Holm-Bonferroni法で補正後のp値 |
| **Effect_Size** | 効果量（CLES: Common Language Effect Size） |

**解釈のポイント:**
- **p-corrected < 0.05**: 2つの条件間に有意差あり
- **W-statistic**: 小さいほど条件1が小さい、大きいほど条件1が大きい

**効果量（CLES）の解釈:**

CLESは「条件1の値が条件2の値より大きい確率」を表します。

- **CLES = 0.5**: 効果なし（どちらも同じ）
- **CLES > 0.5**: 条件1の方が大きい傾向
  - **0.5 < CLES < 0.56**: 小さい効果
  - **0.56 ≤ CLES < 0.64**: 中程度の効果
  - **CLES ≥ 0.64**: 大きい効果
- **CLES < 0.5**: 条件2の方が大きい傾向
  - **0.44 < CLES < 0.5**: 小さい効果
  - **0.36 ≤ CLES ≤ 0.44**: 中程度の効果
  - **CLES ≤ 0.36**: 大きい効果

**例:**
- CLES = 0.70 → 条件1が条件2より大きい確率が70%（大きい効果）
- CLES = 0.30 → 条件2が条件1より大きい確率が70%（大きい効果）

---

## 可視化ファイルの説明

### 1. boxplot_friedman_*.png
- 箱ひげ図：**中央値**（箱の中の線）を強調
- 赤いダイヤモンド：平均値（参考）
- **用途**: 各条件のデータ分布とばらつきを視覚的に比較
- **見方**: 箱の位置（中央値）が条件間で大きくずれていれば差がある可能性

### 2. median_barplot_*.png
- 各条件の**中央値**を棒グラフで表示
- 赤いダイヤモンド：平均値（参考）
- **用途**: 条件間の中央値の差を視覚的に比較
- **見方**: 棒の高さの違いが条件効果を示す

### 3. trajectories_*.png
- 各被験者の条件ごとのパフォーマンスの変化を線グラフで表示
- **用途**: 個人差や条件間のパターンを確認
- **見方**:
  - 線が平行 → 個人差はあるが条件効果は一定
  - 線が交差 → 条件効果が人によって異なる（交互作用）

### 4. rank_plot_*.png (Friedman検定特有)
- 各条件の**平均順位**を横棒グラフで表示
- **用途**: Friedman検定が使用する順位データを視覚化
- **見方**:
  - 順位が低い（左側）→ パフォーマンスが高い
  - 順位が高い（右側）→ パフォーマンスが低い
  - 条件間で順位が大きく異なれば有意差の可能性

**順位の計算方法:**
各被験者内で、5つの条件に1位～5位の順位をつけ、その順位の平均を算出

---

## 統計分析の流れ

```
1. データ読み込み
   ↓
2. 記述統計（中央値、IQRなど）
   ↓
3. Friedman検定
   │  ・各被験者内でデータを順位に変換
   │  ・順位和を条件間で比較
   │  ・Q統計量とp値を計算
   ↓
4. 結果判定
   ├─ p < 0.05 → 有意差あり → 事後検定へ
   └─ p ≥ 0.05 → 有意差なし → 終了
   ↓
5. 事後検定（ペアごとの比較）
   │  ・Wilcoxon signed-rank test
   │  ・Holm-Bonferroni補正
   └─ 効果量（CLES）の計算
```

---

## 現在のデータの分析結果まとめ

### Analysis 1（平均化データ）
- **χ²(4) = 4.27, p = 0.371**
- **W = 0.18**（小さい効果量）
- **結論**: 条件による有意な効果は認められませんでした（p > 0.05）

### Analysis 2（全試行データ）
- **χ²(4) = 4.27, p = 0.371**
- **W = 0.18**（小さい効果量）
- **結論**: 条件による有意な効果は認められませんでした（p > 0.05）

**注**: 両分析とも同じ結果となりました。

---

## Friedman検定 vs 反復測定ANOVA

### 比較表

| 項目 | 反復測定ANOVA | Friedman検定 |
|------|---------------|--------------|
| **仮定** | 正規性、球面性 | **なし** |
| **使用データ** | 実測値 | 順位 |
| **検出力** | 高い（正規の場合） | やや低い |
| **ロバスト性** | 低い（外れ値に弱い） | **高い** |
| **解釈** | 平均値の差 | 分布全体の差 |
| **適用場面** | 正規分布のデータ | 非正規・外れ値あり |

### 本データでの選択理由

今回のデータでは、**すべての条件でShapiro-Wilk検定が有意**（p < 0.05）となり、正規性の仮定が満たされませんでした。そのため、Friedman検定の方が適切です。

### 結果の違い

| 検定方法 | p値 | 結論 |
|---------|-----|------|
| 反復測定ANOVA（GG補正後） | 0.521 | 有意差なし |
| Friedman検定 | 0.371 | 有意差なし |

どちらの検定でも有意差は認められませんでしたが、p値は若干異なります。

---

## スクリプトの実行方法

### 基本的な使い方

```bash
# 仮想環境の有効化
source project/.tobiienv/bin/activate

# デフォルトのCSVファイル（performance.csv）で実行
python project/src/analysis/performance_friedman.py
```

### 他のCSVファイルで実行

```bash
# カスタムCSVファイルを指定
python project/src/analysis/performance_friedman.py path/to/your_data.csv
```

**CSVファイルの形式要件:**
- 列名: `name`（被験者名）、`task`（条件）、`no1`, `no2`, `no3`（繰り返し測定）
- 形式: 各行が1つの被験者×条件の組み合わせ

---

## 使用した統計パッケージ

- **pandas**: データ処理
- **scipy.stats**: 基本的な統計関数
- **pingouin**: Friedman検定、Wilcoxon検定
- **matplotlib & seaborn**: 可視化
- **numpy**: 数値計算

---

## よくある質問（FAQ）

### Q1: Friedman検定でも有意でない場合はどうすればよいですか？
**A**:
- サンプルサイズを増やす
- 効果量（W）を確認する（小さい効果があっても検出力不足の可能性）
- 条件の設定を見直す
- 個人差が大きい場合、共変量を考慮した分析を検討

### Q2: ANOVA とFriedman検定で結果が異なる場合はどちらを信じればよいですか？
**A**:
- 正規性が満たされない場合 → **Friedman検定**を信頼
- 正規性が満たされる場合 → **ANOVA**の方が検出力が高い
- 両方報告して、結果が一貫しているか確認するのがベスト

### Q3: 効果量（W）が小さい場合はどう解釈すればよいですか？
**A**:
- **W < 0.1**: 実質的な効果はほぼない
- サンプルサイズを増やしても有意にならない可能性が高い
- 条件設定や測定方法を見直す

### Q4: 順位（rank）を使うとなぜロバストなのですか？
**A**:
- 外れ値も他の値と同様に順位に変換される
- 例: 20, 25, 30, 100 → 順位: 1, 2, 3, 4（100の影響が小さい）
- 実測値: 100が平均を大きく引き上げる
- 順位: 100も「4位」という1つの順位として扱われる

### Q5: サンプルサイズが小さい（n=6）場合の注意点は？
**A**:
- 検出力が低く、本当は差があっても検出できない可能性
- 小標本ではp値が不安定
- 効果量（W）も併せて解釈する
- できれば n ≥ 10 を目指す

---

## トラブルシューティング

### エラー: "Module not found: pingouin"
```bash
source project/.tobiienv/bin/activate
pip install pingouin
```

### エラー: "No such file or directory: performance.csv"
CSVファイルのパスを確認してください。
```bash
python performance_friedman.py path/to/your_file.csv
```

### 警告: "Sample size is small (n < 10)"
小標本では結果が不安定になる可能性があります。
- 可能であればサンプルサイズを増やす
- 効果量を重視する
- 結果の解釈に慎重になる

---

## 参考文献・リンク

### 統計的検定
- Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675-701.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.

### Pythonパッケージ
- Pingouin documentation: https://pingouin-stats.org/
- Pingouin Friedman test: https://pingouin-stats.org/generated/pingouin.friedman.html

### ノンパラメトリック検定の基礎
- 帰無仮説（H0）: すべての条件の分布が同一
- 対立仮説（H1）: 少なくとも1つの条件が他と異なる
- 多重比較補正: Holm-Bonferroni法

---

## 統計用語の英日対訳

| 英語 | 日本語 |
|------|--------|
| Friedman test | Friedman検定（フリードマン検定） |
| Non-parametric | ノンパラメトリック |
| Repeated measures | 反復測定 |
| Within-subject | 被験者内 |
| Median | 中央値 |
| Interquartile range (IQR) | 四分位範囲 |
| Rank | 順位 |
| Post-hoc test | 事後検定 |
| Wilcoxon signed-rank test | Wilcoxon符号順位検定 |
| Effect size | 効果量 |
| CLES | 共通言語効果量 |
| Kendall's W | Kendallの一致係数 |

---

## 更新履歴

- 2025-10-26: 初版作成

---

## お問い合わせ

分析に関する質問や追加の統計解析が必要な場合は、スクリプト作成者にお問い合わせください。

**生成スクリプト**: `project/src/analysis/performance_friedman.py`
