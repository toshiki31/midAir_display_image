# NASA-TLX Workload Analysis - Friedman Test Results

## 📋 分析概要

**分析日**: 2025-11-02
**データソース**: `nasa-tlx/*.csv`（45 CSVファイル）
**分析手法**: Friedman検定（ノンパラメトリック反復測定）
**被験者数**: 8名（外れ値として谷中を除外）
**実験条件**: 5水準（a, b, c, d, e → 0cm, 15cm, 30cm, 45cm, 60cm）
**評価尺度**: NASA-TLXトータルスコア（0-100、低いほど良い）

---

## 👥 参加者

### 分析対象（8名）
- takiguchi
- ishibashi
- hayashida
- nakayama
- honda
- mitsumaru
- shiojiri
- hatamoto

### 除外
- **taninaka（谷中健介）** - 外れ値として除外（5ファイル）

---

## 📝 NASA-TLXとは

### 概要
**NASA Task Load Index (NASA-TLX)** は、主観的ワークロードを測定する標準化されたツールです。

### 6つのサブスケール
1. **Mental Demand**（知的・知覚的要求）
2. **Physical Demand**（身体的要求）
3. **Temporal Demand**（タイムプレッシャー）
4. **Performance**（作業成績）
5. **Effort**（努力）
6. **Frustration**（フラストレーション）

### トータルスコア
- **範囲**: 0-100
- **解釈**: **低いほど良い**（ワークロードが小さい）
- **計算**: 各サブスケールに重み付けして合計

---

## 📊 記述統計

### 距離条件ごとのNASA-TLXスコア

| 距離 | n | 中央値 | 平均 | SD | Min | Max |
|------|---|--------|------|-----|-----|-----|
| 0cm | 8 | 36.83 | 34.92 | 27.21 | 0.0 | 67.33 |
| 15cm | 8 | 35.67 | 34.00 | 26.44 | 0.0 | 66.00 |
| 30cm | 8 | 45.00 | 39.83 | 28.02 | 0.0 | 70.67 |
| 45cm | 8 | 43.33 | 41.00 | 29.88 | 0.0 | 72.33 |
| **60cm** | 8 | **56.17** | **49.08** | 32.74 | 3.0 | 82.67 |

**傾向**: 距離が増加するにつれてワークロードが増加

### 個人別の平均スコア

| 被験者 | 平均 | 中央値 | Min | Max | 傾向 |
|--------|------|--------|-----|-----|------|
| hayashida | 0.60 | 0.00 | 0.0 | 3.0 | 非常に低い |
| mitsumaru | 14.00 | 10.00 | 8.7 | 30.0 | 低い |
| hatamoto | 20.93 | 19.00 | 6.7 | 40.7 | 低い |
| nakayama | 21.67 | 23.33 | 8.7 | 33.0 | 中 |
| shiojiri | 59.73 | 58.33 | 48.0 | 77.0 | 高い |
| honda | 67.13 | 66.00 | 63.7 | 71.7 | 高い |
| ishibashi | 67.40 | 67.33 | 57.0 | 79.0 | 高い |
| takiguchi | 66.67 | 70.00 | 54.7 | 82.7 | 高い |

**個人差**: 被験者間で大きな差（0.6 ~ 67.4）

---

## 🔍 正規性評価（Shapiro-Wilk検定）

| 距離 | W統計量 | p値 | 正規性 | 判定 |
|------|---------|-----|--------|------|
| 0cm | 0.8844 | 0.2075 | ✅ | 正規 |
| 15cm | 0.8923 | 0.2458 | ✅ | 正規 |
| 30cm | 0.8880 | 0.2242 | ✅ | 正規 |
| 45cm | 0.8612 | 0.1234 | ✅ | 正規 |
| 60cm | 0.8620 | 0.1258 | ✅ | 正規 |

**結論**: ✅ **すべての条件で正規性を満たす**

**注**:
- パラメトリック検定（反復測定ANOVA）も使用可能
- 今回はFriedman検定を実施（一貫性のため）

---

## 🎯 Friedman検定結果

### 主効果

```
χ²(4) = 11.79, p = 0.0190
Kendall's W = 0.368（中程度の効果量）
```

**結果**: ✅ **有意差あり**（p < 0.05）

**解釈**:
- 距離条件がワークロードに**統計的に有意な影響**を与える
- 遠距離ほどワークロード増加
- 効果量は中程度

---

## 🔬 事後検定（Post-hoc）

**手法**: Wilcoxon符号順位検定 + Holm-Bonferroni補正

### ペアワイズ比較結果

| 比較 | W統計量 | p値（未補正） | p値（補正後） | 有意性 | CLES |
|------|---------|---------------|---------------|--------|------|
| 0cm vs 15cm | 9.0 | 0.469 | 0.977 | n.s. | 0.539 |
| 0cm vs 30cm | 7.0 | 0.297 | 0.977 | n.s. | 0.430 |
| 0cm vs 45cm | 3.0 | 0.078 | 0.547 | n.s. | 0.383 |
| **0cm vs 60cm** | **4.0** | **0.055** | **0.492** | **(傾向)** | 0.336 |
| 15cm vs 30cm | 6.0 | 0.219 | 0.977 | n.s. | 0.406 |
| 15cm vs 45cm | 4.0 | 0.109 | 0.656 | n.s. | 0.383 |
| **15cm vs 60cm** | **3.0** | **0.039** | **0.391** | **(傾向)** | 0.328 |
| 30cm vs 45cm | 11.0 | 0.688 | 0.977 | n.s. | 0.453 |
| 30cm vs 60cm | 8.0 | 0.195 | 0.977 | n.s. | 0.359 |
| 45cm vs 60cm | 4.0 | 0.055 | 0.492 | n.s. | 0.359 |

### 重要な知見

✅ **主効果は有意**（p = 0.019）だが、個別比較では補正後に有意差なし

📈 **最も差が大きい比較**:
1. **15cm vs 60cm**: p = 0.391（最も強い傾向）
2. **0cm vs 60cm**: p = 0.492
3. **45cm vs 60cm**: p = 0.492

💡 **解釈**:
- 近距離（0-15cm）と遠距離（60cm）で最大の差
- 効果は段階的な累積
- サンプルサイズ（n=8）で検出力が制限

---

## 📈 距離ごとのワークロード評価

### トレンド分析

| 距離 | 平均スコア | 0cmからの増加率 | 評価 |
|------|------------|-----------------|------|
| **0cm** | 34.92 | - | ⭐⭐⭐⭐⭐ 最良 |
| **15cm** | 34.00 | -2.6% | ⭐⭐⭐⭐⭐ 最良 |
| **30cm** | 39.83 | +14.1% | ⭐⭐⭐⭐ 良 |
| **45cm** | 41.00 | +17.4% | ⭐⭐⭐ 中 |
| **60cm** | 49.08 | +40.6% | ⭐⭐ 悪 |

### 視覚的パターン

```
ワークロード ↑
60  |                           ●
50  |                       ●
40  |                   ●
30  |       ●   ●
20  |
10  |
0   +---+---+---+---+---+---→ 距離
    0  15  30  45  60 (cm)
```

**傾向**:
- 0-15cm: 最も低いワークロード（プラトー）
- 30-45cm: 中間
- 60cm: 急激な増加

---

## 💡 主要な発見

### 1. 距離効果の確認 ✅
- **有意な主効果**（p = 0.019）
- 効果量: 中程度（W = 0.37）
- 距離増加でワークロード増加

### 2. 最適距離の特定 🎯
- **0-15cm**: 最低ワークロード（平均34-35）
- ほぼ同等の負担
- 実用的な最適範囲

### 3. 限界距離の発見 ⚠️
- **60cm**: ワークロード急増（平均49）
- 0cmから+40%の増加
- 実用限界を超える可能性

### 4. 個人差の大きさ 👥
- 被験者間で0 ~ 82.7の範囲
- 一部の被験者は常に低スコア
- 一部の被験者は全条件で高スコア
- **タスク適性の個人差**を反映

---

## 🔍 他の指標との比較

| 指標 | 最適距離 | 最悪距離 | 一致度 |
|------|----------|----------|--------|
| **Performance** | 15cm | 60cm | ✅ 完全一致 |
| **Subjective (Readability)** | 0-15cm | 60cm | ✅ 完全一致 |
| **NASA-TLX** | 15cm | 60cm | ✅ 完全一致 |

**総合結論**: すべての指標で**15cmが最適**、**60cmが最悪**

---

## 📁 ファイル構成

### 統計結果
```
nasatlx_results/
├── README_nasatlx.md                  # このファイル
├── nasatlx_results.xlsx               # 完全な統計結果
│   ├── Raw_Data                      # 生データ
│   ├── Descriptive_Stats             # 記述統計
│   ├── Normality_Tests               # 正規性検定
│   ├── Friedman_Test                 # Friedman検定結果
│   └── PostHoc_Tests                 # 事後検定
```

### 可視化
```
visualizations/
├── boxplot_nasatlx.png                # 箱ひげ図
├── barplot_nasatlx.png                # 平均値の棒グラフ
├── trajectories_nasatlx.png           # 個人別軌跡
└── heatmap_nasatlx.png                # ヒートマップ
```

---

## 🔗 関連分析

- [Performance結果](../friedman_results/README_friedman.md)
- [主観評価結果](../subjective_results/README_subjective.md)

---

## 📖 用語解説

### NASA-TLX
NASA Task Load Index。主観的ワークロードの標準測定ツール。

### トータルスコア
6つのサブスケールを重み付けして合計。0-100の範囲。

### Kendall's W
効果量の指標（0-1）。被験者間での順位の一致度。

### CLES
Common Language Effect Size。条件1が条件2より大きい確率。

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
python project/src/analysis/nasatlx_analysis.py

# カスタムディレクトリで実行
python project/src/analysis/nasatlx_analysis.py path/to/nasa-tlx-dir
```

**必要なCSV形式**:
- ファイル名: `{participant}_{condition} - {displayname}.csv`
- 各CSVの8行目: トータルスコア

---

## 📚 参考文献

### NASA-TLX
- Hart, S. G., & Staveland, L. E. (1988). Development of NASA-TLX: Results of empirical and theoretical research. *Advances in psychology*, 52, 139-183.
- Hart, S. G. (2006). NASA-task load index (NASA-TLX); 20 years later. *Proceedings of the human factors and ergonomics society annual meeting*, 50(9), 904-908.

### 統計分析
- Friedman, M. (1937). The use of ranks to avoid the assumption of normality. *JASA*, 32(200), 675-701.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.

---

## 📝 更新履歴

- **2025-11-02**: 8名データでの分析完了（谷中除外）
  - 有意な主効果を確認（p = 0.019）
  - 15cmで最低ワークロード
  - 60cmでワークロード急増（+40%）
  - すべての指標で一貫した傾向

---

**最終更新**: 2025-11-02
**分析スクリプト**: `project/src/analysis/nasatlx_analysis.py`
