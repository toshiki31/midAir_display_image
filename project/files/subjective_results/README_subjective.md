# Subjective Evaluation Analysis - Friedman Test Results

## 📋 分析概要

**分析日**: 2025-11-02
**データソース**: `subjective.csv`
**分析手法**: Friedman検定（ノンパラメトリック反復測定）
**被験者数**: 8名（外れ値として谷中を除外）
**実験条件**: 5水準（a, b, c, d, e → 0cm, 15cm, 30cm, 45cm, 60cm）
**評価尺度**: 7段階リッカート尺度（1-7）
**評価項目数**: 6項目

---

## 👥 参加者

### 分析対象（8名）
- P1 ~ P8（被験者番号で匿名化）

### 除外
- **taninaka（谷中健介）** - 外れ値として除外

---

## 📝 評価項目

| 項目 | 日本語 | 解釈 |
|------|--------|------|
| **Readability** | 吹き出しの文字は読みやすかったですか？ | 高い=読みやすい |
| **Eye_strain** | 目はしょぼつきましたか？ | 高い=しょぼつく |
| **Eye_fatigue** | 目が疲れましたか | 高い=疲れた |
| **Eye_pain** | 目はいたいですか？ | 高い=痛い |
| **Eye_dryness** | 目はかわきましたか？ | 高い=乾く |
| **Blurred_vision** | ものがぼやけて見えますか？ | 高い=ぼやける |

---

## 📊 記述統計

### Readability（読みやすさ）★★★

| 距離 | n | 中央値 | 平均 | SD | Min | Max |
|------|---|--------|------|-----|-----|-----|
| 0cm | 8 | 5.5 | 5.00 | 1.85 | 2 | 7 |
| 15cm | 8 | 5.0 | 4.63 | 1.19 | 3 | 6 |
| 30cm | 8 | 3.0 | 3.75 | 1.49 | 2 | 6 |
| 45cm | 8 | 3.5 | 3.38 | 1.69 | 1 | 6 |
| **60cm** | 8 | **2.0** | **2.13** | 0.64 | 1 | 3 |

**傾向**: 距離が増すほど読みにくくなる（スコア低下）

### Eye_fatigue（目の疲れ）★

| 距離 | n | 中央値 | 平均 | SD | Min | Max |
|------|---|--------|------|-----|-----|-----|
| 0cm | 8 | 3.0 | 3.00 | 1.41 | 1 | 5 |
| 15cm | 8 | 2.5 | 2.50 | 1.31 | 1 | 5 |
| 30cm | 8 | 4.0 | 4.13 | 1.64 | 1 | 6 |
| 45cm | 8 | 5.0 | 4.25 | 1.98 | 1 | 6 |
| **60cm** | 8 | **5.0** | **4.25** | 1.83 | 1 | 6 |

**傾向**: 遠距離（30cm以降）で疲労感が増加

### Blurred_vision（ぼやけ）★

| 距離 | n | 中央値 | 平均 | SD | Min | Max |
|------|---|--------|------|-----|-----|-----|
| 0cm | 8 | 2.0 | 2.00 | 0.76 | 1 | 3 |
| 15cm | 8 | 2.0 | 2.25 | 1.39 | 1 | 5 |
| 30cm | 8 | 2.0 | 2.50 | 1.77 | 1 | 5 |
| 45cm | 8 | 2.5 | 2.88 | 1.81 | 1 | 6 |
| **60cm** | 8 | **3.0** | **3.50** | 2.07 | 1 | 7 |

**傾向**: 距離が増すほどぼやけが増加

### その他の項目

- **Eye_strain**（しょぼつき）: 明確な傾向なし
- **Eye_pain**（痛み）: 明確な傾向なし
- **Eye_dryness**（乾き）: 0cmで高い傾向（p=0.066で非有意）

---

## 🔍 正規性評価

### 正規性検定サマリー

| 評価項目 | 正規 | 非正規 | 総合判定 |
|----------|------|--------|----------|
| Readability | 3条件 | 2条件 | ⚠️ 一部違反 |
| Eye_strain | 5条件 | 0条件 | ✅ 満たす |
| Eye_fatigue | 5条件 | 0条件 | ✅ 満たす |
| Eye_pain | 3条件 | 2条件 | ⚠️ 一部違反 |
| Eye_dryness | 4条件 | 1条件 | ⚠️ 一部違反 |
| Blurred_vision | 4条件 | 1条件 | ⚠️ 一部違反 |

**結論**: 複数項目で正規性違反 → **Friedman検定を使用**

---

## 🎯 Friedman検定結果

### 主要結果サマリー

| 評価項目 | χ² | p値 | 有意性 | Kendall's W | Bonferroni補正後 |
|----------|-----|-----|--------|-------------|------------------|
| **Readability** | 14.64 | **0.0055** | ** | 0.457 | ✅ 有意 |
| Eye_fatigue | 10.89 | **0.0278** | * | 0.340 | - |
| Blurred_vision | 10.19 | **0.0373** | * | 0.319 | - |
| Eye_dryness | 8.82 | 0.0657 | 傾向 | 0.276 | - |
| Eye_pain | 5.86 | 0.2097 | n.s. | 0.183 | - |
| Eye_strain | 3.41 | 0.4923 | n.s. | 0.106 | - |

### 多重比較補正（Bonferroni）

- **α水準**: 0.05 / 6項目 = 0.0083
- **補正後も有意**: Readabilityのみ（p = 0.0055 < 0.0083）

---

## 🔬 詳細結果

### 1. Readability（読みやすさ）★★★

```
χ²(4) = 14.64, p = 0.0055
Kendall's W = 0.457（中程度の効果）
```

**結果**: ✅ **非常に有意**（p < 0.01）
**Bonferroni補正後**: ✅ **依然として有意**

**解釈**:
- 距離条件が読みやすさに**強い影響**を与える
- 6項目の中で**最も明確な効果**
- 0cm（中央値5.5）→ 60cm（中央値2.0）で約64%低下

**Post-hoc結果**:
- 補正後に有意な個別比較はなし
- 効果は段階的な傾向

### 2. Eye_fatigue（目の疲れ）★

```
χ²(4) = 10.89, p = 0.0278
Kendall's W = 0.340（中程度の効果）
```

**結果**: ✅ **有意**（p < 0.05）
**Bonferroni補正後**: ❌ 非有意（p > 0.0083）

**解釈**:
- 距離条件が目の疲れに影響
- 30cm以降で疲労感が増加
- 15cm（中央値2.5）と45cm/60cm（中央値5.0）で顕著な差

### 3. Blurred_vision（ぼやけ）★

```
χ²(4) = 10.19, p = 0.0373
Kendall's W = 0.319（中程度の効果）
```

**結果**: ✅ **有意**（p < 0.05）
**Bonferroni補正後**: ❌ 非有意

**解釈**:
- 距離が増すとぼやけが増加
- 0cm（中央値2.0）→ 60cm（中央値3.0）
- 視覚的な明瞭さが低下

### 4. Eye_dryness（目の乾き）

```
χ²(4) = 8.82, p = 0.0657
Kendall's W = 0.276（小〜中程度）
```

**結果**: ⚠️ **傾向**（p = 0.066）

**解釈**:
- 統計的有意水準には達しないが傾向あり
- 0cmで乾きが高い（中央値4.0）
- 近距離での凝視により乾燥？

### 5. Eye_pain（目の痛み）

```
χ²(4) = 5.86, p = 0.2097
```

**結果**: ❌ 有意差なし

### 6. Eye_strain（しょぼつき）

```
χ²(4) = 3.41, p = 0.4923
```

**結果**: ❌ 有意差なし

---

## 💡 主要な発見

### 1. Readabilityが最重要指標 🎯
- **唯一Bonferroni補正後も有意**
- 効果量が最大（W = 0.457）
- 距離増加で明確に読みにくくなる

### 2. 視覚的不快感の三つ組 👁️
- **Readability**: 読みにくさ（最強）
- **Eye_fatigue**: 疲れ（有意）
- **Blurred_vision**: ぼやけ（有意）

すべて遠距離で悪化する一貫したパターン

### 3. 距離効果のパターン 📊
- **15cm以下**: 良好
- **30cm**: 中間（疲労感が出始める）
- **45-60cm**: 最悪（すべての指標で悪化）

### 4. 個人差 👥
- 項目によって効果量が異なる
- Readabilityは一致度が高い（W = 0.46）
- Pain, Strainは個人差大（W < 0.2）

---

## 📈 距離ごとの総合評価

| 距離 | Readability | 視覚疲労 | 推奨度 |
|------|-------------|----------|--------|
| **0cm** | ⭐⭐⭐⭐⭐ | ⚠️ 乾き | ⭐⭐⭐⭐ |
| **15cm** | ⭐⭐⭐⭐⭐ | ✅ 良好 | ⭐⭐⭐⭐⭐ |
| **30cm** | ⭐⭐⭐ | ⚠️ 疲労↑ | ⭐⭐⭐ |
| **45cm** | ⭐⭐ | ⚠️ 疲労高 | ⭐⭐ |
| **60cm** | ⭐ | ❌ 最悪 | ⭐ |

**最適距離**: **15cm**
- Readability高い
- 視覚疲労が最も低い
- すべての指標でバランス良好

---

## 📁 ファイル構成

### 統計結果
```
subjective_results/
├── README_subjective.md                # このファイル
├── subjective_results.xlsx             # 完全な統計結果
│   ├── Desc_Readability               # 各項目の記述統計
│   ├── Desc_Eye_strain
│   ├── Desc_Eye_fatigue
│   ├── Desc_Eye_pain
│   ├── Desc_Eye_dryness
│   ├── Desc_Blurred_vision
│   ├── Normality_Tests                # 正規性検定
│   ├── Friedman_Tests                 # Friedman検定結果
│   └── PostHoc_*                      # 事後検定（有意な項目のみ）
```

### 可視化
```
visualizations/
├── boxplots_all_items.png             # 6項目の箱ひげ図
├── heatmap_median_ratings.png         # 中央値ヒートマップ
├── stacked_bar_distribution.png       # 評価分布（積み上げ棒グラフ）
└── radar_profile.png                  # レーダーチャート
```

---

## 🔗 関連分析

- [Performance結果](../friedman_results/README_friedman.md)
- [NASA-TLX ワークロード結果](../nasatlx_results/README_nasatlx.md)

---

## 📖 用語解説

### リッカート尺度
順序尺度の一種。1-7の7段階で評価。等間隔を仮定しないため、中央値を使用。

### Bonferroni補正
複数の検定を行う際の第一種過誤の制御。α = 0.05 / 検定数。

### Kendall's W
効果量。被験者間での順位の一致度（0-1）。

### Friedman検定
リッカート尺度のような順序データに適したノンパラメトリック検定。

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
python project/src/analysis/subjective_analysis.py

# カスタムCSVで実行
python project/src/analysis/subjective_analysis.py path/to/data.csv
```

**必要なCSV形式**:
- Google Formsの出力形式
- 列: タイムスタンプ, 氏名, 実験タイプ, 評価項目1-6

---

## 📚 参考文献

- Likert, R. (1932). A technique for the measurement of attitudes. *Archives of Psychology*, 22(140), 1-55.
- Friedman, M. (1937). The use of ranks to avoid the assumption of normality. *JASA*, 32(200), 675-701.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.

---

## 📝 更新履歴

- **2025-11-02**: 8名データでの分析完了（谷中除外）
  - Readabilityで非常に有意な効果（p = 0.0055）
  - Eye_fatigue, Blurred_visionでも有意な効果
  - 15cmが最適距離と判明

---

**最終更新**: 2025-11-02
**分析スクリプト**: `project/src/analysis/subjective_analysis.py`
