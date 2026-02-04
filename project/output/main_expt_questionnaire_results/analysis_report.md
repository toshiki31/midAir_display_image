# Post-Experiment Questionnaire Analysis

## Experiment Overview

- **Design**: 2 conditions within-subject (A vs B)
- **Participants**: 13
- **Questionnaire Components**:
  - 6 Likert scale items (7-point scale): Task/translation evaluation
  - 10 SUS questions (5-point scale): System usability
- **Analysis Date**: 2026-02-04 09:44:41

## Part 1: Task/Translation Evaluation (Likert Scale)

### 1. Descriptive Statistics

| Item | Condition | N | Mean | SD | Median | Min | Max |
|------|-----------|---|------|----|---------|----|-----|
| task_ease | A | 13 | 3.77 | 1.69 | 3.00 | 2 | 7 |
| task_ease | B | 13 | 4.31 | 1.75 | 4.00 | 2 | 7 |
| task_understanding | A | 13 | 5.31 | 1.03 | 5.00 | 4 | 7 |
| task_understanding | B | 13 | 5.31 | 1.32 | 5.00 | 2 | 7 |
| face_visibility | A | 13 | 2.23 | 1.42 | 2.00 | 1 | 6 |
| face_visibility | B | 13 | 5.23 | 0.83 | 5.00 | 4 | 6 |
| translation_accuracy | A | 13 | 3.54 | 1.27 | 3.00 | 2 | 6 |
| translation_accuracy | B | 13 | 3.77 | 1.54 | 4.00 | 1 | 6 |
| translation_readability | A | 13 | 3.15 | 1.21 | 3.00 | 2 | 5 |
| translation_readability | B | 13 | 3.69 | 1.49 | 4.00 | 2 | 6 |
| translation_unnaturalness | A | 13 | 4.77 | 1.59 | 5.00 | 2 | 7 |
| translation_unnaturalness | B | 13 | 4.23 | 1.36 | 4.00 | 2 | 7 |

### 2. Normality Testing (Shapiro-Wilk)

| Item | Condition | W | p-value | Normal | Test Used |
|------|-----------|---|---------|--------|------------|
| task_ease | A | 0.8499 | 0.0284 | No | Non-parametric (Wilcoxon) |
| task_ease | B | 0.9186 | 0.2399 | Yes | |
| task_understanding | A | 0.8884 | 0.0927 | Yes | Non-parametric (Wilcoxon) |
| task_understanding | B | 0.7692 | 0.0030 | No | |
| face_visibility | A | 0.7712 | 0.0032 | No | Non-parametric (Wilcoxon) |
| face_visibility | B | 0.7849 | 0.0045 | No | |
| translation_accuracy | A | 0.9167 | 0.2266 | Yes | Parametric (paired t-test) |
| translation_accuracy | B | 0.9494 | 0.5889 | Yes | |
| translation_readability | A | 0.8075 | 0.0084 | No | Non-parametric (Wilcoxon) |
| translation_readability | B | 0.8871 | 0.0892 | Yes | |
| translation_unnaturalness | A | 0.8867 | 0.0880 | Yes | Parametric (paired t-test) |
| translation_unnaturalness | B | 0.9529 | 0.6431 | Yes | |

### 3. Paired Comparisons

#### task_ease
**タスクはやりやすかった**

- **Test**: Wilcoxon signed-rank
- **Condition A**: M = 3.77, SD = 1.69
- **Condition B**: M = 4.31, SD = 1.75
- **Difference (B - A)**: 0.54
- **Statistic**: W = 15.5000, p = 0.4609
- **Effect Size**: Rank-biserial = 0.830 (large)
- **Result**: No significant difference (p >= 0.05)

#### task_understanding
**タスクは理解できた**

- **Test**: Wilcoxon signed-rank
- **Condition A**: M = 5.31, SD = 1.03
- **Condition B**: M = 5.31, SD = 1.32
- **Difference (B - A)**: 0.00
- **Statistic**: W = 13.0000, p = 1.0000
- **Effect Size**: Rank-biserial = 0.857 (large)
- **Result**: No significant difference (p >= 0.05)

#### face_visibility
**相手の顔は見やすかった**

- **Test**: Wilcoxon signed-rank
- **Condition A**: M = 2.23, SD = 1.42
- **Condition B**: M = 5.23, SD = 0.83
- **Difference (B - A)**: 3.00
- **Statistic**: W = 0.0000, p = 0.0005
- **Effect Size**: Rank-biserial = 1.000 (large)
- **Result**: Significant difference: Condition B > Condition A (p < 0.05)

#### translation_accuracy
**翻訳は正確だった**

- **Test**: Paired t-test
- **Condition A**: M = 3.54, SD = 1.27
- **Condition B**: M = 3.77, SD = 1.54
- **Difference (B - A)**: 0.23
- **Statistic**: t = -0.4064, p = 0.6916
- **Effect Size**: Hedges' g = -0.159 (negligible)
- **Result**: No significant difference (p >= 0.05)

#### translation_readability
**翻訳は読みやすかった**

- **Test**: Wilcoxon signed-rank
- **Condition A**: M = 3.15, SD = 1.21
- **Condition B**: M = 3.69, SD = 1.49
- **Difference (B - A)**: 0.54
- **Statistic**: W = 19.5000, p = 0.4551
- **Effect Size**: Rank-biserial = 0.786 (medium)
- **Result**: No significant difference (p >= 0.05)

#### translation_unnaturalness
**翻訳は不自然だった**

- **Test**: Paired t-test
- **Condition A**: M = 4.77, SD = 1.59
- **Condition B**: M = 4.23, SD = 1.36
- **Difference (B - A)**: -0.54
- **Statistic**: t = 0.9218, p = 0.3748
- **Effect Size**: Hedges' g = 0.352 (small)
- **Result**: No significant difference (p >= 0.05)

### Summary: Likert Items

**1 items showed significant differences:**

- **face_visibility**: p = 0.0005, B > A, effect size = 1.000

## Part 2: System Usability Scale (SUS) Analysis

### 1. SUS Score Calculation

**Methodology:**
- Odd questions (1,3,5,7,9): contribution = response - 1
- Even questions (2,4,6,8,10): contribution = 5 - response
- SUS Score = sum(contributions) × 2.5
- Score range: 0-100
- Benchmark: 68 points (average), ≥70 (good), <50 (poor)

### 2. SUS Scores by Participant

| Participant | Condition A | Condition B | Change (B-A) |
|-------------|-------------|-------------|---------------|
| aiuraforuniv | 45.0 | 77.5 | +32.5 |
| atsuchi | 60.0 | 50.0 | -10.0 |
| hata | 67.5 | 52.5 | -15.0 |
| hatamoto | 70.0 | 67.5 | -2.5 |
| hiraoka | 65.0 | 57.5 | -7.5 |
| kasuga | 27.5 | 67.5 | +40.0 |
| kuroki | 30.0 | 57.5 | +27.5 |
| machino | 32.5 | 70.0 | +37.5 |
| mitsumaru | 67.5 | 60.0 | -7.5 |
| nakao | 60.0 | 82.5 | +22.5 |
| nakayama | 55.0 | 40.0 | -15.0 |
| takiguchi | 60.0 | 52.5 | -7.5 |
| taninaka | 72.5 | 67.5 | -5.0 |

### 3. Descriptive Statistics

**Condition A:**
- Mean: 54.81
- SD: 15.83
- Median: 60.00
- Range: 27.5 - 72.5
- **Assessment**: Below average (50-68)

**Condition B:**
- Mean: 61.73
- SD: 11.79
- Median: 60.00
- Range: 40.0 - 82.5
- **Assessment**: Below average (50-68)

### 4. Statistical Comparison

**Normality Test (Shapiro-Wilk):**
- Condition A: W = 0.8583, p = 0.0366 (NOT Normal)
- Condition B: W = 0.9760, p = 0.9540 (Normal)
- Test used: Non-parametric (Wilcoxon)

**Paired Comparison (Wilcoxon signed-rank):**
- Statistic: W = 36.0000
- p-value: 0.5295
- Effect size: Rank-biserial = 0.604 (medium)
- **Result**: No significant difference in usability (p >= 0.05)

### Interpretation

No significant difference in usability between conditions was found. Both conditions scored below the average benchmark (68), suggesting room for improvement.

## Visualizations

- Likert boxplots: `visualizations/likert_boxplots.png`
- Likert heatmap: `visualizations/likert_heatmap.png`
- SUS boxplot: `visualizations/sus_boxplot.png`
- SUS trajectories: `visualizations/sus_trajectories.png`
- SUS comparison: `visualizations/sus_comparison.png`

## Statistical Methods

- **Design**: Within-subject (paired comparisons)
- **Normality testing**: Shapiro-Wilk test
- **Paired tests**: t-test (parametric) or Wilcoxon signed-rank (non-parametric)
- **Effect sizes**: Hedges' g (parametric) or rank-biserial correlation (non-parametric)
- **Significance level**: α = 0.05

## Overall Conclusions

**Likert Scale Evaluation:**
- 1 out of 6 task/translation evaluation items showed significant differences
- Significant differences indicate distinct user experiences between conditions

**System Usability (SUS):**
- Condition A mean: 54.8
- Condition B mean: 61.7
- No significant difference in overall system usability (p = 0.5295)

---
Generated by questionnaire_analysis.py on 2026-02-04 09:44:41
