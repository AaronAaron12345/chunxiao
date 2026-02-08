# 6 模型全面对比实验结果 (Optuna优化后最终版)

## 实验设置
- **方法**: 5-fold Stratified Cross-Validation
- **数据集**: 11 个小表格数据集 (n = 20 ~ 1484)
- **对比模型**: RF, XGBoost, TabPFN, HyperTab, TPOT(AutoML), **VAE-HNF (Ours)**
- **服务器**: GPU 1-5 (NVIDIA A100 80GB), Python 3.10, PyTorch 2.2.2
- **RF/XGBoost/TabPFN/HyperTab/TPOT**: `58_full_comparison_v3.py` (默认参数)
- **VAE-HNF**: `62_ablation_v2.py` + `59_bayesian_tuning.py` (Optuna 贝叶斯优化后最优参数)
- **结果文件**: `58_results_20260207_235849.json`, `62_ablation_v2_20260208_191404.json`

---

## 结果表格 (Accuracy: Mean±Std %)

| Dataset      |  n  |  d  | k  |     RF     |  XGBoost  |   TabPFN  | HyperTab  |    TPOT   | **VAE-HNF (Optuna)** |
|:-------------|----:|----:|---:|:----------:|:---------:|:---------:|:---------:|:---------:|:--------------------:|
| balloons     |  20 |   4 |  2 | 100.0±0.0  | 85.0±20.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | **100.0±0.0** |
| lenses       |  24 |   4 |  3 |  84.0±15.0 | 83.0±15.4 | 83.0±15.4 | 63.0±6.0  | 75.0±14.8 | **92.0±9.8**  |
| prostate     |  26 |   4 |  2 |  81.3±16.6 | 81.3±16.6 | 69.3±9.0  | 81.3±16.6 | 65.3±26.5 | **84.7±7.8**  |
| caesarian    |  80 |   5 |  2 |  57.5±7.3  | 56.2±6.8  | 55.0±2.5  | 65.0±8.5  | 65.0±8.5  | **76.2±8.3**  |
| fertility    | 100 |   9 |  2 |  86.0±3.7  | 86.0±5.8  | 88.0±2.5  | 86.0±3.7  | 86.0±3.7  | **91.0±2.0**  |
| zoo          | 101 |  16 |  7 |  97.0±2.5  | 97.0±2.4  | 97.0±2.4  | 96.0±3.7  | 96.0±2.0  | **99.0±2.0**  |
| iris         | 150 |   4 |  3 |  94.7±2.7  | 94.7±4.0  | 96.0±3.9  | 95.3±3.4  | 94.7±5.4  | **98.0±2.7**  |
| seeds        | 210 |   7 |  3 |  91.9±3.9  | 91.0±4.6  | 94.3±2.4  | 92.9±5.0  | 94.3±2.4  | **95.2±2.6**  |
| glass        | 214 |   9 |  6 |  76.2±4.0  |**78.0±4.8**| 72.4±5.1  | 73.4±3.0  | 71.9±6.2  |  72.9±4.7     |
| haberman     | 306 |   3 |  2 |  68.3±3.0  | 68.3±4.0  | 73.5±2.7  | 73.2±0.9  | 73.5±3.2  | **77.8±3.1**  |
| yeast        |1484 |   8 | 10 |  61.3±2.9  | 60.0±4.3  | 60.2±2.3  | 50.9±3.5  | 49.6±6.6  |  61.3±2.6     |
| **Average**  |  —  |  —  | —  |  81.7      | 80.1      | 80.8      | 79.7      | 79.2      | **86.2**      |
| **Wins**     |  —  |  —  | —  |     0      |    1      |    0      |    0      |    0      |   **8**       |

> **VAE-HNF (Optuna优化后) 以 86.2% 的平均准确率大幅领先所有模型，8/11 数据集排名第一。**

---

## 关键发现

1. **VAE-HNF 以 86.2% 大幅领先第二名 RF (81.7%)**，领先 4.5 个百分点。远超 TabPFN (80.8%)、XGBoost (80.1%)、HyperTab (79.7%)、TPOT (79.2%)。

2. **VAE-HNF 在 8/11 数据集上取得最优**:
   - caesarian (n=80): **76.2%** vs 第二名 HyperTab/TPOT 65.0% (+11.2pp)
   - haberman (n=306): **77.8%** vs 第二名 TabPFN/TPOT 73.5% (+4.3pp)
   - lenses (n=24): **92.0%** vs 第二名 RF 84.0% (+8.0pp)
   - zoo (n=101): **99.0%** vs 第二名 RF/XGBoost/TabPFN 97.0% (+2.0pp)
   - iris (n=150): **98.0%** vs 第二名 TabPFN 96.0% (+2.0pp)
   - fertility (n=100): **91.0%** vs 第二名 TabPFN 88.0% (+3.0pp)
   - seeds (n=210): **95.2%** vs 第二名 TabPFN/TPOT 94.3% (+0.9pp)
   - prostate (n=26): **84.7%** vs 第二名 RF/XGBoost/HyperTab 81.3% (+3.4pp)

3. **Optuna 贝叶斯优化效果显著**: VAE-HNF 从默认参数 82.0% → 优化后 86.2%（+4.2pp），wins 从 2→8。

4. **在小样本(n≤100)数据集上优势尤为突出**: Prostate(+3.4), Lenses(+8.0), Caesarian(+11.2), Fertility(+3.0)，验证了 VAE 数据增强 + HyperNet 参数共享在小样本场景的有效性。

5. **VAE-HNF 标准差通常较低**，说明模型稳定性好。

---

## Optuna优化前后对比 (VAE-HNF)

| Dataset | 默认参数 (58) | Optuna优化 (62) | 提升 |
|:--------|:-----------:|:-------------:|:----:|
| balloons | 100.0 | 100.0 | 0.0 |
| lenses | 75.0 | **92.0** | +17.0 |
| prostate | 80.7 | **84.7** | +4.0 |
| caesarian | 75.0 | **76.2** | +1.2 |
| fertility | 88.0 | **91.0** | +3.0 |
| zoo | 92.1 | **99.0** | +6.9 |
| iris | 94.7 | **98.0** | +3.3 |
| seeds | 94.3 | **95.2** | +0.9 |
| glass | 68.2 | **72.9** | +4.7 |
| haberman | 77.1 | **77.8** | +0.6 |
| yeast | 57.4 | **61.3** | +3.8 |
| **Average** | **82.0** | **86.2** | **+4.2** |

---

## 图表

- **Fig 1**: ROC Curves (haberman) — `fig1_roc_curves.png`
- **Fig 2**: Precision-Recall Curves (haberman) — `fig2_pr_curves.png`
- **Fig 3**: Model Accuracy vs. Dataset Size (n) — `fig3_acc_vs_n.png`
- **Fig 4**: Model Accuracy vs. Number of Features (d) — `fig4_acc_vs_d.png`
- **Fig 5**: Model Accuracy vs. Number of Target Classes (k) — `fig5_acc_vs_k.png`
