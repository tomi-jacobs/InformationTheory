# 📊 Information Theory Scores & Phylogenomic Ortholog Analysis

This repository contains scripts, data, and figures for an analysis of information-theoretic scores (TC, RTC, TCA, RTCA) computed across orthologous gene alignments in *Aizoaceae* (ice plants), and their relationship to phylogenetic difficulty as predicted by [Pythia 2.0](https://github.com/tschuelia/PyPythia).

This work is part of a broader phylogenomics project (AIM 1) on *Aizoaceae* using 39 taxa.

---

## 🗂️ Repository Structure

```
InformationTheory/
│
├── data/
│   ├── Correlation_TreeStats.csv       # TC, RTC, TCA, RTCA, Taxa, Pythia scores for all orthologs
│   └── filtered_orthologs.csv          # Orthologs passing the importance filter
│
├── scripts/
│   ├── run_pythia.py                   # Batch Pythia difficulty prediction across all MSAs
│   └── analyse_orthologs.py            # Correlation, regression, filtering, ASTRAL prep
│
├── figures/
│   ├── correlation_heatmap.png         # Spearman correlation matrix across all scores
│   ├── scatter_TC_vs_Pythia.png        # TC vs Pythia difficulty
│   ├── scatter_RTC_vs_Pythia.png       # RTC vs Pythia difficulty
│   ├── scatter_TCA_vs_Pythia.png       # TCA vs Pythia difficulty
│   ├── scatter_RTCA_vs_Pythia.png      # RTCA vs Pythia difficulty
│   ├── scatter_Taxa_vs_Pythia.png      # Taxa count vs Pythia difficulty
│   ├── regression_coefficients.png     # Standardised regression coefficients
│   └── filter_decision_plot.png        # Ortholog filtering decision space
│
└── README.md
```

---

## 📖 Background

### Information-Theoretic Scores
TC (Transfer Concordance), RTC, TCA, and RTCA are branch support scores computed using [RAxML](https://github.com/stamatak/standard-RAxML) that quantify how well individual gene trees agree with a reference topology. Higher absolute values indicate stronger phylogenetic signal.

### Pythia Difficulty
[Pythia 2.0](https://github.com/tschuelia/PyPythia) (Haag & Stamatakis, 2025) predicts the difficulty of a Maximum Likelihood phylogenetic analysis for a given multiple sequence alignment (MSA), on a scale from **0 (easy)** to **1 (difficult)**. Easy alignments yield consistent, reproducible tree topologies; difficult ones yield statistically indistinguishable but topologically distinct trees.

---

## ⚙️ Scripts

### `run_pythia.py`
Batch-runs Pythia 2.0 across all ortholog alignments in a directory and outputs a CSV of predicted difficulty scores.

**Requirements:**
```bash
pip install pythiaphylopredictor
pip install "numpy<2.0"   # Pythia 2.0 requires numpy < 2.0
```

**Usage:**
```bash
# Edit these two lines at the top of the script:
INPUT_DIR = "/path/to/alignments"
RAXML_NG  = "/path/to/raxml-ng/bin/raxml-ng"

python3 run_pythia.py
```

---

### `analyse_orthologs.py`
Runs the full downstream analysis:
- Spearman correlation matrix and heatmap
- Individual scatter plots (each score vs Pythia difficulty)
- Multiple linear regression predicting Pythia from all scores
- Filters orthologs by low Pythia difficulty + high |TCA|
- Builds ASTRAL input tree files (filtered and unfiltered)
- Prints ready-to-run ASTRAL commands

**Requirements:**
```bash
pip install seaborn scikit-learn scipy
```

**Usage:**
```bash
# Edit the configuration block at the top of the script:
CSV_PATH     = "/path/to/Correlation_TreeStats.csv"
TREEFILE_DIR = "/path/to/alignments/"
ASTRAL_JAR   = "/path/to/astral.5.7.8.jar"

python3 analyse_orthologs.py
```

---

## 📊 Key Results

### Correlation with Pythia Difficulty

| Score | Spearman r | p-value |
|-------|-----------|---------|
| TC    | -0.667    | < 0.001 |
| RTC   | -0.673    | < 0.001 |
| TCA   | -0.665    | < 0.001 |
| RTCA  | -0.664    | < 0.001 |
| Taxa  | -0.269    | < 0.001 |

All four information-theoretic scores show a strong, consistent negative correlation with Pythia difficulty — orthologs with higher concordance scores are independently predicted by Pythia as easier to resolve phylogenetically. This validates the information-theoretic approach as a meaningful proxy for phylogenetic signal quality.

### Ortholog Filtering
Orthologs were filtered for downstream "important ortholog" tree inference using two criteria:
- **Pythia difficulty < 0.5** (clean, resolvable phylogenetic signal)
- **|TCA| in the top 25%** (strong branch support)

This yielded **1,095 orthologs** (23% of 4,845 total) for the filtered species tree, compared to using all 4,845 orthologs in the unfiltered tree.

---

## 🌳 Species Tree Inference

Two ASTRAL species trees are inferred for comparison:

```bash
# Unfiltered (all 4845 orthologs)
java -jar astral.5.7.8.jar \
  -i all_orthologs_trees.tre \
  -o ASTRAL_all_orthologs.tre

# Filtered (1095 orthologs: low Pythia + high |TCA|)
java -jar astral.5.7.8.jar \
  -i filtered_orthologs_trees.tre \
  -o ASTRAL_filtered_orthologs.tre
```

---

## 📚 Citations

> Haag, J. & Stamatakis, A. (2025). *Pythia 2.0: New Data, New Prediction Model, New Features*. bioRxiv. https://doi.org/10.1101/2025.03.25.645182

> Haag, J., Höhler, D., Bettisworth, B. & Stamatakis, A. (2022). *From easy to hopeless – predicting the difficulty of phylogenetic analyses*. Molecular Biology and Evolution, 39(12).

> Kozlov, A.M. et al. (2019). *RAxML-NG: a fast, scalable and user-friendly tool for maximum likelihood phylogenetic inference*. Bioinformatics, 35(21):4453–4455.

> Zhang, C. et al. (2018). *ASTRAL-III: polynomial time species tree reconstruction from partially resolved gene trees*. BMC Bioinformatics, 19(S6):153.

---

## 👤 Author

**Tomi Jacobs** — PhD Candidate, Computational Biology / Phylogenomics
