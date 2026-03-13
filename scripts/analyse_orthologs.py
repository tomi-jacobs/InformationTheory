#!/usr/bin/env python3
"""
Ortholog Analysis Pipeline
==========================
1. Correlation analysis (Spearman) between all scores + heatmap
2. Multiple regression predicting Pythia difficulty
3. Filter "important" orthologs using Pythia + TCA scores
4. Write ASTRAL input tree lists (filtered vs all)

Author: Tomi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings("ignore")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
CSV_PATH      = os.path.expanduser("~/data/standard-RAxML/InformationTheoryTJ/Correlation_TreeStats.csv")
TREEFILE_DIR  = os.path.expanduser("~/data/standard-RAxML/InformationTheoryTJ/InfoToCalc")
OUTPUT_DIR    = os.path.expanduser("~/data/standard-RAxML/InformationTheoryTJ/Analysis_Output")

# Filtering thresholds (data-driven, but you can override these manually)
PYTHIA_THRESHOLD  = 0.5    # keep orthologs with Pythia difficulty BELOW this
TCA_PERCENTILE    = 75     # keep orthologs with |TCA| in the top X percentile

# ASTRAL
ASTRAL_JAR    = os.path.expanduser("~/data/ASTRAL/Astral/astral.5.7.8.jar")
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Score columns we care about (excluding 'Not sure')
score_cols = ['TC', 'RTC', 'TCA', 'RTCA', 'Taxa', 'Pythia scores']
df_scores  = df[score_cols].copy()

print(f"  Loaded {len(df)} orthologs")
print(f"  Columns: {df.columns.tolist()}\n")

# ── 2. SPEARMAN CORRELATION MATRIX ───────────────────────────────────────────
print("Computing Spearman correlations...")
corr_matrix = df_scores.corr(method='spearman')

# Also compute p-values
n = len(df_scores)
p_matrix = pd.DataFrame(np.ones((len(score_cols), len(score_cols))),
                         index=score_cols, columns=score_cols)
for i, col1 in enumerate(score_cols):
    for j, col2 in enumerate(score_cols):
        if i != j:
            r, p = stats.spearmanr(df_scores[col1], df_scores[col2])
            p_matrix.loc[col1, col2] = p

# Save correlation table
corr_out = os.path.join(OUTPUT_DIR, "spearman_correlation_matrix.csv")
corr_matrix.round(4).to_csv(corr_out)
print(f"  Saved correlation matrix → {corr_out}")

# Print key correlations with Pythia
print("\n  Spearman correlations with Pythia difficulty:")
for col in ['TC', 'RTC', 'TCA', 'RTCA', 'Taxa']:
    r, p = stats.spearmanr(df_scores[col], df_scores['Pythia scores'])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"    {col:12s}: r = {r:+.4f}, p = {p:.2e} {sig}")

# ── 3. CORRELATION HEATMAP ────────────────────────────────────────────────────
print("\nPlotting correlation heatmap...")
fig, ax = plt.subplots(figsize=(8, 7))

# Mask upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Spearman r"},
    ax=ax
)

# Add significance stars
for i, row in enumerate(score_cols):
    for j, col in enumerate(score_cols):
        if i > j:  # lower triangle only
            p = p_matrix.loc[row, col]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            if sig:
                ax.text(j + 0.5, i + 0.75, sig, ha='center', va='center',
                        fontsize=9, color='black', fontweight='bold')

ax.set_title("Spearman Correlation Matrix\n(* p<0.05, ** p<0.01, *** p<0.001)",
             fontsize=13, pad=15)
plt.tight_layout()
heatmap_out = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
plt.savefig(heatmap_out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved heatmap → {heatmap_out}")

# ── 4. SCATTER PLOTS: each score vs Pythia ────────────────────────────────────
print("Plotting scatter plots vs Pythia difficulty...")
predictors = ['TC', 'RTC', 'TCA', 'RTCA', 'Taxa']

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, col in enumerate(predictors):
    ax = axes[i]
    x = df_scores[col]
    y = df_scores['Pythia scores']

    ax.scatter(x, y, alpha=0.3, s=8, color='steelblue', rasterized=True)

    # Regression line
    m, b, r, p, se = stats.linregress(x, y)
    xline = np.linspace(x.min(), x.max(), 200)
    ax.plot(xline, m * xline + b, color='crimson', linewidth=1.5)

    r_sp, p_sp = stats.spearmanr(x, y)
    sig = "***" if p_sp < 0.001 else ("**" if p_sp < 0.01 else ("*" if p_sp < 0.05 else "ns"))
    ax.set_title(f"{col} vs Pythia\nSpearman r={r_sp:+.3f} {sig}", fontsize=11)
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel("Pythia Difficulty", fontsize=10)
    ax.grid(True, alpha=0.3)

# Hide the unused 6th subplot
axes[5].set_visible(False)

plt.suptitle("Score vs Pythia Difficulty (n={})".format(len(df)), fontsize=13, y=1.01)
plt.tight_layout()
scatter_out = os.path.join(OUTPUT_DIR, "scatter_vs_pythia.png")
plt.savefig(scatter_out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved scatter plots → {scatter_out}")

# ── 5. MULTIPLE REGRESSION ────────────────────────────────────────────────────
print("\nRunning multiple regression (predicting Pythia difficulty)...")
X = df_scores[predictors].values
y = df_scores['Pythia scores'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

reg = LinearRegression()
reg.fit(X_scaled, y)
y_pred = reg.predict(X_scaled)

r2  = r2_score(y, y_pred)
# Adjusted R²
n_samples, n_features = X.shape
r2_adj = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

print(f"  R²          = {r2:.4f}")
print(f"  Adjusted R² = {r2_adj:.4f}")
print(f"  Intercept   = {reg.intercept_:.4f}")
print(f"\n  Standardised coefficients (effect size per 1 SD):")
for name, coef in zip(predictors, reg.coef_):
    print(f"    {name:12s}: {coef:+.4f}")

# Plot regression coefficients
fig, ax = plt.subplots(figsize=(7, 4))
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in reg.coef_]
bars = ax.barh(predictors, reg.coef_, color=colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel("Standardised Regression Coefficient", fontsize=11)
ax.set_title(f"Multiple Regression Predicting Pythia Difficulty\nR² = {r2:.3f}, Adjusted R² = {r2_adj:.3f}",
             fontsize=11)
for bar, val in zip(bars, reg.coef_):
    ax.text(val + (0.001 if val >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}', va='center',
            ha='left' if val >= 0 else 'right', fontsize=9)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
reg_out = os.path.join(OUTPUT_DIR, "regression_coefficients.png")
plt.savefig(reg_out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved regression plot → {reg_out}")

# ── 6. FILTER "IMPORTANT" ORTHOLOGS ──────────────────────────────────────────
print("\nFiltering important orthologs...")

# Criterion 1: Low Pythia difficulty (clean phylogenetic signal)
pythia_mask = df['Pythia scores'] < PYTHIA_THRESHOLD

# Criterion 2: High absolute TCA (strong branch support signal)
tca_cutoff  = np.percentile(df['TCA'].abs(), TCA_PERCENTILE)
tca_mask    = df['TCA'].abs() >= tca_cutoff

combined_mask = pythia_mask & tca_mask
df_filtered   = df[combined_mask].copy()

print(f"  Total orthologs         : {len(df)}")
print(f"  Pythia < {PYTHIA_THRESHOLD}            : {pythia_mask.sum()}")
print(f"  |TCA| >= {tca_cutoff:.2f} (top {100-TCA_PERCENTILE}%): {tca_mask.sum()}")
print(f"  Both filters combined   : {len(df_filtered)}")

# Save filtered orthologs table
filtered_csv = os.path.join(OUTPUT_DIR, "filtered_orthologs.csv")
df_filtered.to_csv(filtered_csv, index=False)
print(f"  Saved filtered table → {filtered_csv}")

# ── 7. PLOT FILTER DECISION ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df.loc[~combined_mask, 'Pythia scores'],
           df.loc[~combined_mask, 'TCA'].abs(),
           alpha=0.2, s=6, color='lightgray', label='Excluded', rasterized=True)
ax.scatter(df.loc[combined_mask, 'Pythia scores'],
           df.loc[combined_mask, 'TCA'].abs(),
           alpha=0.5, s=8, color='#2ecc71', label=f'Selected (n={len(df_filtered)})', rasterized=True)
ax.axvline(PYTHIA_THRESHOLD, color='steelblue', linestyle='--', linewidth=1.2,
           label=f'Pythia < {PYTHIA_THRESHOLD}')
ax.axhline(tca_cutoff, color='crimson', linestyle='--', linewidth=1.2,
           label=f'|TCA| ≥ {tca_cutoff:.2f} (top {100-TCA_PERCENTILE}%)')
ax.set_xlabel("Pythia Difficulty", fontsize=12)
ax.set_ylabel("|TCA|", fontsize=12)
ax.set_title("Ortholog Filtering: Low Difficulty + High |TCA|", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
filter_plot = os.path.join(OUTPUT_DIR, "filter_decision_plot.png")
plt.savefig(filter_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved filter plot → {filter_plot}")

# ── 8. WRITE ASTRAL INPUT TREE LISTS ─────────────────────────────────────────
print("\nCollecting treefile paths...")

def get_treefile(ortholog_name, treefile_dir):
    """Match ortholog name to its .treefile in the directory."""
    # The ortholog column contains the full filename including .fas-cln
    treefile = os.path.join(treefile_dir, ortholog_name + ".treefile")
    if os.path.exists(treefile):
        return treefile
    return None

# All orthologs
all_trees   = []
miss_all    = 0
for name in df['Orthologs']:
    tf = get_treefile(name, TREEFILE_DIR)
    if tf:
        all_trees.append(tf)
    else:
        miss_all += 1

# Filtered orthologs
filtered_trees = []
miss_filt      = 0
for name in df_filtered['Orthologs']:
    tf = get_treefile(name, TREEFILE_DIR)
    if tf:
        filtered_trees.append(tf)
    else:
        miss_filt += 1

print(f"  All orthologs    : {len(all_trees)} treefiles found, {miss_all} missing")
print(f"  Filtered         : {len(filtered_trees)} treefiles found, {miss_filt} missing")

# Write tree list files (cat all treefiles into single newick files for ASTRAL)
all_trees_file      = os.path.join(OUTPUT_DIR, "all_orthologs_trees.tre")
filtered_trees_file = os.path.join(OUTPUT_DIR, "filtered_orthologs_trees.tre")

with open(all_trees_file, 'w') as fout:
    for tf in all_trees:
        with open(tf) as fin:
            fout.write(fin.read().strip() + "\n")

with open(filtered_trees_file, 'w') as fout:
    for tf in filtered_trees:
        with open(tf) as fin:
            fout.write(fin.read().strip() + "\n")

print(f"  Saved all-orthologs tree file      → {all_trees_file}")
print(f"  Saved filtered-orthologs tree file → {filtered_trees_file}")

# ── 9. PRINT ASTRAL COMMANDS ──────────────────────────────────────────────────
all_out      = os.path.join(OUTPUT_DIR, "ASTRAL_all_orthologs.tre")
filtered_out = os.path.join(OUTPUT_DIR, "ASTRAL_filtered_orthologs.tre")

print("\n" + "="*70)
print("ASTRAL COMMANDS — copy and paste these into your terminal:")
print("="*70)
print(f"\n# 1. Species tree from ALL orthologs (unfiltered):")
print(f"java -jar {ASTRAL_JAR} \\")
print(f"  -i {all_trees_file} \\")
print(f"  -o {all_out} 2> {all_out}.log")

print(f"\n# 2. Species tree from FILTERED orthologs (low Pythia + high |TCA|):")
print(f"java -jar {ASTRAL_JAR} \\")
print(f"  -i {filtered_trees_file} \\")
print(f"  -o {filtered_out} 2> {filtered_out}.log")

print("\n" + "="*70)
print("After both trees are built, compare them with:")
print(f"java -jar {ASTRAL_JAR} --rfdist \\")
print(f"  -i {all_out} \\")  
print(f"  -q {filtered_out}")
print("="*70)

# ── 10. SUMMARY REPORT ───────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    ANALYSIS COMPLETE                         ║
╠══════════════════════════════════════════════════════════════╣
║  Output directory: {OUTPUT_DIR:<42}║
║                                                              ║
║  Files produced:                                             ║
║  • spearman_correlation_matrix.csv                           ║
║  • correlation_heatmap.png                                   ║
║  • scatter_vs_pythia.png                                     ║
║  • regression_coefficients.png                               ║
║  • filtered_orthologs.csv  ({len(df_filtered):>4} orthologs)              ║
║  • filter_decision_plot.png                                  ║
║  • all_orthologs_trees.tre  ({len(all_trees):>4} trees)                 ║
║  • filtered_orthologs_trees.tre  ({len(filtered_trees):>4} trees)            ║
╚══════════════════════════════════════════════════════════════╝
""")
