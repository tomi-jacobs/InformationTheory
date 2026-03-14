#!/usr/bin/env python3
"""
Ortholog Analysis Pipeline - Publication Style 3rd Trial
Author: Tomi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
OUTPUT_DIR    = os.path.expanduser("~/data/standard-RAxML/InformationTheoryTJ/Analysis_Output_3rdTrial")

PYTHIA_THRESHOLD = 0.5
TCA_PERCENTILE   = 75
ASTRAL_JAR       = os.path.expanduser("~/data/ASTRAL/Astral/astral.5.7.8.jar")
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── COLORS ────────────────────────────────────────────────────────────────────
DEEP_BLUE = '#1a5276'   # deep blue (not black)
CRIMSON   = '#c0392b'
GREEN     = '#27ae60'
GRAY      = '#bdc3c7'

# ── HELPER: apply bold ticks ──────────────────────────────────────────────────
def bold_ticks(ax, fontsize=13):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(fontsize)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
score_cols = ['TC', 'RTC', 'TCA', 'RTCA', 'Taxa', 'Pythia scores']
df_scores  = df[score_cols].copy()
print(f"  Loaded {len(df)} orthologs\n")

# ── 2. SPEARMAN CORRELATION MATRIX ───────────────────────────────────────────
print("Computing Spearman correlations...")
corr_matrix = df_scores.corr(method='spearman')

p_matrix = pd.DataFrame(np.ones((len(score_cols), len(score_cols))),
                         index=score_cols, columns=score_cols)
for i, col1 in enumerate(score_cols):
    for j, col2 in enumerate(score_cols):
        if i != j:
            r, p = stats.spearmanr(df_scores[col1], df_scores[col2])
            p_matrix.loc[col1, col2] = p

corr_out = os.path.join(OUTPUT_DIR, "spearman_correlation_matrix.csv")
corr_matrix.round(4).to_csv(corr_out)

print("\n  Spearman correlations with Pythia difficulty:")
for col in ['TC', 'RTC', 'TCA', 'RTCA', 'Taxa']:
    r, p = stats.spearmanr(df_scores[col], df_scores['Pythia scores'])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"    {col:12s}: r = {r:+.4f}, p = {p:.2e} {sig}")

# ── 3. CORRELATION HEATMAP ───────────────────────────────────────────────────
print("\nPlotting correlation heatmap...")
fig, ax = plt.subplots(figsize=(9, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".3f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5,
    annot_kws={"size": 13, "weight": "bold"},
    cbar_kws={"shrink": 0.8, "label": "Spearman r"}, ax=ax
)
for i, row in enumerate(score_cols):
    for j, col in enumerate(score_cols):
        if i > j:
            p = p_matrix.loc[row, col]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            if sig:
                ax.text(j + 0.5, i + 0.75, sig, ha='center', va='center',
                        fontsize=11, color='black', fontweight='bold')
ax.set_title("Spearman Correlation Matrix\n(* p<0.05, ** p<0.01, *** p<0.001)",
             fontsize=17, fontweight='bold', pad=15)
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', fontsize=13)
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold', fontsize=13)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {path}")

# ── 4. INDIVIDUAL SCATTER PLOTS ───────────────────────────────────────────────
print("\nPlotting individual scatter plots vs Pythia difficulty...")
predictors = ['TC', 'RTC', 'TCA', 'RTCA', 'Taxa']

for col in predictors:
    fig, ax = plt.subplots(figsize=(8, 6))

    x = df_scores[col]
    y = df_scores['Pythia scores']

    # Deep blue dots
    ax.scatter(x, y, alpha=0.45, s=14, color=DEEP_BLUE,
               rasterized=True, linewidths=0)

    # Crimson regression line
    m, b, _, _, _ = stats.linregress(x, y)
    xline = np.linspace(x.min(), x.max(), 300)
    ax.plot(xline, m * xline + b, color=CRIMSON, linewidth=2.5, zorder=5)

    r_sp, p_sp = stats.spearmanr(x, y)
    sig = "***" if p_sp < 0.001 else ("**" if p_sp < 0.01 else ("*" if p_sp < 0.05 else "ns"))

    ax.set_xlabel(col, fontsize=15, fontweight='bold', labelpad=8)
    ax.set_ylabel("Pythia Difficulty", fontsize=15, fontweight='bold', labelpad=8)
    ax.set_title(f"{col} vs Pythia Difficulty\nSpearman r = {r_sp:+.3f}  {sig}  (n={len(df)})",
                 fontsize=16, fontweight='bold', pad=10)

    bold_ticks(ax, fontsize=13)
    ax.tick_params(axis='both', which='major', length=6, width=1.5, direction='out')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    ax.text(0.97, 0.97,
            f"r = {r_sp:+.3f}\np = {p_sp:.2e}",
            transform=ax.transAxes,
            ha='right', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"scatter_{col}_vs_Pythia.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {fname}")

# ── 5. MULTIPLE REGRESSION ────────────────────────────────────────────────────
print("\nRunning multiple regression...")
X = df_scores[predictors].values
y = df_scores['Pythia scores'].values

scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
reg     = LinearRegression()
reg.fit(X_sc, y)
y_pred  = reg.predict(X_sc)
r2      = r2_score(y, y_pred)
n_s, n_f = X.shape
r2_adj  = 1 - (1 - r2) * (n_s - 1) / (n_s - n_f - 1)

print(f"  R²          = {r2:.4f}")
print(f"  Adjusted R² = {r2_adj:.4f}")

# Wide figure + large left margin to fully clear all y-tick labels
fig, ax = plt.subplots(figsize=(11, 5))
fig.subplots_adjust(left=0.22, right=0.88)   # left clears labels, right clears value annotations

colors = [GREEN if c > 0 else CRIMSON for c in reg.coef_]
bars   = ax.barh(predictors, reg.coef_, color=colors, edgecolor='white', height=0.55)
ax.axvline(0, color='black', linewidth=1.8)

ax.set_xlabel("Standardised Regression Coefficient", fontsize=14,
              fontweight='bold', labelpad=8)
ax.set_title(
    f"Multiple Regression: Predicting Pythia Difficulty\n"
    f"R\u00b2 = {r2:.3f}  |  Adjusted R\u00b2 = {r2_adj:.3f}",
    fontsize=16, fontweight='bold', pad=10
)

# Explicitly set y tick labels so they never overlap
ax.set_yticks(range(len(predictors)))
ax.set_yticklabels(predictors, fontsize=14, fontweight='bold', ha='right')
ax.tick_params(axis='y', pad=8)           # padding between label and axis
ax.tick_params(axis='x', which='major', length=6, width=1.5, direction='out')
bold_ticks(ax, fontsize=13)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# Value labels — placed well outside the bar ends
x_range = max(abs(v) for v in reg.coef_)
offset  = x_range * 0.04   # 4% of range as offset

for bar, val in zip(bars, reg.coef_):
    ax.text(val + (offset if val >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}', va='center',
            ha='left' if val >= 0 else 'right',
            fontsize=12, fontweight='bold')

ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
path = os.path.join(OUTPUT_DIR, "regression_coefficients.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {path}")

# ── 6. FILTER IMPORTANT ORTHOLOGS ────────────────────────────────────────────
print("\nFiltering important orthologs...")
pythia_mask  = df['Pythia scores'] < PYTHIA_THRESHOLD
tca_cutoff   = np.percentile(df['TCA'].abs(), TCA_PERCENTILE)
tca_mask     = df['TCA'].abs() >= tca_cutoff
combined     = pythia_mask & tca_mask
df_filtered  = df[combined].copy()

print(f"  Total                        : {len(df)}")
print(f"  Pythia < {PYTHIA_THRESHOLD}                  : {pythia_mask.sum()}")
print(f"  abs(TCA) >= {tca_cutoff:.2f} (top {100-TCA_PERCENTILE}%) : {tca_mask.sum()}")
print(f"  Both filters combined        : {len(df_filtered)}")

filtered_csv = os.path.join(OUTPUT_DIR, "filtered_orthologs.csv")
df_filtered.to_csv(filtered_csv, index=False)
print(f"  Saved → {filtered_csv}")

# ── 7. FILTER DECISION PLOT ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df.loc[~combined, 'Pythia scores'],
           df.loc[~combined, 'TCA'].abs(),
           alpha=0.2, s=8, color=GRAY,
           label='Excluded', rasterized=True)
ax.scatter(df.loc[combined, 'Pythia scores'],
           df.loc[combined, 'TCA'].abs(),
           alpha=0.6, s=10, color=GREEN,
           label=f'Selected (n={len(df_filtered)})', rasterized=True)
ax.axvline(PYTHIA_THRESHOLD, color='steelblue', linestyle='--',
           linewidth=2, label=f'Pythia < {PYTHIA_THRESHOLD}')
ax.axhline(tca_cutoff, color=CRIMSON, linestyle='--',
           linewidth=2, label=f'abs(TCA) >= {tca_cutoff:.2f} (top {100-TCA_PERCENTILE}%)')
ax.set_xlabel("Pythia Difficulty", fontsize=15, fontweight='bold', labelpad=8)
ax.set_ylabel("abs(TCA)", fontsize=15, fontweight='bold', labelpad=8)
ax.set_title("Ortholog Filtering: Low Difficulty + High abs(TCA)",
             fontsize=16, fontweight='bold', pad=10)
bold_ticks(ax, fontsize=13)
ax.tick_params(axis='both', which='major', length=6, width=1.5, direction='out')
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
ax.legend(fontsize=12, frameon=True, edgecolor='gray')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "filter_decision_plot.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {path}")

# ── 8. ASTRAL INPUT TREES ─────────────────────────────────────────────────────
print("\nCollecting treefiles...")

def get_treefile(name, tdir):
    tf = os.path.join(tdir, name + ".treefile")
    return tf if os.path.exists(tf) else None

all_trees      = [t for n in df['Orthologs']
                  for t in [get_treefile(n, TREEFILE_DIR)] if t]
filtered_trees = [t for n in df_filtered['Orthologs']
                  for t in [get_treefile(n, TREEFILE_DIR)] if t]

all_tre  = os.path.join(OUTPUT_DIR, "all_orthologs_trees.tre")
filt_tre = os.path.join(OUTPUT_DIR, "filtered_orthologs_trees.tre")

for outfile, treelist in [(all_tre, all_trees), (filt_tre, filtered_trees)]:
    with open(outfile, 'w') as fout:
        for tf in treelist:
            with open(tf) as fin:
                fout.write(fin.read().strip() + "\n")

print(f"  All orthologs : {len(all_trees)} treefiles → {all_tre}")
print(f"  Filtered      : {len(filtered_trees)} treefiles → {filt_tre}")

# ── 9. ASTRAL COMMANDS ────────────────────────────────────────────────────────
all_out  = os.path.join(OUTPUT_DIR, "ASTRAL_all_orthologs.tre")
filt_out = os.path.join(OUTPUT_DIR, "ASTRAL_filtered_orthologs.tre")

print("\n" + "="*65)
print("ASTRAL COMMANDS:")
print("="*65)
print(f"\n# 1. Unfiltered ({len(all_trees)} orthologs):")
print(f"java -jar {ASTRAL_JAR} -i {all_tre} -o {all_out} 2> {all_out}.log")
print(f"\n# 2. Filtered ({len(filtered_trees)} orthologs):")
print(f"java -jar {ASTRAL_JAR} -i {filt_tre} -o {filt_out} 2> {filt_out}.log")
print("="*65)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    ANALYSIS COMPLETE                         ║
╠══════════════════════════════════════════════════════════════╣
║  Individual scatter plots:                                   ║
║  • scatter_TC_vs_Pythia.png                                  ║
║  • scatter_RTC_vs_Pythia.png                                 ║
║  • scatter_TCA_vs_Pythia.png                                 ║
║  • scatter_RTCA_vs_Pythia.png                                ║
║  • scatter_Taxa_vs_Pythia.png                                ║
║  Other plots:                                                ║
║  • correlation_heatmap.png                                   ║
║  • regression_coefficients.png                               ║
║  • filter_decision_plot.png                                  ║
║  Data files:                                                 ║
║  • spearman_correlation_matrix.csv                           ║
║  • filtered_orthologs.csv  ({len(df_filtered):>4} orthologs)              ║
║  • all_orthologs_trees.tre  ({len(all_trees):>4} trees)                ║
║  • filtered_orthologs_trees.tre  ({len(filtered_trees):>4} trees)           ║
╚══════════════════════════════════════════════════════════════╝
""")
