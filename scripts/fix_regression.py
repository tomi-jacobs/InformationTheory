import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os

CSV_PATH   = os.path.expanduser("~/data/standard-RAxML/InformationTheoryTJ/Correlation_TreeStats.csv")
OUTPUT_DIR = os.path.expanduser("~/data/standard-RAxML/InformationTheoryTJ/Analysis_Output_3rdTrial")
GREEN = '#27ae60'; CRIMSON = '#c0392b'

df = pd.read_csv(CSV_PATH); df.columns = df.columns.str.strip()
predictors = ['TC', 'RTC', 'TCA', 'RTCA', 'Taxa']
X = StandardScaler().fit_transform(df[predictors].values)
y = df['Pythia scores'].values
reg = LinearRegression().fit(X, y)
r2 = r2_score(y, reg.predict(X))
r2_adj = 1-(1-r2)*(len(y)-1)/(len(y)-len(predictors)-1)

fig, ax = plt.subplots(figsize=(12, 5))
colors = [GREEN if c > 0 else CRIMSON for c in reg.coef_]
bars = ax.barh(predictors, reg.coef_, color=colors, edgecolor='white', height=0.55)
ax.axvline(0, color='black', linewidth=1.8)

ax.set_xlabel("Standardized Regression Coefficient", fontsize=14, fontweight='bold')
ax.set_title(f"Multiple Regression: Predicting Pythia Difficulty\nR\u00b2 = {r2:.3f}  |  Adjusted R\u00b2 = {r2_adj:.3f}", fontsize=16, fontweight='bold')
ax.set_yticks(range(len(predictors)))
ax.set_yticklabels(predictors, fontsize=14, fontweight='bold')
ax.tick_params(axis='y', pad=12)
for label in ax.get_xticklabels()+ax.get_yticklabels():
    label.set_fontweight('bold'); label.set_fontsize(13)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

for bar, val, name in zip(bars, reg.coef_, predictors):
    bar_top = bar.get_y() + bar.get_height()
    bar_mid = bar.get_y() + bar.get_height()/2
    if name == 'RTC':
        ax.text(-0.12, bar_top + 0.05, f'{val:+.3f}',
                va='bottom', ha='center', fontsize=12, fontweight='bold', color='black')
    elif name == 'RTCA':
        ax.text(0.009, bar_top + 0.05, f'{val:+.3f}',
                va='bottom', ha='center', fontsize=12, fontweight='bold', color='black')
    elif abs(val) > 0.05:
        ax.text(val/2, bar_mid, f'{val:+.3f}',
                va='center', ha='center', fontsize=12, fontweight='bold', color='white')
    else:
        gap = 0.003
        ax.text(val+(gap if val>=0 else -gap), bar_mid, f'{val:+.3f}',
                va='center', ha='left' if val>=0 else 'right',
                fontsize=12, fontweight='bold', color='black')

ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
fig.subplots_adjust(left=0.15, right=0.92, top=0.88, bottom=0.15)
plt.savefig(os.path.join(OUTPUT_DIR, "regression_coefficients.png"), dpi=150)
print("Saved!")
