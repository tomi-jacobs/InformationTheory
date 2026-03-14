#!/usr/bin/env python3
"""
Visualize and compare ASTRAL species trees
- All orthologs tree vs Filtered orthologs tree
- RF distance annotation
Author: Tomi
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
OUTPUT_DIR  = os.path.expanduser("~/data/standard-RAxML/InformationTheoryTJ/Analysis_Output_3rdTrial")
ALL_TREE    = os.path.expanduser("~/data/standard-RAxML/InformationTheoryTJ/Analysis_Output/ASTRAL_all_orthologs.tre")
FILT_TREE   = os.path.expanduser("~/data/standard-RAxML/InformationTheoryTJ/Analysis_Output/ASTRAL_filtered_orthologs.tre")
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    from ete3 import Tree, TreeStyle, NodeStyle, TextFace, AttrFace
    ETE3 = True
except ImportError:
    ETE3 = False
    print("ete3 not available, using matplotlib text-based visualization")

def parse_newick(filepath):
    with open(filepath) as f:
        return f.read().strip()

def draw_tree_matplotlib(newick, title, ax, color):
    """Draw a simple cladogram using ete3 + matplotlib."""
    try:
        t = Tree(newick)
        
        # Get leaf order
        leaves = t.get_leaves()
        leaf_names = [l.name for l in leaves]
        n = len(leaf_names)
        
        # Assign y positions to leaves
        y_pos = {name: i for i, name in enumerate(leaf_names)}
        
        def get_y(node):
            if node.is_leaf():
                return y_pos[node.name]
            else:
                children_y = [get_y(c) for c in node.children]
                return sum(children_y) / len(children_y)
        
        def draw_node(node, x, ax):
            y = get_y(node)
            if not node.is_leaf():
                children_y = [get_y(c) for c in node.children]
                # Vertical line connecting children
                ax.plot([x, x], [min(children_y), max(children_y)],
                        color=color, linewidth=1.2)
                for child in node.children:
                    child_y = get_y(child)
                    child_x = x + 1
                    # Horizontal line to child
                    ax.plot([x, child_x], [child_y, child_y],
                            color=color, linewidth=1.2)
                    draw_node(child, child_x, ax)
            else:
                ax.text(x + 0.1, y, node.name,
                        va='center', ha='left',
                        fontsize=7, fontweight='bold', color='black')

        draw_node(t, 0, ax)
        
        ax.set_xlim(-0.5, 12)
        ax.set_ylim(-1, n)
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not render tree:\n{str(e)[:100]}",
                transform=ax.transAxes, ha='center', va='center', fontsize=9)
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')

# ── READ TREES ────────────────────────────────────────────────────────────────
print("Reading trees...")
all_nwk  = parse_newick(ALL_TREE)
filt_nwk = parse_newick(FILT_TREE)

# ── PLOT ──────────────────────────────────────────────────────────────────────
print("Plotting trees...")
fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(1, 2, figure=fig, wspace=0.15)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

draw_tree_matplotlib(all_nwk,
                     "ASTRAL Species Tree\nAll Orthologs (n=4,845)",
                     ax1, color='#1a5276')

draw_tree_matplotlib(filt_nwk,
                     "ASTRAL Species Tree\nFiltered Orthologs (n=1,095)",
                     ax2, color='#c0392b')

# RF distance annotation
fig.text(0.5, 0.02,
         "RF Distance = 0  |  Topologies are identical  |  "
         "Filter: Pythia difficulty < 0.5  &  |TCA| ≥ 75th percentile",
         ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#eafaf1',
                   edgecolor='#27ae60', alpha=0.9))

fig.suptitle("Comparison of ASTRAL Species Trees: Filtered vs Unfiltered Orthologs",
             fontsize=16, fontweight='bold', y=1.01)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "ASTRAL_tree_comparison.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {path}")

# ── ALSO SAVE INDIVIDUAL TREES ────────────────────────────────────────────────
for nwk, title, fname, color in [
    (all_nwk,  "ASTRAL Species Tree — All Orthologs (n=4,845)",
     "ASTRAL_all_tree.png", '#1a5276'),
    (filt_nwk, "ASTRAL Species Tree — Filtered Orthologs (n=1,095)",
     "ASTRAL_filtered_tree.png", '#c0392b'),
]:
    fig, ax = plt.subplots(figsize=(9, 12))
    draw_tree_matplotlib(nwk, title, ax, color)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")

print("\nDone! Files saved:")
print(f"  • ASTRAL_tree_comparison.png")
print(f"  • ASTRAL_all_tree.png")
print(f"  • ASTRAL_filtered_tree.png")
