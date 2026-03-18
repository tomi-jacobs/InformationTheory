#!/usr/bin/env python3
"""
RAxML TCA/IC Score Batch Calculator
=====================================
Calculates Tree Certainty (TC), Relative Tree Certainty (RTC),
Tree Certainty Assessment (TCA), and Relative Tree Certainty
Assessment (RTCA) scores for each ortholog gene tree using RAxML
v8.2.12 with the -f i flag under the GTRGAMMA model of evolution.

Each ortholog gene tree is paired with its corresponding 200
standard bootstrap replicate file and processed in batch.

Usage:
    python3 run_raxml_tca.py

Input:
    - Directory containing *.fas-cln.treefile files (gene trees)
    - Directory containing *.fas-cln.boottrees files (bootstrap replicates)
    - Both expected in the same directory (ALIGNMENT_DIR)

Output:
    - Per-gene RAxML output files (*TC_run*)
    - Summary CSV: tca_scores_summary.csv
      Columns: cluster, treefile, TC, RTC, TCA, RTCA

Author: Tomi Jacobs — PhD Candidate, University of Illinois Chicago
"""

import os
import re
import csv
import subprocess
import glob

# ── Configuration ───────────────────────────────────────────────────────────
ALIGNMENT_DIR = "/home/tomi/data/standard-RAxML/InformationTheoryTJ/InfoToCalc"
RAXML_BIN     = "/usr/bin/raxmlHPC-PTHREADS-AVX"
THREADS       = 64
OUTPUT_CSV    = "tca_scores_summary.csv"
RUN_PREFIX    = "TC_run"
# ────────────────────────────────────────────────────────────────────────────

def parse_raxml_info(info_file):
    """Extract TC, RTC, TCA, RTCA from RAxML info file."""
    scores = {"TC": None, "RTC": None, "TCA": None, "RTCA": None}
    if not os.path.exists(info_file):
        return scores
    with open(info_file) as f:
        content = f.read()
    patterns = {
        "TC":   r"Tree certainty for this tree:\s+([\d.eE+-]+)",
        "RTC":  r"Relative tree certainty for this tree:\s+([\d.eE+-]+)",
        "TCA":  r"Tree certainty including missing taxa for this tree:\s+([\d.eE+-]+)",
        "RTCA": r"Relative tree certainty including missing taxa for this tree:\s+([\d.eE+-]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, content)
        if m:
            scores[key] = float(m.group(1))
    return scores

def main():
    treefiles = sorted(glob.glob(os.path.join(ALIGNMENT_DIR, "*.fas-cln.treefile")))
    print(f"Found {len(treefiles)} gene trees to process")

    results = []

    for treefile in treefiles:
        base = os.path.basename(treefile)
        cluster = base.replace(".treefile", "")
        boottrees = treefile.replace(".treefile", ".boottrees")

        if not os.path.exists(boottrees):
            print(f"  [SKIP] No boottrees for {cluster}")
            continue

        run_name = f"{RUN_PREFIX}_{cluster}"

        # Skip if already done
        info_file = f"RAxML_info.{run_name}"
        if os.path.exists(info_file):
            print(f"  [SKIP] Already done: {cluster}")
        else:
            cmd = [
                RAXML_BIN,
                "-T", str(THREADS),
                "-f", "i",
                "-t", treefile,
                "-z", boottrees,
                "-m", "GTRGAMMA",
                "-n", run_name,
                "-p", "12345"
            ]
            try:
                subprocess.run(cmd, check=True,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                print(f"  [OK] {cluster}")
            except subprocess.CalledProcessError:
                print(f"  [ERROR] {cluster}")
                continue

        scores = parse_raxml_info(info_file)
        results.append({
            "cluster": cluster,
            "treefile": treefile,
            "TC":   scores["TC"],
            "RTC":  scores["RTC"],
            "TCA":  scores["TCA"],
            "RTCA": scores["RTCA"],
        })

    # Write summary CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f,
            fieldnames=["cluster", "treefile", "TC", "RTC", "TCA", "RTCA"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! {len(results)} gene trees processed.")
    print(f"Summary written to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
