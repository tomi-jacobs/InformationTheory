#!/usr/bin/env python3
import os
import sys
import csv
import subprocess
import re
from pathlib import Path

# ── EDIT THESE IF NEEDED ─────────────────────────────────────────────────────
INPUT_DIR  = os.path.expanduser("~/data/standard-RAxML/InformationTheoryTJ/InfoToCalc")
RAXML_NG   = "/home/tomi/data/raxml-ng/raxml-ng/bin/raxml-ng"
OUTPUT_CSV = "pythia_scores.csv"
# ─────────────────────────────────────────────────────────────────────────────

ALIGNMENT_PATTERN = re.compile(r"^cluster\d+.*\.fas-cln$")

def find_alignment_files(directory):
    return sorted(
        f for f in Path(directory).iterdir()
        if f.is_file() and ALIGNMENT_PATTERN.match(f.name)
    )

def run_pythia(alignment_path):
    try:
        result = subprocess.run(
            ["pythia", "-m", str(alignment_path), "-r", RAXML_NG],
            capture_output=True, text=True, timeout=600
        )

        # First try: read the .pythia.csv file Pythia writes
        csv_out = Path(str(alignment_path) + ".pythia.csv")
        if csv_out.exists():
            with open(csv_out) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key in row:
                        if "difficulty" in key.lower() or "prediction" in key.lower():
                            try:
                                score = float(row[key])
                                if 0.0 <= score <= 1.0:
                                    return score, None
                            except ValueError:
                                continue
                    # Try any float value in the row if headers unclear
                    for key, val in row.items():
                        try:
                            score = float(val)
                            if 0.0 <= score <= 1.0:
                                return score, None
                        except ValueError:
                            continue

        # Second try: parse stderr for the difficulty line
        combined = result.stdout + "\n" + result.stderr
        match = re.search(r"predicted difficulty.*?:\s*([0-9]\.[0-9]+)", combined, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            if 0.0 <= score <= 1.0:
                return score, None

        return None, f"STDOUT:[{result.stdout[-300:]}] STDERR:[{result.stderr[-300:]}]"

    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except FileNotFoundError:
        return None, "pythia not found in PATH"
    except Exception as e:
        return None, str(e)

def extract_cluster_name(filepath):
    name = filepath.name
    for suffix in ['.fas-cln', '-cln']:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name

def main():
    print(f"Scanning: {INPUT_DIR}")
    aln_files = find_alignment_files(INPUT_DIR)
    print(f"Found {len(aln_files)} cluster alignment files\n")

    if not aln_files:
        print("No alignment files found.")
        sys.exit(1)

    # Sanity check on first file
    print(f"=== Sanity check on: {aln_files[0].name} ===")
    score, error = run_pythia(aln_files[0])
    if score is not None:
        print(f"✓ Success! Difficulty = {score:.4f}")
    else:
        print(f"✗ Failed: {error}")
        sys.exit(1)
    print("=== End sanity check ===\n")

    answer = input("Proceed with all files? (yes/no): ").strip().lower()
    if answer != "yes":
        sys.exit(0)

    results = []
    for i, aln_file in enumerate(aln_files, 1):
        cluster = extract_cluster_name(aln_file)
        print(f"[{i}/{len(aln_files)}] {aln_file.name}", end=" ... ", flush=True)

        score, error = run_pythia(aln_file)

        if score is not None:
            print(f"{score:.4f}")
            results.append({"cluster": cluster, "file": aln_file.name,
                            "pythia_difficulty": score, "error": ""})
        else:
            print(f"ERROR: {error}")
            results.append({"cluster": cluster, "file": aln_file.name,
                            "pythia_difficulty": "", "error": error})

        # Save progress every 100 files in case of interruption
        if i % 100 == 0:
            with open(OUTPUT_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["cluster", "file", "pythia_difficulty", "error"])
                writer.writeheader()
                writer.writerows(results)
            print(f"  → Progress saved ({i}/{len(aln_files)})")

    # Final save
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster", "file", "pythia_difficulty", "error"])
        writer.writeheader()
        writer.writerows(results)

    ok = sum(1 for r in results if r["pythia_difficulty"] != "")
    print(f"\nDone! {ok}/{len(results)} successful → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()