#!/usr/bin/env python3
"""
Batch-run RAxML IC/TC on matched (.treefile, .boottrees) pairs and compile scores.

Focus: compile TC, RTC, TCA, RTCA for all clusters in a folder.
(phyx + Pythia 2.0 will be added later — placeholders included.)
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ScoreRow:
    stem: str
    treefile: str
    boottrees: str
    ntaxa: Optional[int]
    n_trees_in_set: Optional[int]
    tc: Optional[float]
    rtc: Optional[float]
    tca: Optional[float]
    rtca: Optional[float]
    ic_tree_path: Optional[str]
    raxml_info_path: Optional[str]
    status: str
    message: str


TC_RE = re.compile(r"Tree certainty for this tree:\s*([-\d\.eE]+)")
RTC_RE = re.compile(r"Relative tree certainty for this tree:\s*([-\d\.eE]+)")
TCA_RE = re.compile(r"Tree certainty including all conflicting bipartitions\s*\(TCA\)\s*for this tree:\s*([-\d\.eE]+)")
RTCA_RE = re.compile(r"Relative tree certainty including all conflicting bipartitions\s*\(TCA\)\s*for this tree:\s*([-\d\.eE]+)")
NTAXA_RE = re.compile(r"found\s+(\d+)\s+taxa,\s+reference tree has\s+(\d+)\s+taxa", re.IGNORECASE)
NSET_RE = re.compile(r"Tree\s+set\s+contains\s+(\d+)\s+trees", re.IGNORECASE)  # may not appear in all builds
FOUND1_RE = re.compile(r"Found\s+(\d+)\s+tree\s+in\s+File", re.IGNORECASE)


def safe_runname(stem: str) -> str:
    # RAxML run name should be filesystem-friendly
    # keep letters, digits, underscore, dash, dot; replace others with underscore
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:180]


def parse_scores(info_text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[int], Optional[int]]:
    def grab_float(rx: re.Pattern) -> Optional[float]:
        m = rx.search(info_text)
        return float(m.group(1)) if m else None

    tc = grab_float(TC_RE)
    rtc = grab_float(RTC_RE)
    tca = grab_float(TCA_RE)
    rtca = grab_float(RTCA_RE)

    # taxa: take the first "found X taxa" line, typically appears many times
    ntaxa = None
    m = NTAXA_RE.search(info_text)
    if m:
        # both numbers should match; keep the reference value
        ntaxa = int(m.group(2))

    # number of trees: different builds log it differently; try a couple patterns
    ntrees = None
    m = NSET_RE.search(info_text)
    if m:
        ntrees = int(m.group(1))
    else:
        # fallback: sometimes you only see "Tree 1: ... Tree N: ..." but that’s hard to parse reliably
        # another fallback: some outputs show "Found 1 tree in File <treefile>" (that's the reference tree)
        # For the evaluation set, if not available, leave None.
        pass

    return tc, rtc, tca, rtca, ntaxa, ntrees


def run_raxml(
    raxml_bin: str,
    threads: int,
    model: str,
    treefile: Path,
    boottrees: Path,
    outdir: Path,
    runname: str,
    timeout: Optional[int] = None,
) -> Tuple[int, str, str, Path, Path, Path]:
    """
    Runs RAxML -f i in outdir so outputs are contained.
    Returns (returncode, stdout, stderr, info_path, ic_tree_path, workdir)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # RAxML writes output files into current working directory.
    # We run in outdir and point -t/-z to absolute paths.
    cmd = [
        raxml_bin,
        "-T", str(threads),
        "-f", "i",
        "-t", str(treefile),
        "-z", str(boottrees),
        "-m", model,
        "-n", runname,
    ]

    p = subprocess.run(
        cmd,
        cwd=str(outdir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )

    info_path = outdir / f"RAxML_info.{runname}"
    ic_tree_path = outdir / f"RAxML_IC_Score_BranchLabels.{runname}"
    # Other outputs may exist; we key off these two.
    return p.returncode, p.stdout, p.stderr, info_path, ic_tree_path, outdir


def find_pairs(folder: Path) -> List[Tuple[str, Path, Path]]:
    """
    Match by stem:
      <stem>.treefile  with <stem>.boottrees
    """
    treefiles = {p.stem: p for p in folder.glob("*.treefile")}
    boottrees = {p.stem: p for p in folder.glob("*.boottrees")}

    stems = sorted(set(treefiles.keys()) & set(boottrees.keys()))
    pairs = [(stem, treefiles[stem], boottrees[stem]) for stem in stems]
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile RAxML TC/RTC/TCA/RTCA scores for all matched treefile+boottrees pairs.")
    ap.add_argument(
        "--indir",
        default="~/data/standard-RAxML/InformationTheoryTJ/InfoToCalc",
        help="Folder containing *.treefile and *.boottrees (default: your InfoToCalc path).",
    )
    ap.add_argument(
        "--outdir",
        default="~/data/standard-RAxML/InformationTheoryTJ/raxml_TC_runs",
        help="Folder to write RAxML outputs + compiled CSV.",
    )
    ap.add_argument(
        "--csv",
        default="raxml_TC_scores.csv",
        help="Output CSV filename (written inside --outdir unless absolute path).",
    )
    ap.add_argument(
        "--raxml",
        default="/usr/bin/raxmlHPC-PTHREADS-AVX",
        help="Path to RAxML binary.",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Threads for RAxML (-T). Avoid huge values on login nodes. Default=8.",
    )
    ap.add_argument(
        "--model",
        default="GTRGAMMA",
        help="RAxML model (-m). Default=GTRGAMMA.",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout (seconds) per run.",
    )
    args = ap.parse_args()

    indir = Path(os.path.expanduser(args.indir)).resolve()
    outdir = Path(os.path.expanduser(args.outdir)).resolve()
    raxml_bin = args.raxml

    if not indir.exists():
        raise SystemExit(f"ERROR: indir does not exist: {indir}")

    if shutil.which(raxml_bin) is None and not Path(raxml_bin).exists():
        raise SystemExit(f"ERROR: RAxML binary not found: {raxml_bin}")

    pairs = find_pairs(indir)
    if not pairs:
        raise SystemExit(f"ERROR: No matched pairs found in {indir} (need both *.treefile and *.boottrees with same stem).")

    # Placeholder for future additions:
    # - phyx
    # - Pythia 2.0 (new model/new features)
    # Not used yet.

    rows: List[ScoreRow] = []

    for stem, treefile, boottrees in pairs:
        runname = safe_runname(stem)
        run_outdir = outdir / runname

        try:
            rc, so, se, info_path, ic_tree_path, _wd = run_raxml(
                raxml_bin=raxml_bin,
                threads=args.threads,
                model=args.model,
                treefile=treefile,
                boottrees=boottrees,
                outdir=run_outdir,
                runname=runname,
                timeout=args.timeout,
            )

            # Prefer parsing the info file (more stable than stdout).
            info_text = ""
            if info_path.exists():
                info_text = info_path.read_text()
            else:
                # fallback: use stdout+stderr
                info_text = (so or "") + "\n" + (se or "")

            tc, rtc, tca, rtca, ntaxa, ntrees = parse_scores(info_text)

            status = "OK" if rc == 0 and (tc is not None and rtc is not None and tca is not None and rtca is not None) else "WARN"
            msg = "completed" if status == "OK" else "ran but could not parse all scores (check RAxML_info)"

            rows.append(
                ScoreRow(
                    stem=stem,
                    treefile=str(treefile),
                    boottrees=str(boottrees),
                    ntaxa=ntaxa,
                    n_trees_in_set=ntrees,
                    tc=tc,
                    rtc=rtc,
                    tca=tca,
                    rtca=rtca,
                    ic_tree_path=str(ic_tree_path) if ic_tree_path.exists() else None,
                    raxml_info_path=str(info_path) if info_path.exists() else None,
                    status=status,
                    message=msg if rc == 0 else f"RAxML failed rc={rc}: {se.strip()[:300]}",
                )
            )

        except subprocess.TimeoutExpired:
            rows.append(
                ScoreRow(
                    stem=stem,
                    treefile=str(treefile),
                    boottrees=str(boottrees),
                    ntaxa=None,
                    n_trees_in_set=None,
                    tc=None,
                    rtc=None,
                    tca=None,
                    rtca=None,
                    ic_tree_path=None,
                    raxml_info_path=None,
                    status="FAIL",
                    message="timeout",
                )
            )
        except Exception as e:
            rows.append(
                ScoreRow(
                    stem=stem,
                    treefile=str(treefile),
                    boottrees=str(boottrees),
                    ntaxa=None,
                    n_trees_in_set=None,
                    tc=None,
                    rtc=None,
                    tca=None,
                    rtca=None,
                    ic_tree_path=None,
                    raxml_info_path=None,
                    status="FAIL",
                    message=str(e),
                )
            )

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = outdir / csv_path

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "stem",
            "treefile",
            "boottrees",
            "ntaxa",
            "n_trees_in_set",
            "TC",
            "RTC",
            "TCA",
            "RTCA",
            "ic_tree_with_branchlabels",
            "raxml_info",
            "status",
            "message",
        ])
        for r in rows:
            w.writerow([
                r.stem,
                r.treefile,
                r.boottrees,
                r.ntaxa if r.ntaxa is not None else "",
                r.n_trees_in_set if r.n_trees_in_set is not None else "",
                r.tc if r.tc is not None else "",
                r.rtc if r.rtc is not None else "",
                r.tca if r.tca is not None else "",
                r.rtca if r.rtca is not None else "",
                r.ic_tree_path or "",
                r.raxml_info_path or "",
                r.status,
                r.message,
            ])

    ok = sum(1 for r in rows if r.status == "OK")
    warn = sum(1 for r in rows if r.status == "WARN")
    fail = sum(1 for r in rows if r.status == "FAIL")
    print(f"Done. OK={ok} WARN={warn} FAIL={fail}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
