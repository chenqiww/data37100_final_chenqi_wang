#!/usr/bin/env python3
"""
final/tools/visualize_samples.py

Automatic sample grid visualizer for final project outputs.

Supports:
- A single run directory containing PNG grids (e.g., sample grids saved by diffusion_baseline.py)
- A results.csv manifest produced by --grid runs (diffusion or transformer starter)

Examples (from repo root):
  python final/tools/visualize_samples.py --run-dir ./untrack/outputs/final/diffusion/ds-mnist_T-200_target-eps_b2-0.02_ch-64
  python final/tools/visualize_samples.py --results-csv ./untrack/outputs/final/diffusion/results.csv
  python final/tools/visualize_samples.py --results-csv ./untrack/outputs/final/transformer/results.csv --show-text

Outputs:
- Saves a contact sheet PNG alongside the inputs (or in --out-dir).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


def find_pngs(run_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    if (run_dir / "samples").exists():
        candidates += sorted((run_dir / "samples").glob("*.png"))
    candidates += sorted(run_dir.glob("*.png"))
    # prioritize "grid" or "sample" names, then latest modified
    candidates = sorted(candidates, key=lambda p: (("grid" not in p.name.lower() and "sample" not in p.name.lower()), p.stat().st_mtime))
    return candidates


def read_results_csv(p: Path) -> List[Path]:
    run_dirs: List[Path] = []
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "run_dir" not in (r.fieldnames or []):
            raise ValueError(f"results.csv missing 'run_dir' column: {p}")
        for row in r:
            run_dirs.append(Path(row["run_dir"]))
    return run_dirs


def load_image(path: Path):
    import matplotlib.image as mpimg
    return mpimg.imread(str(path))


def make_contact_sheet(images: List[Path], titles: List[str], out_path: Path, ncols: int = 3):
    n = len(images)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols

    plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i, (img_path, title) in enumerate(zip(images, titles), start=1):
        ax = plt.subplot(nrows, ncols, i)
        ax.imshow(load_image(img_path))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"[Saved] {out_path}")


def maybe_show_text(run_dir: Path, max_chars: int = 1200):
    sample_txt = run_dir / "sample.txt"
    if sample_txt.exists():
        txt = sample_txt.read_text(encoding="utf-8", errors="ignore")
        txt = txt[:max_chars] + ("\n...\n" if len(txt) > max_chars else "")
        print("\n" + "=" * 80)
        print(f"[sample.txt] {run_dir}")
        print("=" * 80)
        print(txt)
        print("=" * 80 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Final Project - Automatic sample grid visualizer")
    p.add_argument("--run-dir", type=str, default="", help="Single run directory")
    p.add_argument("--results-csv", type=str, default="", help="results.csv produced by --grid")
    p.add_argument("--out-dir", type=str, default="", help="Optional output directory (default: alongside inputs)")
    p.add_argument("--ncols", type=int, default=3)
    p.add_argument("--show-text", action="store_true", help="For transformer runs: print sample.txt to terminal")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else None

    if args.run_dir and args.results_csv:
        raise SystemExit("Provide only one of --run-dir or --results-csv")

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        pngs = find_pngs(run_dir)
        if not pngs:
            raise FileNotFoundError(f"No PNG grids found under: {run_dir} (or {run_dir}/samples)")
        titles = [p.name for p in pngs]
        out_path = (out_dir or run_dir) / "contact_sheet.png"
        make_contact_sheet(pngs, titles, out_path, ncols=args.ncols)
        if args.show_text:
            maybe_show_text(run_dir)
        return 0

    if args.results_csv:
        res = Path(args.results_csv).expanduser()
        run_dirs = read_results_csv(res)
        if not run_dirs:
            raise FileNotFoundError(f"No run_dir entries in: {res}")
        imgs: List[Path] = []
        titles: List[str] = []
        for rd in run_dirs:
            pngs = find_pngs(rd)
            if pngs:
                # take the "best" candidate (first after our sorting)
                imgs.append(pngs[0])
                titles.append(rd.name)
            if args.show_text:
                maybe_show_text(rd)
        if not imgs:
            raise FileNotFoundError("No PNG grids found for any run_dir listed in results.csv")
        default_out = out_dir or res.parent
        out_path = default_out / "grid_contact_sheet.png"
        make_contact_sheet(imgs, titles, out_path, ncols=args.ncols)
        return 0

    raise SystemExit("Provide --run-dir or --results-csv")


if __name__ == "__main__":
    raise SystemExit(main())
