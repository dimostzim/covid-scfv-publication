#!/usr/bin/env python3
"""Extract R4-specific candidate sequences for downstream analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import ensure_dir, load_candidates, write_fasta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract candidate nt_scFv sequences.")
    p.add_argument(
        "--input-csv",
        default="r_output/tables/filt_df.csv",
        help="Input table with nt_scFv and barcode columns.",
    )
    p.add_argument("--present-col", default="barcode09", help="Column that must be > 0")
    p.add_argument("--absent-col", default="barcode12", help="Column that must be == 0")
    p.add_argument("--sequence-col", default="nt_scFv", help="Sequence column name")
    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory for candidates.csv and FASTA.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(Path(args.outdir))

    cand = load_candidates(
        input_csv=Path(args.input_csv),
        present_col=args.present_col,
        absent_col=args.absent_col,
        sequence_col=args.sequence_col,
    )

    csv_path = outdir / "candidates.csv"
    fa_path = outdir / "candidates_nt_scFv.fasta"
    summary_path = outdir / "candidates_summary.txt"
    params_path = outdir / "candidates_params.json"

    cand.to_csv(csv_path, index=False)
    write_fasta(cand, fa_path, id_col="seq_id", seq_col=args.sequence_col)

    counts = cand[args.present_col].to_numpy()
    lengths = cand["seq_len"].to_numpy()

    summary_lines = [
        f"input_csv={Path(args.input_csv).resolve()}",
        f"filter={args.present_col}>0 and {args.absent_col}==0",
        f"n_candidates={len(cand)}",
        f"total_present_reads={int(counts.sum()) if len(counts) else 0}",
        f"median_present_reads={float(np.median(counts)) if len(counts) else 0:.3f}",
        f"max_present_reads={int(counts.max()) if len(counts) else 0}",
        f"len_min={int(lengths.min()) if len(lengths) else 0}",
        f"len_median={float(np.median(lengths)) if len(lengths) else 0:.3f}",
        f"len_max={int(lengths.max()) if len(lengths) else 0}",
        f"candidates_csv={csv_path}",
        f"candidates_fasta={fa_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    params = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "present_col": args.present_col,
        "absent_col": args.absent_col,
        "sequence_col": args.sequence_col,
        "outdir": str(outdir.resolve()),
        "n_candidates": int(len(cand)),
    }
    params_path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {fa_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
