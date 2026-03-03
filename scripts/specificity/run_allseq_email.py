#!/usr/bin/env python3
"""All-sequence (full candidate set) email-ready plots and summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from common import ecdf, ensure_dir, kmer_matrix, pca_from_features, topk_neighbors_cosine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate all-sequence colored PCA + heatmap + summary plots.")
    p.add_argument("--candidates-csv", required=True, help="Path to candidates.csv")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--present-col", default="barcode09", help="Abundance column")
    p.add_argument("--kmer", type=int, default=5, help="k-mer size")
    p.add_argument("--n-clusters", type=int, default=20, help="MiniBatchKMeans clusters")
    p.add_argument("--heatmap-bins", type=int, default=140, help="Number of bins for all-seq heatmap")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def _cluster_colors(labels: np.ndarray) -> np.ndarray:
    uniq = sorted(np.unique(labels).tolist())
    cmap = plt.get_cmap("tab20")
    color_map = {u: cmap(i % 20) for i, u in enumerate(uniq)}
    return np.array([color_map[x] for x in labels], dtype=object)


def _binned_similarity_heatmap(
    x: np.ndarray,
    labels: np.ndarray,
    pc1: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    n_bins = max(10, min(n_bins, n))
    order = np.lexsort((pc1, labels))
    chunks = np.array_split(order, n_bins)

    bin_vecs = []
    bin_major_labels = []
    for idx in chunks:
        if idx.size == 0:
            continue
        m = x[idx].mean(axis=0)
        norm = np.linalg.norm(m)
        if norm > 0:
            m = m / norm
        bin_vecs.append(m)

        lab = labels[idx]
        binc = np.bincount(lab.astype(int))
        bin_major_labels.append(int(np.argmax(binc)))

    if not bin_vecs:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.int32)

    b = np.vstack(bin_vecs).astype(np.float32)
    sim = b @ b.T
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)

    return sim, np.array(bin_major_labels, dtype=np.int32)


def plot_pca(out: Path, pcs: np.ndarray, var: np.ndarray, labels: np.ndarray, counts: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = _cluster_colors(labels)
    sizes = 8 + 12 * np.log10(counts.astype(float) + 1.0)
    ax.scatter(pcs[:, 0], pcs[:, 1], c=colors, s=sizes, alpha=0.72, edgecolors="none")
    ax.set_title("All candidates PCA (colored by k-mer clusters)")
    ax.set_xlabel(f"PC1 ({var[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.2f}% variance)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_ecdf(out: Path, nearest: np.ndarray) -> None:
    x, y = ecdf(nearest[np.isfinite(nearest)])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y, lw=2, color="#be123c")
    ax.set_title("All candidates nearest-neighbor similarity ECDF")
    ax.set_xlabel("Nearest-neighbor cosine similarity")
    ax.set_ylabel("Cumulative fraction")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_rank(out: Path, cluster_sizes: pd.DataFrame) -> None:
    vals = cluster_sizes["cluster_size"].sort_values(ascending=False).to_numpy()
    ranks = np.arange(1, len(vals) + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ranks, vals, marker="o", ms=3, lw=1.3, color="#065f46")
    ax.set_yscale("log")
    ax.set_title("All candidates cluster size rank")
    ax.set_xlabel("Cluster rank")
    ax.set_ylabel("Cluster size (log scale)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_heatmap(out: Path, sim: np.ndarray, bin_labels: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sim, cmap="inferno", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine similarity (bin-averaged)")
    ax.set_title("All candidates similarity heatmap (binned from full set)")
    ax.set_xlabel("Ordered bins (cluster then PC1)")
    ax.set_ylabel("Ordered bins (cluster then PC1)")
    ax.set_xticks([])
    ax.set_yticks([])

    if bin_labels.size > 1:
        changes = np.where(np.diff(bin_labels) != 0)[0]
        for c in changes:
            pos = c + 0.5
            ax.axhline(pos, color="white", lw=0.55, alpha=0.55)
            ax.axvline(pos, color="white", lw=0.55, alpha=0.55)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_4panel(
    out: Path,
    pcs: np.ndarray,
    var: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    nearest: np.ndarray,
    cluster_sizes: pd.DataFrame,
    sim_bin: np.ndarray,
    bin_labels: np.ndarray,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # A: PCA
    ax = axs[0, 0]
    colors = _cluster_colors(labels)
    sizes = 8 + 12 * np.log10(counts.astype(float) + 1.0)
    ax.scatter(pcs[:, 0], pcs[:, 1], c=colors, s=sizes, alpha=0.72, edgecolors="none")
    ax.set_title("A) PCA (all candidates, colored by clusters)")
    ax.set_xlabel(f"PC1 ({var[0] * 100:.2f}% var)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.2f}% var)")
    ax.grid(alpha=0.2)

    # B: ECDF
    ax = axs[0, 1]
    x, y = ecdf(nearest[np.isfinite(nearest)])
    ax.plot(x, y, lw=2, color="#be123c")
    ax.set_title("B) Nearest-neighbor similarity ECDF")
    ax.set_xlabel("Nearest-neighbor cosine similarity")
    ax.set_ylabel("Cumulative fraction")
    ax.grid(alpha=0.2)

    # C: Cluster rank
    ax = axs[1, 0]
    vals = cluster_sizes["cluster_size"].sort_values(ascending=False).to_numpy()
    ranks = np.arange(1, len(vals) + 1)
    ax.plot(ranks, vals, marker="o", ms=3, lw=1.3, color="#065f46")
    ax.set_yscale("log")
    ax.set_title("C) Cluster size rank")
    ax.set_xlabel("Cluster rank")
    ax.set_ylabel("Cluster size (log scale)")
    ax.grid(alpha=0.25)

    # D: Heatmap
    ax = axs[1, 1]
    im = ax.imshow(sim_bin, cmap="inferno", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_title("D) Similarity heatmap (binned from all sequences)")
    ax.set_xlabel("Ordered bins")
    ax.set_ylabel("Ordered bins")
    ax.set_xticks([])
    ax.set_yticks([])
    if bin_labels.size > 1:
        changes = np.where(np.diff(bin_labels) != 0)[0]
        for c in changes:
            pos = c + 0.5
            ax.axhline(pos, color="white", lw=0.55, alpha=0.55)
            ax.axvline(pos, color="white", lw=0.55, alpha=0.55)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity (bin-averaged)")

    fig.suptitle("R4 H460 PCNB-specific sequences: all-candidate summary", y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(Path(args.outdir))
    plot_dir = ensure_dir(outdir / "plots")

    cand = pd.read_csv(args.candidates_csv)
    if len(cand) == 0:
        raise SystemExit("No candidates found.")
    if args.present_col not in cand.columns:
        raise SystemExit(f"Missing present column: {args.present_col}")

    seqs = cand["nt_scFv"].astype(str).tolist()
    x = kmer_matrix(seqs, k=args.kmer, l2_normalize=True)

    pcs, var = pca_from_features(x, n_components=2)

    km = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        n_init=10,
        batch_size=2048,
        max_iter=300,
    )
    labels = km.fit_predict(x).astype(np.int32)

    _, nearest = topk_neighbors_cosine(x, top_k=1, chunk_size=512)

    cluster_sizes = (
        pd.DataFrame({"cluster_id": labels})
        .groupby("cluster_id", as_index=False)
        .size()
        .rename(columns={"size": "cluster_size"})
        .sort_values("cluster_size", ascending=False)
        .reset_index(drop=True)
    )

    sim_bin, bin_labels = _binned_similarity_heatmap(
        x=x,
        labels=labels,
        pc1=pcs[:, 0],
        n_bins=args.heatmap_bins,
    )

    out_assign = cand[["seq_id", "nt_scFv", "seq_len", args.present_col]].copy()
    out_assign["cluster_id"] = labels
    out_assign["PC1"] = pcs[:, 0]
    out_assign["PC2"] = pcs[:, 1]
    out_assign["nearest_similarity"] = nearest
    out_assign.to_csv(outdir / "allseq_assignments.csv", index=False)
    cluster_sizes.to_csv(outdir / "allseq_cluster_sizes.csv", index=False)
    pd.DataFrame(sim_bin).to_csv(outdir / "allseq_similarity_heatmap_binned.csv", index=False)

    counts = cand[args.present_col].to_numpy(dtype=float)
    plot_pca(plot_dir / "plot_01_allseq_pca_colored.png", pcs=pcs, var=var, labels=labels, counts=counts)
    plot_ecdf(plot_dir / "plot_02_allseq_nn_similarity_ecdf.png", nearest=nearest)
    plot_rank(plot_dir / "plot_03_allseq_cluster_size_rank.png", cluster_sizes=cluster_sizes)
    plot_heatmap(plot_dir / "plot_04_allseq_similarity_heatmap_binned.png", sim=sim_bin, bin_labels=bin_labels)
    plot_4panel(
        plot_dir / "plot_00_allseq_summary_4panel.png",
        pcs=pcs,
        var=var,
        labels=labels,
        counts=counts,
        nearest=nearest,
        cluster_sizes=cluster_sizes,
        sim_bin=sim_bin,
        bin_labels=bin_labels,
    )

    summary_lines = [
        "pipeline=allseq_email",
        f"candidates_csv={Path(args.candidates_csv).resolve()}",
        f"n_candidates={len(cand)}",
        f"kmer={args.kmer}",
        f"n_clusters={args.n_clusters}",
        f"heatmap_bins={sim_bin.shape[0]}",
        f"pca_var_pc1={float(var[0]):.6f}",
        f"pca_var_pc2={float(var[1]):.6f}",
        f"nearest_similarity_median={float(np.median(nearest[np.isfinite(nearest)])):.6f}",
        f"nearest_similarity_q90={float(np.quantile(nearest[np.isfinite(nearest)], 0.9)):.6f}",
        f"outputs_dir={outdir.resolve()}",
    ]
    (outdir / "summary_allseq.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    params = vars(args)
    params["n_candidates"] = int(len(cand))
    (outdir / "params_allseq.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

    print(f"All-sequence email plots complete: {outdir}")


if __name__ == "__main__":
    main()
