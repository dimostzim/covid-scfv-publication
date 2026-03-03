#!/usr/bin/env python3
"""Shared utilities for specificity analyses (pure numpy/pandas/matplotlib stack)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DNA_MAP = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


def timestamp_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_candidates(
    input_csv: Path,
    present_col: str,
    absent_col: str,
    sequence_col: str = "nt_scFv",
) -> pd.DataFrame:
    usecols = [sequence_col, present_col, absent_col]
    df = pd.read_csv(input_csv, usecols=usecols)
    df = df[df[sequence_col].notna() & (df[sequence_col] != "")].copy()

    agg = (
        df.groupby(sequence_col, as_index=False)[[present_col, absent_col]]
        .sum()
        .sort_values([present_col, sequence_col], ascending=[False, True])
        .reset_index(drop=True)
    )

    cand = agg[(agg[present_col] > 0) & (agg[absent_col] == 0)].copy().reset_index(drop=True)
    cand["seq_id"] = [f"seq_{i + 1:05d}" for i in range(len(cand))]
    cand["seq_len"] = cand[sequence_col].str.len()
    return cand[["seq_id", sequence_col, "seq_len", present_col, absent_col]]


def write_fasta(df: pd.DataFrame, path: Path, id_col: str = "seq_id", seq_col: str = "nt_scFv") -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in df.itertuples(index=False):
            handle.write(f">{getattr(row, id_col)}\n")
            handle.write(f"{getattr(row, seq_col)}\n")


def _kmer_index(kmer: str) -> int | None:
    idx = 0
    for c in kmer:
        v = DNA_MAP.get(c)
        if v is None:
            return None
        idx = (idx << 2) | v
    return idx


def kmer_matrix(seqs: Iterable[str], k: int = 5, l2_normalize: bool = True) -> np.ndarray:
    n = len(seqs)
    dim = 4**k
    x = np.zeros((n, dim), dtype=np.float32)

    for i, seq in enumerate(seqs):
        s = str(seq).upper()
        if len(s) < k:
            continue
        vec = x[i]
        for j in range(0, len(s) - k + 1):
            idx = _kmer_index(s[j : j + k])
            if idx is not None:
                vec[idx] += 1.0
        if l2_normalize:
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
    return x


def pca_from_features(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    n, p = x.shape
    if n < 2:
        pcs = np.zeros((n, n_components), dtype=np.float32)
        var = np.zeros((n_components,), dtype=np.float32)
        return pcs, var

    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    m = min(n_components, eigvecs.shape[1])
    comp = eigvecs[:, :m]
    pcs = xc @ comp

    total = float(np.clip(eigvals.sum(), 1e-12, None))
    var = eigvals[:m] / total

    if m < n_components:
        pad = np.zeros((n, n_components - m), dtype=pcs.dtype)
        pcs = np.hstack([pcs, pad])
        var = np.concatenate([var, np.zeros((n_components - m,), dtype=var.dtype)])

    return pcs.astype(np.float32), var.astype(np.float32)


def pca_from_square_matrix(mtx: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    if mtx.ndim != 2 or mtx.shape[0] != mtx.shape[1]:
        raise ValueError("mtx must be square")
    return pca_from_features(mtx, n_components=n_components)


def topk_neighbors_cosine(
    x: np.ndarray,
    top_k: int = 25,
    chunk_size: int = 512,
) -> tuple[pd.DataFrame, np.ndarray]:
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    n = x.shape[0]
    if n == 0:
        return pd.DataFrame(columns=["query_idx", "neighbor_idx", "rank", "similarity"]), np.array([])

    top_k = max(1, min(top_k, n - 1))
    nearest = np.full((n,), -np.inf, dtype=np.float32)

    out_query = []
    out_neighbor = []
    out_rank = []
    out_sim = []

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sims = x[start:end] @ x.T

        rows = np.arange(start, end)
        sims[np.arange(end - start), rows] = -np.inf

        kidx = np.argpartition(sims, -top_k, axis=1)[:, -top_k:]
        kval = np.take_along_axis(sims, kidx, axis=1)

        order = np.argsort(kval, axis=1)[:, ::-1]
        kidx = np.take_along_axis(kidx, order, axis=1)
        kval = np.take_along_axis(kval, order, axis=1)

        nearest[start:end] = kval[:, 0]

        for i in range(end - start):
            q = start + i
            for r in range(top_k):
                out_query.append(q)
                out_neighbor.append(int(kidx[i, r]))
                out_rank.append(r + 1)
                out_sim.append(float(kval[i, r]))

    df = pd.DataFrame(
        {
            "query_idx": out_query,
            "neighbor_idx": out_neighbor,
            "rank": out_rank,
            "similarity": out_sim,
        }
    )
    return df, nearest


def pairwise_similarity_matrix(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    sim = x @ x.T
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


@dataclass
class LinkageResult:
    linkage: np.ndarray
    leaf_order: list[int]


def average_linkage(distance: np.ndarray) -> LinkageResult:
    """Naive average-linkage agglomeration for moderate N (e.g., <=300)."""
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be square")
    n = distance.shape[0]
    if n < 2:
        return LinkageResult(linkage=np.zeros((0, 4), dtype=np.float64), leaf_order=list(range(n)))

    def key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    d: dict[tuple[int, int], float] = {}
    active = set(range(n))
    sizes = {i: 1 for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            d[(i, j)] = float(distance[i, j])

    next_id = n
    rows: list[list[float]] = []

    while len(active) > 1:
        best_pair = None
        best_val = np.inf
        for (a, b), val in d.items():
            if a in active and b in active and val < best_val:
                best_val = val
                best_pair = (a, b)

        if best_pair is None:
            break

        a, b = best_pair
        sa, sb = sizes[a], sizes[b]
        c = next_id
        next_id += 1
        sizes[c] = sa + sb

        rows.append([float(a), float(b), float(best_val), float(sa + sb)])

        others = [x for x in active if x not in (a, b)]
        for x in others:
            dax = d[key(a, x)]
            dbx = d[key(b, x)]
            dcx = (sa * dax + sb * dbx) / (sa + sb)
            d[key(c, x)] = float(dcx)

        # remove all keys touching a or b
        keys_to_drop = [k for k in d.keys() if a in k or b in k]
        for kdrop in keys_to_drop:
            d.pop(kdrop, None)

        active.remove(a)
        active.remove(b)
        active.add(c)

    linkage = np.array(rows, dtype=np.float64)

    # Build leaf order via DFS traversal of the final tree.
    children: dict[int, tuple[int, int]] = {}
    for i, row in enumerate(linkage):
        node_id = n + i
        children[node_id] = (int(row[0]), int(row[1]))

    if len(linkage) == 0:
        order = list(range(n))
    else:
        root = n + len(linkage) - 1
        order: list[int] = []

        def dfs(node: int) -> None:
            if node < n:
                order.append(node)
                return
            left, right = children[node]
            dfs(left)
            dfs(right)

        dfs(root)

    return LinkageResult(linkage=linkage, leaf_order=order)


def cut_linkage_n_clusters(n_leaves: int, linkage: np.ndarray, n_clusters: int) -> np.ndarray:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if n_leaves == 0:
        return np.array([], dtype=np.int32)
    if n_clusters >= n_leaves or linkage.shape[0] == 0:
        return np.arange(n_leaves, dtype=np.int32)

    members: dict[int, list[int]] = {i: [i] for i in range(n_leaves)}
    active = set(range(n_leaves))
    next_id = n_leaves

    for row in linkage:
        a = int(row[0])
        b = int(row[1])
        c = next_id
        next_id += 1

        members[c] = members[a] + members[b]

        active.remove(a)
        active.remove(b)
        active.add(c)

        if len(active) <= n_clusters:
            break

    labels = np.full((n_leaves,), -1, dtype=np.int32)
    for label, cluster_id in enumerate(sorted(active)):
        for leaf in members[cluster_id]:
            labels[leaf] = label

    # Safety fallback if any unassigned (should not happen)
    unassigned = np.where(labels < 0)[0]
    for leaf in unassigned:
        labels[leaf] = int(labels.max() + 1)

    return labels


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1, dtype=np.float64) / x.size
    return x, y
