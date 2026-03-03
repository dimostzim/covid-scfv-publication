#!/usr/bin/env python3
"""Publication-ready UpSet plot of shared scFv CDR3 pairs across barcodes 07-12."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from upsetplot import UpSet, from_indicators
from upsetplot import plotting as upsetplot_plotting

BARCODE_COLS = ["barcode07", "barcode08", "barcode09", "barcode10", "barcode11", "barcode12"]
LEFT_GROUP = ["barcode07", "barcode08", "barcode09"]
RIGHT_GROUP = ["barcode10", "barcode11", "barcode12"]

# Suppress very small intersections that overcrowd the x-axis in publication figures.
MIN_SUBSET_SIZE = 10


def _normalize_barcode_key(value: str) -> str:
    text = str(value).strip().lower()
    match = re.search(r"(\d+)$", text)
    if not match:
        return text
    return f"barcode{int(match.group(1))}"


def _load_barcode_labels() -> dict[str, str]:
    """Load pretty labels for barcode columns from barcode_map.csv."""
    candidate_paths = [
        Path(__file__).resolve().parents[2] / "barcode_map.csv",
        Path("barcode_map.csv"),
    ]
    map_path = next((p for p in candidate_paths if p.exists()), None)
    if map_path is None:
        return {col: col for col in BARCODE_COLS}

    barcode_map = pd.read_csv(map_path)
    if not {"Sample", "Pool", "Round", "Barcode"}.issubset(barcode_map.columns):
        return {col: col for col in BARCODE_COLS}

    lookup = {}
    for _, row in barcode_map.iterrows():
        key = _normalize_barcode_key(row["Barcode"])
        sample = f"{int(row['Sample']):02d}"
        pool = str(row["Pool"]).strip()
        round_id = str(row["Round"]).strip()
        lookup[key] = f"{sample} {pool} R{round_id}"

    labels = {}
    for col in BARCODE_COLS:
        key = _normalize_barcode_key(col)
        labels[col] = lookup.get(key, col)
    return labels


def _patch_upsetplot_matrix_for_modern_pandas() -> None:
    """Patch UpSet matrix styling to avoid pandas 3 Copy-on-Write issues."""
    if getattr(upsetplot_plotting.UpSet, "_modern_matrix_patch", False):
        return

    def patched_plot_matrix(self, ax):
        ax = self._reorient(ax)
        data = self.intersections
        n_cats = data.index.nlevels
        inclusion = data.index.to_frame().values

        styles = [
            [
                self.subset_styles[i]
                if inclusion[i, j]
                else {"facecolor": self._other_dots_color, "linewidth": 0}
                for j in range(n_cats)
            ]
            for i in range(len(data))
        ]
        styles = sum(styles, [])
        style_columns = {
            "facecolor": "facecolors",
            "edgecolor": "edgecolors",
            "linewidth": "linewidths",
            "linestyle": "linestyles",
            "hatch": "hatch",
        }
        styles = (
            pd.DataFrame(styles)
            .reindex(columns=style_columns.keys())
            .astype(
                {
                    "facecolor": "O",
                    "edgecolor": "O",
                    "linewidth": float,
                    "linestyle": "O",
                    "hatch": "O",
                }
            )
        )
        styles["linewidth"] = styles["linewidth"].fillna(1)
        styles["facecolor"] = styles["facecolor"].fillna(self._facecolor)
        styles["edgecolor"] = styles["edgecolor"].fillna(styles["facecolor"])
        styles["linestyle"] = styles["linestyle"].fillna("solid")
        del styles["hatch"]

        x = np.repeat(np.arange(len(data)), n_cats)
        y = np.tile(np.arange(n_cats), len(data))

        dot_size = (self._element_size * 0.35) ** 2 if self._element_size is not None else 200
        ax.scatter(
            *self._swapaxes(x, y),
            s=dot_size,
            zorder=10,
            **styles.rename(columns=style_columns),
        )

        if self._with_lines:
            idx = np.flatnonzero(inclusion)
            line_data = (
                pd.Series(y[idx], index=x[idx]).groupby(level=0).aggregate(["min", "max"])
            )
            colors = pd.Series(
                [
                    style.get("edgecolor", style.get("facecolor", self._facecolor))
                    for style in self.subset_styles
                ],
                name="color",
            )
            line_data = line_data.join(colors)
            ax.vlines(
                line_data.index.values,
                line_data["min"],
                line_data["max"],
                lw=2,
                colors=line_data["color"],
                zorder=5,
            )

        tick_axis = ax.yaxis
        tick_axis.set_ticks(np.arange(n_cats))
        tick_axis.set_ticklabels(data.index.names, rotation=0 if self._horizontal else -90)
        ax.xaxis.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
        if not self._horizontal:
            ax.yaxis.set_ticks_position("top")
        ax.set_frame_on(False)
        ax.set_xlim(-0.5, x[-1] + 0.5, auto=False)
        ax.grid(False)

    upsetplot_plotting.UpSet.plot_matrix = patched_plot_matrix
    upsetplot_plotting.UpSet._modern_matrix_patch = True


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def main() -> None:
    _patch_upsetplot_matrix_for_modern_pandas()
    _configure_style()

    df = pd.read_csv("r_output/tables/CDR3s.csv")
    label_map = _load_barcode_labels()
    presence_raw = (df[BARCODE_COLS] > 0)
    presence_raw = presence_raw.loc[presence_raw.any(axis=1)].copy()
    presence = presence_raw.rename(columns=label_map)

    left_labels = [label_map[col] for col in LEFT_GROUP]
    right_labels = [label_map[col] for col in RIGHT_GROUP]

    cross_mask = presence_raw[LEFT_GROUP].any(axis=1) & presence_raw[RIGHT_GROUP].any(axis=1)
    print(f"Total unique CDR3 pairs in barcodes 07-12: {len(presence)}")
    print(f"Cross-group pairs (07-09 with 10-12): {int(cross_mask.sum())}")
    print("Per-barcode counts:")
    for col in BARCODE_COLS:
        print(f"  {label_map[col]}: {int(presence_raw[col].sum())}")

    upset_data = from_indicators(presence)
    upset = UpSet(
        upset_data,
        subset_size="count",
        sort_by="cardinality",
        sort_categories_by="input",
        min_subset_size=MIN_SUBSET_SIZE,
        facecolor="#111827",
        other_dots_color=0.3,
        shading_color=0.08,
        with_lines=True,
        element_size=42,
        intersection_plot_elements=10,
        totals_plot_elements=3,
        show_counts=False,
        show_percentages=False,
    )

    # Highlight any intersection containing at least one barcode from each group.
    first_highlight = True
    for left in left_labels:
        for right in right_labels:
            upset.style_subsets(
                present=[left, right],
                facecolor="#0f766e",
                edgecolor="#0f766e",
                linewidth=0.8,
                label="Cross-group subset (07-09 and 10-12)" if first_highlight else None,
            )
            first_highlight = False

    fig = plt.figure(figsize=(15, 9))
    axes = upset.plot(fig=fig)

    if "intersections" in axes:
        axes["intersections"].set_ylabel("Intersection size (unique CDR3 pairs)")
        axes["intersections"].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        axes["intersections"].spines["top"].set_visible(False)
        axes["intersections"].spines["right"].set_visible(False)
    if "totals" in axes:
        axes["totals"].set_xlabel("Per-barcode total")
        axes["totals"].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        axes["totals"].spines["top"].set_visible(False)
        axes["totals"].spines["right"].set_visible(False)

    fig.suptitle("Shared scFv CDR3 pairs across barcodes 07-12", fontsize=16, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.945,
        f"UpSet plot with subset size >= {MIN_SUBSET_SIZE} (cross-group subsets highlighted in teal)",
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.08)

    out_dir = Path("r_output/cooccurence")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "upset_CDR3s_barcodes7_12.png"
    out_pdf = out_dir / "upset_CDR3s_barcodes7_12.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"Intersection subsets shown (size >= {MIN_SUBSET_SIZE}): {len(upset.intersections)}")
    print(f"Saved to {out_png}")
    print(f"Saved to {out_pdf}")


if __name__ == "__main__":
    main()
