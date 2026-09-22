"""Pathway enrichment (Enrichr) for the "biology" evaluation stage.

Runs Enrichr over a panel's gene list and saves significant pathways plus
top-hit bar charts. Mirrors ``Analysis-scripts/run_pathway_enrichment.py``'s
enrichment logic, adapted to operate on preprocessed panel AnnDatas already
loaded by ``run_evaluation.py`` rather than re-deriving panel paths from
``plot_comparison_groups.py``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-spatial-probe-design")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gseapy as gp

logger = logging.getLogger(__name__)

# Enrichr gene-set libraries, resolved per organism. GO Biological Process (2023)
# and Reactome (2022) both accept mouse symbols (Enrichr matches case-insensitively /
# via orthologs); only KEGG has a dedicated mouse library, so a `--pathway_organism
# mouse` run that does not override --pathway_libraries uses KEGG_2019_Mouse rather
# than the Human KEGG.
PATHWAY_LIBRARIES_BY_ORGANISM: dict[str, list[str]] = {
    "human": ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"],
    "mouse": ["GO_Biological_Process_2023", "KEGG_2019_Mouse", "Reactome_2022"],
}

# Default library set (the human set).
DEFAULT_PATHWAY_LIBRARIES: list[str] = PATHWAY_LIBRARIES_BY_ORGANISM["human"]

# ``ranked_gene_list.csv`` column produced by the Selection module
# (``derive_informative_celltypes`` in ``Selection-module/_gene_list_builder.py``): a
# ``|``-separated, de-duplicated list of the cell types each panel gene is informative for.
# ``|`` (not ``,``) is the separator because cell-type labels themselves contain commas.
COL_INFORMATIVE_CELLTYPES: str = "informative_celltypes"
INFORMATIVE_CELLTYPES_SEP: str = "|"

# Fallback for older gene lists without ``informative_celltypes``. Only ``|``-joined /
# single-value columns are usable here: ``rf_contributing_celltypes`` is ``|``-joined and
# ``celltype`` / ``rf_celltype`` are single labels. ``contributing_celltypes`` (NMF genes)
# is ``", "``-joined with labels that themselves contain commas, so it needs the frame's
# known-cell-type vocabulary to split safely (``Selection-module/_gene_list_builder.py``)
# and is skipped here — regenerate the gene list to get the ``informative_celltypes``
# column instead. ``rf_celltype`` is the sole surviving cell-type column in the trimmed
# rf_deg/rf_simple ``*_panel_information.csv`` schema (no ``celltype`` or
# ``rf_contributing_celltypes`` column there), so it's checked last.
_FALLBACK_PIPE_COLUMN: str = "rf_contributing_celltypes"
_FALLBACK_SINGLE_COLUMN: str = "celltype"
_FALLBACK_RF_CELLTYPE_COLUMN: str = "rf_celltype"
_NON_CELLTYPE_LABELS: frozenset[str] = frozenset(
    {"", "global", "all", "nan", "none", "na"}
)

# Cell types with fewer than this many genes are skipped for per-cell-type enrichment
# (Enrichr needs a handful of genes to return anything meaningful).
DEFAULT_PATHWAY_MIN_GENES_PER_CELLTYPE: int = 5


def resolve_pathway_libraries(
    organism: str, override: list[str] | None = None
) -> list[str]:
    """Return the Enrichr library set to query.

    An explicit ``override`` (from ``--pathway_libraries``) wins; otherwise the
    organism-appropriate default is used, falling back to the human set for an
    unknown organism.
    """
    if override:
        return list(override)
    return list(
        PATHWAY_LIBRARIES_BY_ORGANISM.get(organism, PATHWAY_LIBRARIES_BY_ORGANISM["human"])
    )

TITLE_FS = 16
LABEL_FS = 14
TICK_FS = 12


def run_pathway_enrichment_for_panel(
    label: str,
    genes: list[str],
    out_dir: Path,
    libraries: list[str] = DEFAULT_PATHWAY_LIBRARIES,
    organism: str = "human",
    top_n: int = 12,
) -> pd.DataFrame | None:
    """Run Enrichr on one panel's gene list and save results/plots.

    Args:
        label: Human-readable panel identifier, used in log messages and
            plot titles.
        genes: Panel gene symbols.
        out_dir: Directory to write ``significant_pathways.csv`` and
            ``top_pathways_<library>.svg`` into (created if missing).
        libraries: Enrichr gene-set libraries to query.
        organism: Enrichr organism (``"human"`` or ``"mouse"``).
        top_n: Number of top (lowest adjusted p-value) pathways to plot
            per library.

    Returns:
        DataFrame of significant pathways, or ``None`` if enrichment failed or
        found nothing. "Significant" = Enrichr's own per-library Benjamini-Hochberg
        ``Adjusted P-value`` (read directly, not recomputed here) < 0.05, with the
        libraries then pooled. No additional across-library correction is applied.
    """
    if not genes:
        logger.warning("No genes provided for %s – skipping pathway enrichment", label)
        return None

    try:
        enr = gp.enrichr(gene_list=genes, gene_sets=libraries, organism=organism, outdir=None)
    except Exception as e:
        logger.warning("Enrichr failed for %s: %s", label, e)
        return None
    res = enr.results
    if res is None or res.empty:
        logger.info("No Enrichr results for %s", label)
        return None

    sig = res[res["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value").reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    sig.to_csv(out_dir / "significant_pathways.csv", index=False)

    if sig.empty:
        logger.info("%s: 0 significant pathways (%d genes)", label, len(genes))
        return sig

    for library in libraries:
        lib_sig = sig[sig["Gene_set"] == library]
        if lib_sig.empty:
            continue
        top = lib_sig.head(top_n).iloc[::-1].copy()
        top["neglog10_padj"] = -np.log10(top["Adjusted P-value"].astype(float).clip(lower=1e-300))
        top["short_term"] = top["Term"].str.slice(0, 60)

        fig, ax = plt.subplots(figsize=(11, max(4, len(top) * 0.7)))
        ax.barh(top["short_term"], top["neglog10_padj"], color="#1565C0", alpha=0.88,
                edgecolor="#00008B", linewidth=1.2)
        ax.set_xlabel("-log10(adjusted p-value)", fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.set_title(f"{label} — {library}", fontsize=TITLE_FS, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"top_pathways_{library}.svg", bbox_inches="tight")
        plt.close(fig)

    logger.info("%s: %d significant / %d genes -> %s", label, len(sig), len(genes), out_dir)
    return sig


# ---------------------------------------------------------------------------
# Per-cell-type pathway enrichment
# ---------------------------------------------------------------------------


def _safe_celltype(name: str) -> str:
    """Filesystem-safe cell-type label (labels contain spaces / slashes / commas)."""
    return str(name).replace("/", "_").replace(" ", "_").replace(",", "")


def _split_celltype_field(value: object, sep: str | None) -> list[str]:
    """Real cell-type labels in one provenance-column cell.

    ``sep=None`` treats the whole value as a single label (e.g. the ``celltype`` column);
    otherwise the value is ``sep``-joined. Non-cell-type placeholders (``global``, ``All``,
    ``nan``, empty) are dropped.
    """
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in _NON_CELLTYPE_LABELS:
        return []
    tokens = [text] if sep is None else text.split(sep)
    out: list[str] = []
    for token in tokens:
        token = token.strip()
        if token and token.lower() not in _NON_CELLTYPE_LABELS:
            out.append(token)
    return out


def load_panel_informative_map(
    ranked_csv_path: str | Path, panel_genes: list[str]
) -> dict[str, list[str]]:
    """Map each cell type to the panel genes informative for it.

    Reads a Selection-module ``ranked_gene_list.csv`` and, for every cell type listed in
    its :data:`COL_INFORMATIVE_CELLTYPES` column, collects the panel genes attributed to
    that cell type. Rows are restricted to final-panel genes — those where
    ``final_selection`` / ``in_panel`` is truthy when such a column exists (single-strategy
    output); all rows otherwise (combination-strategy output is panel-only) — then
    intersected with *panel_genes*.

    If ``informative_celltypes`` is absent (older gene lists, or a trimmed rf_deg/rf_simple
    ``*_panel_information.csv``), falls back to a best-effort union of
    ``rf_contributing_celltypes`` (``|``-joined), ``celltype`` (single label), and
    ``rf_celltype`` (single label — the sole cell-type column in the trimmed RF schema),
    skipping non-cell-type labels (``global``, ``All``, ``nan``). ``contributing_celltypes``
    (NMF genes) is *not* parsed in the fallback — it is ``", "``-joined with labels that
    contain commas — so those genes get no attribution; regenerate the gene list to obtain
    the ``informative_celltypes`` column for full coverage.

    Args:
        ranked_csv_path: Path to a ``ranked_gene_list.csv``.
        panel_genes: The panel's gene symbols (from the preprocessed panel AnnData).

    Returns:
        ``{celltype: [genes]}``, gene order following the CSV row order; empty if the file
        is missing or carries no usable cell-type provenance.
    """
    path = Path(ranked_csv_path)
    if not path.exists():
        logger.warning("ranked_gene_list.csv not found: %s", path)
        return {}

    df = pd.read_csv(path)
    if "gene" not in df.columns:
        logger.warning("No 'gene' column in %s – cannot build per-cell-type map", path)
        return {}

    for panel_col in ("final_selection", "in_panel"):
        if panel_col in df.columns:
            df = df[df[panel_col].fillna(False).astype(bool)]
            break

    panel_set = set(panel_genes)

    # (source column, separator) pairs contributing cell-type labels per row.
    if COL_INFORMATIVE_CELLTYPES in df.columns:
        sources = [(COL_INFORMATIVE_CELLTYPES, INFORMATIVE_CELLTYPES_SEP)]
    else:
        sources = [
            (c, sep)
            for c, sep in (
                (_FALLBACK_PIPE_COLUMN, INFORMATIVE_CELLTYPES_SEP),
                (_FALLBACK_SINGLE_COLUMN, None),  # single label, do not split
                (_FALLBACK_RF_CELLTYPE_COLUMN, None),  # single label, do not split
            )
            if c in df.columns
        ]
        if not sources:
            logger.warning(
                "%s has no '%s' column and no usable fallback column – "
                "per-cell-type enrichment unavailable for this panel",
                path, COL_INFORMATIVE_CELLTYPES,
            )
            return {}
        logger.warning(
            "%s lacks '%s'; falling back to %s (NMF-gene 'contributing_celltypes' not "
            "parsed – regenerate the gene list for full per-cell-type coverage)",
            path.name, COL_INFORMATIVE_CELLTYPES, [c for c, _ in sources],
        )

    genes_by_celltype: dict[str, list[str]] = {}
    for row in df.itertuples(index=False):
        gene = getattr(row, "gene")
        if gene not in panel_set:
            continue
        for source_col, sep in sources:
            for celltype in _split_celltype_field(getattr(row, source_col), sep):
                bucket = genes_by_celltype.setdefault(celltype, [])
                if gene not in bucket:
                    bucket.append(gene)
    return genes_by_celltype


def run_pathway_enrichment_by_celltype(
    label: str,
    genes_by_celltype: dict[str, list[str]],
    out_dir: Path,
    libraries: list[str] = DEFAULT_PATHWAY_LIBRARIES,
    organism: str = "human",
    top_n: int = 12,
    min_genes: int = DEFAULT_PATHWAY_MIN_GENES_PER_CELLTYPE,
    sleep: float = 1.0,
) -> pd.DataFrame:
    """Run Enrichr once per cell type on that cell type's gene subset.

    For each cell type with at least *min_genes* genes, calls
    :func:`run_pathway_enrichment_for_panel` with ``out_dir = out_dir / <safe celltype>``.
    Cell types below the threshold are recorded in the summary with a ``skipped_reason``
    and not queried.

    Args:
        label: Panel identifier, prefixed onto each per-cell-type Enrichr label / title.
        genes_by_celltype: ``{celltype: [genes]}`` (e.g. from
            :func:`load_panel_informative_map`, or the full panel per cell type).
        out_dir: Parent directory for the per-cell-type result subdirectories.
        libraries: Enrichr gene-set libraries.
        organism: Enrichr organism.
        top_n: Top pathways plotted per library.
        min_genes: Minimum genes for a cell type to be queried.
        sleep: Seconds slept between per-cell-type Enrichr calls.

    Returns:
        Summary DataFrame with one row per cell type: columns ``celltype``, ``n_genes``,
        ``n_significant``, ``skipped_reason`` (empty string when the cell type was run).
    """
    rows: list[dict[str, object]] = []
    for celltype in sorted(genes_by_celltype):
        genes = genes_by_celltype[celltype]
        if len(genes) < min_genes:
            rows.append({
                "celltype": celltype,
                "n_genes": len(genes),
                "n_significant": 0,
                "skipped_reason": f"fewer than {min_genes} genes",
            })
            continue
        sig = run_pathway_enrichment_for_panel(
            label=f"{label} — {celltype}",
            genes=genes,
            out_dir=out_dir / _safe_celltype(celltype),
            libraries=libraries,
            organism=organism,
            top_n=top_n,
        )
        rows.append({
            "celltype": celltype,
            "n_genes": len(genes),
            "n_significant": 0 if sig is None else len(sig),
            "skipped_reason": "",
        })
        time.sleep(sleep)

    return pd.DataFrame(
        rows, columns=["celltype", "n_genes", "n_significant", "skipped_reason"]
    )
