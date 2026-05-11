"""Dataset name parsing and panel filtering utilities.

This module provides functions to parse structured dataset names produced by
the selection pipeline and to filter sets of preprocessed h5ad files based
on strategy, panel size, preprocessing method, and other attributes.

Dataset names follow the convention::

    {Filter}_{Baseline}_{Strategy}_{N-genes}[_{addon}][_{gap-filling}]

Examples::

    Scanpy-Filter_All-Genes_deg_only_100
    Xenium-Filter_HVG-Subset_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_5factors_200_5k-addon
    Scanpy-Filter_Spapros_100_mMulti-addon_cell-type-specific-filling
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "extract_strategy_from_dataset_name",
    "parse_dataset_attributes",
    "group_datasets_by_attributes",
    "filter_datasets_by_keywords",
    "filter_datasets_by_args",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FILLING_STRATEGIES: tuple[str, ...] = (
    "cell-type-specific-filling",
    "global-gene-filling",
    "DEG-based-filling",
)

_SIMPLE_STRATEGIES: tuple[str, ...] = (
    "deg_only",
    "dt_simple",
    "dt_deg",
    "hvg",
    "random",
)

_EXTERNAL_PANEL_NAMES: tuple[str, ...] = ("Spapros", "mMulti_v1", "5k")


# ---------------------------------------------------------------------------
# Dataset name parsing
# ---------------------------------------------------------------------------


def extract_strategy_from_dataset_name(
    dataset_name: str,
) -> tuple[str, str | None, int | None]:
    """Extract strategy, filling method, and factor count from a dataset name.

    This function identifies the core selection strategy used, ignoring
    settings like filter, baseline gene pool, panel size, and addon panels.

    Args:
        dataset_name: Full dataset name or file path (with or without
            ``.h5ad`` extension).

    Returns:
        A 3-tuple of ``(strategy, filling_method, n_factors)`` where:

        - *strategy* is the core strategy identifier
          (e.g. ``"deg_only"``, ``"dt_pca_DT0.5_Dimred0.5_global_method_a"``).
        - *filling_method* is one of the gap-filling identifiers or ``None``.
        - *n_factors* is the number of NMF/PCA factors, or ``None`` if absent.

    Example:
        >>> extract_strategy_from_dataset_name(
        ...     "Scanpy-Filter_All-Genes_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_5factors_100"
        ... )
        ('dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a', None, 5)
    """
    import os  # keep stdlib import; os.path is lightweight

    name = os.path.basename(dataset_name)
    if name.endswith(".h5ad"):
        name = name[:-5]

    # Extract factor count (e.g. "5factors" → 5)
    n_factors: int | None = None
    factor_match = re.search(r"_(\d+)factors_", name)
    if factor_match:
        n_factors = int(factor_match.group(1))
        name = re.sub(r"_(\d+)factors_", "_", name)

    # Check for gap-filling suffix
    filling_method: str | None = None
    for filling in _FILLING_STRATEGIES:
        if filling in name:
            filling_method = filling
            name = name.replace(f"_{filling}", "")
            break

    parts = name.split("_")

    # External standalone panels (not addon suffixes)
    if "Spapros" in parts and not name.endswith("Spapros-addon"):
        return ("Spapros", filling_method, n_factors)
    if ("mMulti" in parts or "mMulti_v1" in name) and not name.endswith("mMulti-addon"):
        return ("mMulti_v1", filling_method, n_factors)
    if "5k" in parts and "5k-addon" not in name:
        return ("5k", filling_method, n_factors)

    # Simple strategies
    for strategy in _SIMPLE_STRATEGIES:
        if f"_{strategy}_" in name or name.endswith(f"_{strategy}"):
            return (strategy, filling_method, n_factors)

    # Baselines used as strategy (capitalised)
    if "Random" in parts and parts.index("Random") > 1:
        return ("Random", filling_method, n_factors)
    if "HVG" in parts and parts.index("HVG") > 1:
        return ("HVG", filling_method, n_factors)

    # Dimensionality-reduction-only strategies: pca/nmf + global/per_celltype + method_a/b
    dimred_match = re.search(r"(pca|nmf)_(global|per_celltype)_method_([ab])", name)
    if dimred_match:
        dimred_type, analysis_type, method = dimred_match.groups()
        return (f"{dimred_type}_{analysis_type}_method_{method}", filling_method, n_factors)

    # Hybrid strategies: dt_pca/dt_nmf + ratio parameters
    combo_match = re.search(
        r"dt_(pca|nmf)_DT([\d.]+)_Dimred([\d.]+)_(global|per_celltype)_method_([ab])",
        name,
    )
    if combo_match:
        dimred_type, dt_ratio, dimred_ratio, analysis_type, method = combo_match.groups()
        strategy = (
            f"dt_{dimred_type}_DT{dt_ratio}_Dimred{dimred_ratio}_{analysis_type}_method_{method}"
        )
        return (strategy, filling_method, n_factors)

    logger.warning("Could not extract strategy from dataset name: %s", dataset_name)
    return (name, filling_method, n_factors)


def parse_dataset_attributes(dataset_name: str) -> dict[str, Any]:
    """Parse a dataset name into its component attributes.

    Args:
        dataset_name: Full dataset name (no path, no extension needed but
            both are handled).

    Returns:
        Dictionary with keys: ``filter``, ``baseline``, ``strategy``,
        ``panel_size``, ``addon_type``, ``filling_method``, ``n_factors``,
        ``full_name``.

    Example:
        >>> attrs = parse_dataset_attributes(
        ...     "Scanpy-Filter_HVG-Subset_deg_only_100_5k-addon"
        ... )
        >>> attrs["strategy"]
        'deg_only'
        >>> attrs["addon_type"]
        '5k-addon'
    """
    parts = dataset_name.split("_")

    strategy, filling_method, n_factors = extract_strategy_from_dataset_name(dataset_name)
    is_standalone_external = strategy in _EXTERNAL_PANEL_NAMES

    attributes: dict[str, Any] = {
        "filter": None,
        "baseline": None,
        "strategy": strategy,
        "panel_size": None,
        "addon_type": "no-addon",
        "filling_method": filling_method,
        "n_factors": n_factors,
        "full_name": dataset_name,
    }

    # Filter (first part)
    if parts:
        if is_standalone_external and parts[0] in ("5k", "mMulti", "mMulti_v1"):
            attributes["filter"] = None
        else:
            attributes["filter"] = parts[0]

    # Baseline (second part)
    external_names = {"Spapros", "mMulti", "5k"}
    is_external = any(ext in strategy for ext in external_names)
    if len(parts) > 1:
        if is_external:
            if "All-Genes" in parts[1] or "HVG" in parts[1]:
                attributes["baseline"] = parts[1]
            else:
                attributes["baseline"] = "All-Genes"
        else:
            if "All-Genes" in parts[1] or "HVG" in parts[1]:
                attributes["baseline"] = parts[1]
            else:
                attributes["baseline"] = "All-Genes"

    # Addon type
    for part in parts:
        if "5k-addon" in part:
            attributes["addon_type"] = "5k-addon"
            break
        if "mMulti-addon" in part:
            attributes["addon_type"] = "mMulti-addon"
            break
        if "addon" in part and "no-addon" not in part:
            attributes["addon_type"] = "addon"
            break

    # Standalone external panels: assign reference addon type
    if is_standalone_external and attributes["addon_type"] == "no-addon":
        if strategy == "5k":
            attributes["addon_type"] = "5k-reference"
        elif strategy in ("mMulti_v1", "mMulti"):
            attributes["addon_type"] = "mMulti-reference"

    # Panel size (first purely numeric part)
    for part in parts:
        if part.isdigit():
            attributes["panel_size"] = part
            break

    return attributes


# ---------------------------------------------------------------------------
# Dataset grouping
# ---------------------------------------------------------------------------


def group_datasets_by_attributes(
    dataset_names: list[str],
) -> dict[str, list[str]]:
    """Group dataset names by filter, baseline, panel size, and addon type.

    External standalone panels (``5k``, ``mMulti_v1``) are merged into their
    corresponding addon groups rather than appearing as separate groups.

    Args:
        dataset_names: List of dataset names to group.

    Returns:
        Mapping of ``{group_key: [dataset_name, ...]}`` where *group_key* has
        the form ``{filter}_{baseline}_{panel_size}_{addon_type}``.

    Example:
        >>> groups = group_datasets_by_attributes(["Scanpy-Filter_All-Genes_deg_only_100"])
        >>> list(groups.keys())
        ['Scanpy-Filter_All-Genes_100_no-addon']
    """
    groups: dict[str, list[str]] = defaultdict(list)
    external_panels: dict[str, list[str]] = {"5k": [], "mMulti_v1": []}

    for name in dataset_names:
        attrs = parse_dataset_attributes(name)
        strategy = attrs["strategy"]

        if strategy == "5k":
            external_panels["5k"].append(name)
            continue
        if strategy in ("mMulti_v1", "mMulti"):
            external_panels["mMulti_v1"].append(name)
            continue

        filter_key = attrs["filter"] or "unknown-filter"
        baseline_key = attrs["baseline"] or "unknown-baseline"
        size_key = attrs["panel_size"] or "unknown-size"
        addon_key = attrs["addon_type"]
        group_key = f"{filter_key}_{baseline_key}_{size_key}_{addon_key}"
        groups[group_key].append(name)

    # Merge external panels into addon groups
    for group_key in list(groups):
        if "5k-addon" in group_key:
            groups[group_key].extend(external_panels["5k"])
        if "mMulti-addon" in group_key:
            groups[group_key].extend(external_panels["mMulti_v1"])

    return dict(groups)


def filter_datasets_by_keywords(
    dataset_names: list[str],
    keywords: list[str],
) -> list[str]:
    """Filter dataset names whose strategy matches any keyword.

    Matching is case-insensitive and supports partial matches so that
    e.g. ``"pca_global"`` matches ``"pca_global_method_a"``.

    Args:
        dataset_names: All available dataset names.
        keywords: Keywords to match against extracted strategy strings.

    Returns:
        Filtered list of dataset names.
    """
    filtered: list[str] = []
    for name in dataset_names:
        strategy, *_ = extract_strategy_from_dataset_name(name)
        strategy_lower = strategy.lower()
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in strategy_lower or strategy_lower in kw_lower:
                filtered.append(name)
                break
    return filtered


# ---------------------------------------------------------------------------
# h5ad file filtering
# ---------------------------------------------------------------------------


def filter_datasets_by_args(
    h5ad_files: list[str],
    filter_args: dict[str, Any],
) -> list[str]:
    """Filter a list of h5ad file paths using pipeline filter arguments.

    Args:
        h5ad_files: List of file paths to candidate h5ad files.
        filter_args: Dictionary of filter parameters. Supported keys:

            - ``strategies``: list of strategy names to include.
            - ``probeset_sizes``: list of int panel sizes to include.
            - ``filter_methods``: list of preprocessing filters
              (``"scanpy"``, ``"10x"``, ``"no_filter"``).
            - ``hvg_subset_options``: list of ``"true"``/``"false"`` strings.
            - ``reduction_types``: list of reduction types.
            - ``analysis_types``: list of analysis types.
            - ``dimred_methods``: list of dimred methods.
            - ``dt_percentages``: list of DT percentage floats.
            - ``dimred_percentages``: list of Dimred percentage floats.
            - ``run_celltype_specific_filling``: ``"true"``/``"false"`` or ``None``.
            - ``run_global_gene_filling``: ``"true"``/``"false"`` or ``None``.
            - ``run_deg_based_filling``: ``"true"``/``"false"`` or ``None``.
            - ``preferred_strategy``: specific strategy name or ``None``.

    Returns:
        Filtered list of file paths.
    """
    import os

    filtered: list[str] = []

    for h5ad_file in h5ad_files:
        filename = os.path.basename(h5ad_file).replace(".h5ad", "")
        if _file_passes_all_filters(filename, filter_args):
            filtered.append(h5ad_file)

    logger.info("Filtering: %d → %d datasets", len(h5ad_files), len(filtered))
    return filtered


def _file_passes_all_filters(
    filename: str,
    filter_args: dict[str, Any],
) -> bool:
    """Return True if *filename* satisfies all active filters.

    Args:
        filename: Dataset name (without path or extension).
        filter_args: Filter criteria dict (see :func:`filter_datasets_by_args`).

    Returns:
        ``True`` if the file passes all active filters.
    """
    attrs = parse_dataset_attributes(filename)
    strategy = attrs["strategy"]
    filling_method = attrs.get("filling_method")

    # Strategy filter
    if filter_args.get("strategies"):
        if not any(s.lower() in strategy.lower() for s in filter_args["strategies"]):
            logger.debug("Excluded %s: strategy '%s' not in %s", filename, strategy, filter_args["strategies"])
            return False

    # Panel size filter
    if filter_args.get("probeset_sizes"):
        panel_size = attrs.get("panel_size")
        if panel_size and int(panel_size) not in filter_args["probeset_sizes"]:
            logger.debug("Excluded %s: size %s not in %s", filename, panel_size, filter_args["probeset_sizes"])
            return False

    # Preprocessing filter method
    if filter_args.get("filter_methods"):
        filter_name = attrs.get("filter", "")
        matched = any(
            (fm == "scanpy" and "Scanpy" in filter_name)
            or (fm == "10x" and "Xenium" in filter_name)
            or (fm == "no_filter" and "No-Filter" in filter_name)
            for fm in filter_args["filter_methods"]
        )
        if not matched:
            logger.debug("Excluded %s: filter '%s' not matched", filename, filter_name)
            return False

    # HVG subset option
    if filter_args.get("hvg_subset_options"):
        baseline = attrs.get("baseline", "")
        matched = any(
            (opt == "true" and "HVG" in baseline)
            or (opt == "false" and "All-Genes" in baseline)
            for opt in filter_args["hvg_subset_options"]
        )
        if not matched:
            logger.debug("Excluded %s: baseline '%s' not matched", filename, baseline)
            return False

    # Reduction type
    if filter_args.get("reduction_types"):
        if not any(rt in strategy.lower() for rt in filter_args["reduction_types"]):
            logger.debug("Excluded %s: no reduction type matched", filename)
            return False

    # Analysis type
    if filter_args.get("analysis_types"):
        if not any(at in strategy.lower() for at in filter_args["analysis_types"]):
            logger.debug("Excluded %s: no analysis type matched", filename)
            return False

    # Dimred method
    if filter_args.get("dimred_methods"):
        if not any(dm.replace("_", "") in strategy.lower() for dm in filter_args["dimred_methods"]):
            logger.debug("Excluded %s: no dimred method matched", filename)
            return False

    # DT percentages
    if filter_args.get("dt_percentages"):
        dt_match = re.search(r"DT([\d.]+)", strategy)
        if dt_match:
            if float(dt_match.group(1)) not in filter_args["dt_percentages"]:
                logger.debug("Excluded %s: DT percentage not matched", filename)
                return False
        else:
            logger.debug("Excluded %s: no DT percentage in strategy", filename)
            return False

    # Dimred percentages
    if filter_args.get("dimred_percentages"):
        dimred_match = re.search(r"Dimred([\d.]+)", strategy)
        if dimred_match:
            if float(dimred_match.group(1)) not in filter_args["dimred_percentages"]:
                logger.debug("Excluded %s: Dimred percentage not matched", filename)
                return False
        else:
            logger.debug("Excluded %s: no Dimred percentage in strategy", filename)
            return False

    # Gap-filling filters
    if filter_args.get("run_celltype_specific_filling"):
        expected = filter_args["run_celltype_specific_filling"].lower() == "true"
        if (filling_method == "cell-type-specific-filling") != expected:
            return False

    if filter_args.get("run_global_gene_filling"):
        expected = filter_args["run_global_gene_filling"].lower() == "true"
        if (filling_method == "global-gene-filling") != expected:
            return False

    if filter_args.get("run_deg_based_filling"):
        expected = filter_args["run_deg_based_filling"].lower() == "true"
        if (filling_method == "DEG-based-filling") != expected:
            return False

    if filter_args.get("preferred_strategy"):
        pref = filter_args["preferred_strategy"].lower()
        if filling_method and pref not in filling_method.lower():
            return False

    return True
