#!/bin/bash
#SBATCH --job-name=Plot-Eval-v2
#SBATCH --output=logs/Plot-Eval-v2_%j.out
#SBATCH --error=logs/Plot-Eval-v2_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G

# =============================================================================
# Flexible Plotting Pipeline (Modules v2)
# =============================================================================
# Usage: [sbatch] run_plotting_pipeline.sh
# Configure via variables in the CONFIGURATION section below.
#
# Creates plots from pre-computed evaluation results without re-running
# evaluation. Supports:
#   - Any combination of probe panels in visualisations
#   - Mixing panels of different sizes in the same plot
#   - Separate baseline and variability plotting
#   - Category-based plot groups for systematic comparisons
#
# To add a new category:
#   1. Define a new array: MY_GROUPS=(...)
#   2. Add a line to CATEGORY_CONFIGS:
#        "MY_GROUPS:baseline:Baseline/My-Output:clustering,neighborhood,celltype"
#   3. Re-run; the category is processed automatically.
#
# Author: Helene Hemmer
# Date: 2026-03-18
# =============================================================================

set -euo pipefail

echo "=== Flexible Plotting Pipeline (Modules v2) ==="
echo "Starting at $(date)"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dry-run: set to "true" to print the full command list without executing
DRY_RUN=false

# --- Paths ---

# Python environment
ENV_PYTHON="/home/gruengroup/helene/.conda/envs/SpaprosProbeDesign/bin/python"

# Path to the v2 plotting entry point (relative to this script's location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOTTING_SCRIPT="${SCRIPT_DIR}/Modules_v2/Plotting-module/plot_evaluation.py"

# Results directories produced by run_evaluation_pipeline.sh
BASELINE_RESULTS_DIR="/home/gruengroup/helene/SpatialProbeDesign_tmp/Experiments/Niec_Intestinal-data/Evaluation-pipeline_v2-RF/Baseline-Evaluation/results"
# LCA:       "/home/gruengroup/helene/SpatialProbeDesign_tmp/Experiments/LCA/Evaluation-pipeline/Baseline-Evaluation/results"

VARIABILITY_RESULTS_DIR="/home/gruengroup/helene/SpatialProbeDesign_tmp/Experiments/Niec_Intestinal-data/Evaluation-pipeline_v2-RF/Variability-Evaluation/results"
# LCA:       "/home/gruengroup/helene/SpatialProbeDesign_tmp/Experiments/LCA/Evaluation-pipeline/Variability-Evaluation/results"

# Root output directory for user-defined plots
PLOTS_OUTPUT_DIR="/home/gruengroup/helene/SpatialProbeDesign_tmp/Experiments/Niec_Intestinal-data/Evaluation-pipeline_v2-RF/User-defined-plots"
# LCA:       "/home/gruengroup/helene/SpatialProbeDesign_tmp/Experiments/LCA/Evaluation-pipeline/User-defined-plots"

# --- Plot settings ---

PNG_DPI=300

# --- External panel names ---
# Simple identifiers as returned by extract_strategy_and_display_name()
# (e.g. "Spapros" "mMulti_v1" "NSForest") – used for colour assignment
EXTERNAL_NAMES=("Spapros" "mMulti_v1")

# --- Plot type switches ---
# These act as global on/off switches for individual plot types.
# Fine-grained control per category is done via the PLOT_FLAGS field in
# CATEGORY_CONFIGS below.

# Baseline evaluation plots
PLOT_CLUSTERING=true              # Clustering quality (ARI, NMI)
PLOT_NEIGHBORHOOD=true            # Neighbourhood preservation
PLOT_CELLTYPE=true                # Cell type identification

# Variability evaluation plots
PLOT_MECHANISTIC=true             # Mechanistic representation
PLOT_CELLTYPE_SPECIFIC=false      # Per-celltype breakdowns (expensive)

# =============================================================================
# PLOT GROUP DEFINITIONS
# =============================================================================
# Format: "group_name:panel1,panel2,panel3,..."
# group_name  – used as output subdirectory name and plot title
# panel names – match the dataset names in evaluation result CSVs
#               (preprocessed .h5ad filenames without the .h5ad extension)
#
# Add as many category arrays as needed; enable them in CATEGORY_CONFIGS below.

# ---------------------------------------------------------------------------
# Modules v2 – Random forest selection (Niec Intestinal data)
# Category 1: Compare DEG:dimred ratios vs benchmarks (no 10x add-on)
# ---------------------------------------------------------------------------
CATEGORY_1_PLOT_GROUPS_v2_RF_Eval=(
    # Scanpy-Filter_All-Genes 100-gene NMF
    "Scanpy-Filter_All-Genes_100_NMF_25-75_v2:Scanpy-Filter_All-Genes_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_100,Scanpy-Filter_All-Genes_nmf_per_celltype_method_a_100,Scanpy-Filter_All-Genes_dt_simple_100,Scanpy-Filter_All-Genes_dt_deg_100,Scanpy-Filter_All-Genes_deg_only_100,Scanpy-Filter_All-Genes_hvg_100,Scanpy-Filter_All-Genes_random_100,Scanpy-Filter_Spapros_100"
    "Scanpy-Filter_All-Genes_100_NMF_50-50_v2:Scanpy-Filter_All-Genes_dt_nmf_DT0.5_Dimred0.5_per_celltype_method_a_100,Scanpy-Filter_All-Genes_nmf_per_celltype_method_a_100,Scanpy-Filter_All-Genes_dt_simple_100,Scanpy-Filter_All-Genes_dt_deg_100,Scanpy-Filter_All-Genes_deg_only_100,Scanpy-Filter_All-Genes_hvg_100,Scanpy-Filter_All-Genes_random_100,Scanpy-Filter_Spapros_100"
    "Scanpy-Filter_All-Genes_100_NMF_75-25_v2:Scanpy-Filter_All-Genes_dt_nmf_DT0.75_Dimred0.25_per_celltype_method_a_100,Scanpy-Filter_All-Genes_nmf_per_celltype_method_a_100,Scanpy-Filter_All-Genes_dt_simple_100,Scanpy-Filter_All-Genes_dt_deg_100,Scanpy-Filter_All-Genes_deg_only_100,Scanpy-Filter_All-Genes_hvg_100,Scanpy-Filter_All-Genes_random_100,Scanpy-Filter_Spapros_100"
    # Scanpy-Filter_All-Genes 100-gene PCA (DT0.25 and DT0.75 have filling suffix)
    "Scanpy-Filter_All-Genes_100_PCA_25-75_v2:Scanpy-Filter_All-Genes_dt_pca_DT0.25_Dimred0.75_per_celltype_method_a_100_cell-type-specific-filling,Scanpy-Filter_All-Genes_pca_per_celltype_method_a_100,Scanpy-Filter_All-Genes_dt_simple_100,Scanpy-Filter_All-Genes_dt_deg_100,Scanpy-Filter_All-Genes_deg_only_100,Scanpy-Filter_All-Genes_hvg_100,Scanpy-Filter_All-Genes_random_100,Scanpy-Filter_Spapros_100"
    "Scanpy-Filter_All-Genes_100_PCA_50-50_v2:Scanpy-Filter_All-Genes_dt_pca_DT0.5_Dimred0.5_per_celltype_method_a_100,Scanpy-Filter_All-Genes_pca_per_celltype_method_a_100,Scanpy-Filter_All-Genes_dt_simple_100,Scanpy-Filter_All-Genes_dt_deg_100,Scanpy-Filter_All-Genes_deg_only_100,Scanpy-Filter_All-Genes_hvg_100,Scanpy-Filter_All-Genes_random_100,Scanpy-Filter_Spapros_100"
    "Scanpy-Filter_All-Genes_100_PCA_75-25_v2:Scanpy-Filter_All-Genes_dt_pca_DT0.75_Dimred0.25_per_celltype_method_a_100,Scanpy-Filter_All-Genes_pca_per_celltype_method_a_100,Scanpy-Filter_All-Genes_dt_simple_100,Scanpy-Filter_All-Genes_dt_deg_100,Scanpy-Filter_All-Genes_deg_only_100,Scanpy-Filter_All-Genes_hvg_100,Scanpy-Filter_All-Genes_random_100,Scanpy-Filter_Spapros_100"
    # Scanpy-Filter_All-Genes 200-gene NMF (all have filling suffix for v2-RF)
    "Scanpy-Filter_All-Genes_200_NMF_25-75_v2:Scanpy-Filter_All-Genes_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_200_cell-type-specific-filling,Scanpy-Filter_All-Genes_nmf_per_celltype_method_a_200,Scanpy-Filter_All-Genes_dt_simple_200,Scanpy-Filter_All-Genes_dt_deg_200,Scanpy-Filter_All-Genes_deg_only_200,Scanpy-Filter_All-Genes_hvg_200,Scanpy-Filter_All-Genes_random_200,Scanpy-Filter_Spapros_200"
    "Scanpy-Filter_All-Genes_200_NMF_50-50_v2:Scanpy-Filter_All-Genes_dt_nmf_DT0.5_Dimred0.5_per_celltype_method_a_200_cell-type-specific-filling,Scanpy-Filter_All-Genes_nmf_per_celltype_method_a_200,Scanpy-Filter_All-Genes_dt_simple_200,Scanpy-Filter_All-Genes_dt_deg_200,Scanpy-Filter_All-Genes_deg_only_200,Scanpy-Filter_All-Genes_hvg_200,Scanpy-Filter_All-Genes_random_200,Scanpy-Filter_Spapros_200"
    "Scanpy-Filter_All-Genes_200_NMF_75-25_v2:Scanpy-Filter_All-Genes_dt_nmf_DT0.75_Dimred0.25_per_celltype_method_a_200_cell-type-specific-filling,Scanpy-Filter_All-Genes_nmf_per_celltype_method_a_200,Scanpy-Filter_All-Genes_dt_simple_200,Scanpy-Filter_All-Genes_dt_deg_200,Scanpy-Filter_All-Genes_deg_only_200,Scanpy-Filter_All-Genes_hvg_200,Scanpy-Filter_All-Genes_random_200,Scanpy-Filter_Spapros_200"
    # Scanpy-Filter_All-Genes 200-gene PCA (DT0.5 and DT0.75 have filling suffix)
    "Scanpy-Filter_All-Genes_200_PCA_25-75_v2:Scanpy-Filter_All-Genes_dt_pca_DT0.25_Dimred0.75_per_celltype_method_a_200,Scanpy-Filter_All-Genes_pca_per_celltype_method_a_200,Scanpy-Filter_All-Genes_dt_simple_200,Scanpy-Filter_All-Genes_dt_deg_200,Scanpy-Filter_All-Genes_deg_only_200,Scanpy-Filter_All-Genes_hvg_200,Scanpy-Filter_All-Genes_random_200,Scanpy-Filter_Spapros_200"
    "Scanpy-Filter_All-Genes_200_PCA_50-50_v2:Scanpy-Filter_All-Genes_dt_pca_DT0.5_Dimred0.5_per_celltype_method_a_200_cell-type-specific-filling,Scanpy-Filter_All-Genes_pca_per_celltype_method_a_200,Scanpy-Filter_All-Genes_dt_simple_200,Scanpy-Filter_All-Genes_dt_deg_200,Scanpy-Filter_All-Genes_deg_only_200,Scanpy-Filter_All-Genes_hvg_200,Scanpy-Filter_All-Genes_random_200,Scanpy-Filter_Spapros_200"
    "Scanpy-Filter_All-Genes_200_PCA_75-25_v2:Scanpy-Filter_All-Genes_dt_pca_DT0.75_Dimred0.25_per_celltype_method_a_200_cell-type-specific-filling,Scanpy-Filter_All-Genes_pca_per_celltype_method_a_200,Scanpy-Filter_All-Genes_dt_simple_200,Scanpy-Filter_All-Genes_dt_deg_200,Scanpy-Filter_All-Genes_deg_only_200,Scanpy-Filter_All-Genes_hvg_200,Scanpy-Filter_All-Genes_random_200,Scanpy-Filter_Spapros_200"
    # Scanpy-Filter_HVG-Subset 100-gene NMF (DT0.25 and DT0.75 have filling suffix)
    "Scanpy-Filter_HVG-Subset_100_NMF_25-75_v2:Scanpy-Filter_HVG-Subset_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_100_cell-type-specific-filling,Scanpy-Filter_HVG-Subset_nmf_per_celltype_method_a_100,Scanpy-Filter_HVG-Subset_dt_simple_100,Scanpy-Filter_HVG-Subset_dt_deg_100,Scanpy-Filter_HVG-Subset_deg_only_100,Scanpy-Filter_HVG-Subset_hvg_100,Scanpy-Filter_HVG-Subset_random_100,Scanpy-Filter_Spapros_100"
    "Scanpy-Filter_HVG-Subset_100_NMF_50-50_v2:Scanpy-Filter_HVG-Subset_dt_nmf_DT0.5_Dimred0.5_per_celltype_method_a_100,Scanpy-Filter_HVG-Subset_nmf_per_celltype_method_a_100,Scanpy-Filter_HVG-Subset_dt_simple_100,Scanpy-Filter_HVG-Subset_dt_deg_100,Scanpy-Filter_HVG-Subset_deg_only_100,Scanpy-Filter_HVG-Subset_hvg_100,Scanpy-Filter_HVG-Subset_random_100,Scanpy-Filter_Spapros_100"
    "Scanpy-Filter_HVG-Subset_100_NMF_75-25_v2:Scanpy-Filter_HVG-Subset_dt_nmf_DT0.75_Dimred0.25_per_celltype_method_a_100_cell-type-specific-filling,Scanpy-Filter_HVG-Subset_nmf_per_celltype_method_a_100,Scanpy-Filter_HVG-Subset_dt_simple_100,Scanpy-Filter_HVG-Subset_dt_deg_100,Scanpy-Filter_HVG-Subset_deg_only_100,Scanpy-Filter_HVG-Subset_hvg_100,Scanpy-Filter_HVG-Subset_random_100,Scanpy-Filter_Spapros_100"
    # Scanpy-Filter_HVG-Subset 100-gene PCA (DT0.25 and DT0.5 have filling suffix)
    "Scanpy-Filter_HVG-Subset_100_PCA_25-75_v2:Scanpy-Filter_HVG-Subset_dt_pca_DT0.25_Dimred0.75_per_celltype_method_a_100_cell-type-specific-filling,Scanpy-Filter_HVG-Subset_pca_per_celltype_method_a_100,Scanpy-Filter_HVG-Subset_dt_simple_100,Scanpy-Filter_HVG-Subset_dt_deg_100,Scanpy-Filter_HVG-Subset_deg_only_100,Scanpy-Filter_HVG-Subset_hvg_100,Scanpy-Filter_HVG-Subset_random_100,Scanpy-Filter_Spapros_100"
    "Scanpy-Filter_HVG-Subset_100_PCA_50-50_v2:Scanpy-Filter_HVG-Subset_dt_pca_DT0.5_Dimred0.5_per_celltype_method_a_100_cell-type-specific-filling,Scanpy-Filter_HVG-Subset_pca_per_celltype_method_a_100,Scanpy-Filter_HVG-Subset_dt_simple_100,Scanpy-Filter_HVG-Subset_dt_deg_100,Scanpy-Filter_HVG-Subset_deg_only_100,Scanpy-Filter_HVG-Subset_hvg_100,Scanpy-Filter_HVG-Subset_random_100,Scanpy-Filter_Spapros_100"
    "Scanpy-Filter_HVG-Subset_100_PCA_75-25_v2:Scanpy-Filter_HVG-Subset_dt_pca_DT0.75_Dimred0.25_per_celltype_method_a_100,Scanpy-Filter_HVG-Subset_pca_per_celltype_method_a_100,Scanpy-Filter_HVG-Subset_dt_simple_100,Scanpy-Filter_HVG-Subset_dt_deg_100,Scanpy-Filter_HVG-Subset_deg_only_100,Scanpy-Filter_HVG-Subset_hvg_100,Scanpy-Filter_HVG-Subset_random_100,Scanpy-Filter_Spapros_100"
    # Scanpy-Filter_HVG-Subset 200-gene NMF (all with filling suffix)
    "Scanpy-Filter_HVG-Subset_200_NMF_25-75_v2:Scanpy-Filter_HVG-Subset_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_200_cell-type-specific-filling,Scanpy-Filter_HVG-Subset_nmf_per_celltype_method_a_200,Scanpy-Filter_HVG-Subset_dt_simple_200,Scanpy-Filter_HVG-Subset_dt_deg_200,Scanpy-Filter_HVG-Subset_deg_only_200,Scanpy-Filter_HVG-Subset_hvg_200,Scanpy-Filter_HVG-Subset_random_200,Scanpy-Filter_Spapros_200"
    "Scanpy-Filter_HVG-Subset_200_NMF_50-50_v2:Scanpy-Filter_HVG-Subset_dt_nmf_DT0.5_Dimred0.5_per_celltype_method_a_200_cell-type-specific-filling,Scanpy-Filter_HVG-Subset_nmf_per_celltype_method_a_200,Scanpy-Filter_HVG-Subset_dt_simple_200,Scanpy-Filter_HVG-Subset_dt_deg_200,Scanpy-Filter_HVG-Subset_deg_only_200,Scanpy-Filter_HVG-Subset_hvg_200,Scanpy-Filter_HVG-Subset_random_200,Scanpy-Filter_Spapros_200"
    "Scanpy-Filter_HVG-Subset_200_NMF_75-25_v2:Scanpy-Filter_HVG-Subset_dt_nmf_DT0.75_Dimred0.25_per_celltype_method_a_200_cell-type-specific-filling,Scanpy-Filter_HVG-Subset_nmf_per_celltype_method_a_200,Scanpy-Filter_HVG-Subset_dt_simple_200,Scanpy-Filter_HVG-Subset_dt_deg_200,Scanpy-Filter_HVG-Subset_deg_only_200,Scanpy-Filter_HVG-Subset_hvg_200,Scanpy-Filter_HVG-Subset_random_200,Scanpy-Filter_Spapros_200"
    # Scanpy-Filter_HVG-Subset 200-gene PCA (all with filling suffix)
    "Scanpy-Filter_HVG-Subset_200_PCA_25-75_v2:Scanpy-Filter_HVG-Subset_dt_pca_DT0.25_Dimred0.75_per_celltype_method_a_200_cell-type-specific-filling,Scanpy-Filter_HVG-Subset_pca_per_celltype_method_a_200,Scanpy-Filter_HVG-Subset_dt_simple_200,Scanpy-Filter_HVG-Subset_dt_deg_200,Scanpy-Filter_HVG-Subset_deg_only_200,Scanpy-Filter_HVG-Subset_hvg_200,Scanpy-Filter_HVG-Subset_random_200,Scanpy-Filter_Spapros_200"
    "Scanpy-Filter_HVG-Subset_200_PCA_50-50_v2:Scanpy-Filter_HVG-Subset_dt_pca_DT0.5_Dimred0.5_per_celltype_method_a_200_cell-type-specific-filling,Scanpy-Filter_HVG-Subset_pca_per_celltype_method_a_200,Scanpy-Filter_HVG-Subset_dt_simple_200,Scanpy-Filter_HVG-Subset_dt_deg_200,Scanpy-Filter_HVG-Subset_deg_only_200,Scanpy-Filter_HVG-Subset_hvg_200,Scanpy-Filter_HVG-Subset_random_200,Scanpy-Filter_Spapros_200"
    "Scanpy-Filter_HVG-Subset_200_PCA_75-25_v2:Scanpy-Filter_HVG-Subset_dt_pca_DT0.75_Dimred0.25_per_celltype_method_a_200_cell-type-specific-filling,Scanpy-Filter_HVG-Subset_pca_per_celltype_method_a_200,Scanpy-Filter_HVG-Subset_dt_simple_200,Scanpy-Filter_HVG-Subset_dt_deg_200,Scanpy-Filter_HVG-Subset_deg_only_200,Scanpy-Filter_HVG-Subset_hvg_200,Scanpy-Filter_HVG-Subset_random_200,Scanpy-Filter_Spapros_200"
    # Xenium-Filter_All-Genes 100-gene NMF (all with filling suffix for v2-RF)
    "Xenium-Filter_All-Genes_100_NMF_25-75_v2:Xenium-Filter_All-Genes_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_100_cell-type-specific-filling,Xenium-Filter_All-Genes_nmf_per_celltype_method_a_100,Xenium-Filter_All-Genes_dt_simple_100,Xenium-Filter_All-Genes_dt_deg_100,Xenium-Filter_All-Genes_deg_only_100,Xenium-Filter_All-Genes_hvg_100,Xenium-Filter_All-Genes_random_100,Xenium-Filter_Spapros_100"
    "Xenium-Filter_All-Genes_100_NMF_50-50_v2:Xenium-Filter_All-Genes_dt_nmf_DT0.5_Dimred0.5_per_celltype_method_a_100_cell-type-specific-filling,Xenium-Filter_All-Genes_nmf_per_celltype_method_a_100,Xenium-Filter_All-Genes_dt_simple_100,Xenium-Filter_All-Genes_dt_deg_100,Xenium-Filter_All-Genes_deg_only_100,Xenium-Filter_All-Genes_hvg_100,Xenium-Filter_All-Genes_random_100,Xenium-Filter_Spapros_100"
    "Xenium-Filter_All-Genes_100_NMF_75-25_v2:Xenium-Filter_All-Genes_dt_nmf_DT0.75_Dimred0.25_per_celltype_method_a_100_cell-type-specific-filling,Xenium-Filter_All-Genes_nmf_per_celltype_method_a_100,Xenium-Filter_All-Genes_dt_simple_100,Xenium-Filter_All-Genes_dt_deg_100,Xenium-Filter_All-Genes_deg_only_100,Xenium-Filter_All-Genes_hvg_100,Xenium-Filter_All-Genes_random_100,Xenium-Filter_Spapros_100"
    # Xenium-Filter_All-Genes 100-gene PCA (DT0.5 has suffix; DT0.25 and DT0.75 do not)
    "Xenium-Filter_All-Genes_100_PCA_25-75_v2:Xenium-Filter_All-Genes_dt_pca_DT0.25_Dimred0.75_per_celltype_method_a_100,Xenium-Filter_All-Genes_pca_per_celltype_method_a_100,Xenium-Filter_All-Genes_dt_simple_100,Xenium-Filter_All-Genes_dt_deg_100,Xenium-Filter_All-Genes_deg_only_100,Xenium-Filter_All-Genes_hvg_100,Xenium-Filter_All-Genes_random_100,Xenium-Filter_Spapros_100"
    "Xenium-Filter_All-Genes_100_PCA_50-50_v2:Xenium-Filter_All-Genes_dt_pca_DT0.5_Dimred0.5_per_celltype_method_a_100_cell-type-specific-filling,Xenium-Filter_All-Genes_pca_per_celltype_method_a_100,Xenium-Filter_All-Genes_dt_simple_100,Xenium-Filter_All-Genes_dt_deg_100,Xenium-Filter_All-Genes_deg_only_100,Xenium-Filter_All-Genes_hvg_100,Xenium-Filter_All-Genes_random_100,Xenium-Filter_Spapros_100"
    "Xenium-Filter_All-Genes_100_PCA_75-25_v2:Xenium-Filter_All-Genes_dt_pca_DT0.75_Dimred0.25_per_celltype_method_a_100,Xenium-Filter_All-Genes_pca_per_celltype_method_a_100,Xenium-Filter_All-Genes_dt_simple_100,Xenium-Filter_All-Genes_dt_deg_100,Xenium-Filter_All-Genes_deg_only_100,Xenium-Filter_All-Genes_hvg_100,Xenium-Filter_All-Genes_random_100,Xenium-Filter_Spapros_100"
    # Xenium-Filter_All-Genes 200-gene NMF (all with filling suffix)
    "Xenium-Filter_All-Genes_200_NMF_25-75_v2:Xenium-Filter_All-Genes_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_200_cell-type-specific-filling,Xenium-Filter_All-Genes_nmf_per_celltype_method_a_200,Xenium-Filter_All-Genes_dt_simple_200,Xenium-Filter_All-Genes_dt_deg_200,Xenium-Filter_All-Genes_deg_only_200,Xenium-Filter_All-Genes_hvg_200,Xenium-Filter_All-Genes_random_200,Xenium-Filter_Spapros_200"
    "Xenium-Filter_All-Genes_200_NMF_50-50_v2:Xenium-Filter_All-Genes_dt_nmf_DT0.5_Dimred0.5_per_celltype_method_a_200_cell-type-specific-filling,Xenium-Filter_All-Genes_nmf_per_celltype_method_a_200,Xenium-Filter_All-Genes_dt_simple_200,Xenium-Filter_All-Genes_dt_deg_200,Xenium-Filter_All-Genes_deg_only_200,Xenium-Filter_All-Genes_hvg_200,Xenium-Filter_All-Genes_random_200,Xenium-Filter_Spapros_200"
    "Xenium-Filter_All-Genes_200_NMF_75-25_v2:Xenium-Filter_All-Genes_dt_nmf_DT0.75_Dimred0.25_per_celltype_method_a_200_cell-type-specific-filling,Xenium-Filter_All-Genes_nmf_per_celltype_method_a_200,Xenium-Filter_All-Genes_dt_simple_200,Xenium-Filter_All-Genes_dt_deg_200,Xenium-Filter_All-Genes_deg_only_200,Xenium-Filter_All-Genes_hvg_200,Xenium-Filter_All-Genes_random_200,Xenium-Filter_Spapros_200"
    # Xenium-Filter_All-Genes 200-gene PCA (all with filling suffix for v2-RF)
    "Xenium-Filter_All-Genes_200_PCA_25-75_v2:Xenium-Filter_All-Genes_dt_pca_DT0.25_Dimred0.75_per_celltype_method_a_200_cell-type-specific-filling,Xenium-Filter_All-Genes_pca_per_celltype_method_a_200,Xenium-Filter_All-Genes_dt_simple_200,Xenium-Filter_All-Genes_dt_deg_200,Xenium-Filter_All-Genes_deg_only_200,Xenium-Filter_All-Genes_hvg_200,Xenium-Filter_All-Genes_random_200,Xenium-Filter_Spapros_200"
    "Xenium-Filter_All-Genes_200_PCA_50-50_v2:Xenium-Filter_All-Genes_dt_pca_DT0.5_Dimred0.5_per_celltype_method_a_200_cell-type-specific-filling,Xenium-Filter_All-Genes_pca_per_celltype_method_a_200,Xenium-Filter_All-Genes_dt_simple_200,Xenium-Filter_All-Genes_dt_deg_200,Xenium-Filter_All-Genes_deg_only_200,Xenium-Filter_All-Genes_hvg_200,Xenium-Filter_All-Genes_random_200,Xenium-Filter_Spapros_200"
    "Xenium-Filter_All-Genes_200_PCA_75-25_v2:Xenium-Filter_All-Genes_dt_pca_DT0.75_Dimred0.25_per_celltype_method_a_200_cell-type-specific-filling,Xenium-Filter_All-Genes_pca_per_celltype_method_a_200,Xenium-Filter_All-Genes_dt_simple_200,Xenium-Filter_All-Genes_dt_deg_200,Xenium-Filter_All-Genes_deg_only_200,Xenium-Filter_All-Genes_hvg_200,Xenium-Filter_All-Genes_random_200,Xenium-Filter_Spapros_200"
    # Xenium-Filter_HVG-Subset 100-gene NMF (DT0.25 and DT0.5 have suffix; DT0.75 does not)
    "Xenium-Filter_HVG-Subset_100_NMF_25-75_v2:Xenium-Filter_HVG-Subset_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_100_cell-type-specific-filling,Xenium-Filter_HVG-Subset_nmf_per_celltype_method_a_100,Xenium-Filter_HVG-Subset_dt_simple_100,Xenium-Filter_HVG-Subset_dt_deg_100,Xenium-Filter_HVG-Subset_deg_only_100,Xenium-Filter_HVG-Subset_hvg_100,Xenium-Filter_HVG-Subset_random_100,Xenium-Filter_Spapros_100"
    "Xenium-Filter_HVG-Subset_100_NMF_50-50_v2:Xenium-Filter_HVG-Subset_dt_nmf_DT0.5_Dimred0.5_per_celltype_method_a_100,Xenium-Filter_HVG-Subset_nmf_per_celltype_method_a_100,Xenium-Filter_HVG-Subset_dt_simple_100,Xenium-Filter_HVG-Subset_dt_deg_100,Xenium-Filter_HVG-Subset_deg_only_100,Xenium-Filter_HVG-Subset_hvg_100,Xenium-Filter_HVG-Subset_random_100,Xenium-Filter_Spapros_100"
    "Xenium-Filter_HVG-Subset_100_NMF_75-25_v2:Xenium-Filter_HVG-Subset_dt_nmf_DT0.75_Dimred0.25_per_celltype_method_a_100,Xenium-Filter_HVG-Subset_nmf_per_celltype_method_a_100,Xenium-Filter_HVG-Subset_dt_simple_100,Xenium-Filter_HVG-Subset_dt_deg_100,Xenium-Filter_HVG-Subset_deg_only_100,Xenium-Filter_HVG-Subset_hvg_100,Xenium-Filter_HVG-Subset_random_100,Xenium-Filter_Spapros_100"
    # Xenium-Filter_HVG-Subset 100-gene PCA (all have filling suffix for v2-RF)
    "Xenium-Filter_HVG-Subset_100_PCA_25-75_v2:Xenium-Filter_HVG-Subset_dt_pca_DT0.25_Dimred0.75_per_celltype_method_a_100_cell-type-specific-filling,Xenium-Filter_HVG-Subset_pca_per_celltype_method_a_100,Xenium-Filter_HVG-Subset_dt_simple_100,Xenium-Filter_HVG-Subset_dt_deg_100,Xenium-Filter_HVG-Subset_deg_only_100,Xenium-Filter_HVG-Subset_hvg_100,Xenium-Filter_HVG-Subset_random_100,Xenium-Filter_Spapros_100"
    "Xenium-Filter_HVG-Subset_100_PCA_50-50_v2:Xenium-Filter_HVG-Subset_dt_pca_DT0.5_Dimred0.5_per_celltype_method_a_100_cell-type-specific-filling,Xenium-Filter_HVG-Subset_pca_per_celltype_method_a_100,Xenium-Filter_HVG-Subset_dt_simple_100,Xenium-Filter_HVG-Subset_dt_deg_100,Xenium-Filter_HVG-Subset_deg_only_100,Xenium-Filter_HVG-Subset_hvg_100,Xenium-Filter_HVG-Subset_random_100,Xenium-Filter_Spapros_100"
    "Xenium-Filter_HVG-Subset_100_PCA_75-25_v2:Xenium-Filter_HVG-Subset_dt_pca_DT0.75_Dimred0.25_per_celltype_method_a_100_cell-type-specific-filling,Xenium-Filter_HVG-Subset_pca_per_celltype_method_a_100,Xenium-Filter_HVG-Subset_dt_simple_100,Xenium-Filter_HVG-Subset_dt_deg_100,Xenium-Filter_HVG-Subset_deg_only_100,Xenium-Filter_HVG-Subset_hvg_100,Xenium-Filter_HVG-Subset_random_100,Xenium-Filter_Spapros_100"
    # Xenium-Filter_HVG-Subset 200-gene NMF (all with filling suffix)
    "Xenium-Filter_HVG-Subset_200_NMF_25-75_v2:Xenium-Filter_HVG-Subset_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_200_cell-type-specific-filling,Xenium-Filter_HVG-Subset_nmf_per_celltype_method_a_200,Xenium-Filter_HVG-Subset_dt_simple_200,Xenium-Filter_HVG-Subset_dt_deg_200,Xenium-Filter_HVG-Subset_deg_only_200,Xenium-Filter_HVG-Subset_hvg_200,Xenium-Filter_HVG-Subset_random_200,Xenium-Filter_Spapros_200"
    "Xenium-Filter_HVG-Subset_200_NMF_50-50_v2:Xenium-Filter_HVG-Subset_dt_nmf_DT0.5_Dimred0.5_per_celltype_method_a_200_cell-type-specific-filling,Xenium-Filter_HVG-Subset_nmf_per_celltype_method_a_200,Xenium-Filter_HVG-Subset_dt_simple_200,Xenium-Filter_HVG-Subset_dt_deg_200,Xenium-Filter_HVG-Subset_deg_only_200,Xenium-Filter_HVG-Subset_hvg_200,Xenium-Filter_HVG-Subset_random_200,Xenium-Filter_Spapros_200"
    "Xenium-Filter_HVG-Subset_200_NMF_75-25_v2:Xenium-Filter_HVG-Subset_dt_nmf_DT0.75_Dimred0.25_per_celltype_method_a_200_cell-type-specific-filling,Xenium-Filter_HVG-Subset_nmf_per_celltype_method_a_200,Xenium-Filter_HVG-Subset_dt_simple_200,Xenium-Filter_HVG-Subset_dt_deg_200,Xenium-Filter_HVG-Subset_deg_only_200,Xenium-Filter_HVG-Subset_hvg_200,Xenium-Filter_HVG-Subset_random_200,Xenium-Filter_Spapros_200"
    # Xenium-Filter_HVG-Subset 200-gene PCA (all with filling suffix)
    "Xenium-Filter_HVG-Subset_200_PCA_25-75_v2:Xenium-Filter_HVG-Subset_dt_pca_DT0.25_Dimred0.75_per_celltype_method_a_200_cell-type-specific-filling,Xenium-Filter_HVG-Subset_pca_per_celltype_method_a_200,Xenium-Filter_HVG-Subset_dt_simple_200,Xenium-Filter_HVG-Subset_dt_deg_200,Xenium-Filter_HVG-Subset_deg_only_200,Xenium-Filter_HVG-Subset_hvg_200,Xenium-Filter_HVG-Subset_random_200,Xenium-Filter_Spapros_200"
    "Xenium-Filter_HVG-Subset_200_PCA_50-50_v2:Xenium-Filter_HVG-Subset_dt_pca_DT0.5_Dimred0.5_per_celltype_method_a_200_cell-type-specific-filling,Xenium-Filter_HVG-Subset_pca_per_celltype_method_a_200,Xenium-Filter_HVG-Subset_dt_simple_200,Xenium-Filter_HVG-Subset_dt_deg_200,Xenium-Filter_HVG-Subset_deg_only_200,Xenium-Filter_HVG-Subset_hvg_200,Xenium-Filter_HVG-Subset_random_200,Xenium-Filter_Spapros_200"
    "Xenium-Filter_HVG-Subset_200_PCA_75-25_v2:Xenium-Filter_HVG-Subset_dt_pca_DT0.75_Dimred0.25_per_celltype_method_a_200_cell-type-specific-filling,Xenium-Filter_HVG-Subset_pca_per_celltype_method_a_200,Xenium-Filter_HVG-Subset_dt_simple_200,Xenium-Filter_HVG-Subset_dt_deg_200,Xenium-Filter_HVG-Subset_deg_only_200,Xenium-Filter_HVG-Subset_hvg_200,Xenium-Filter_HVG-Subset_random_200,Xenium-Filter_Spapros_200"
)

# ---------------------------------------------------------------------------
# Modules v2 – RF – Category 4: Best configuration vs benchmarks (with add-on)
# ---------------------------------------------------------------------------
CATEGORY_4_PLOT_GROUPS_v2_RF_Eval=(
    "Scanpy-Filter_All-Genes_100_mMulti_NMF_25-75_v2:Scanpy-Filter_All-Genes_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_100_mMulti-addon,Scanpy-Filter_All-Genes_nmf_per_celltype_method_a_100_mMulti-addon,Scanpy-Filter_All-Genes_dt_simple_100_mMulti-addon,Scanpy-Filter_All-Genes_dt_deg_100_mMulti-addon,Scanpy-Filter_All-Genes_deg_only_100_mMulti-addon,Scanpy-Filter_Spapros_100_mMulti-addon,mMulti_v1"
    "Scanpy-Filter_All-Genes_100_5k_NMF_25-75_v2:Scanpy-Filter_All-Genes_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_100_5k-addon,Scanpy-Filter_All-Genes_nmf_per_celltype_method_a_100_5k-addon,Scanpy-Filter_All-Genes_dt_simple_100_5k-addon,Scanpy-Filter_All-Genes_dt_deg_100_5k-addon,Scanpy-Filter_All-Genes_deg_only_100_5k-addon,Scanpy-Filter_Spapros_100_5k-addon,5k"
    "Scanpy-Filter_HVG-Subset_100_mMulti_NMF_25-75_v2:Scanpy-Filter_HVG-Subset_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_100_cell-type-specific-filling_mMulti-addon,Scanpy-Filter_HVG-Subset_nmf_per_celltype_method_a_100_mMulti-addon,Scanpy-Filter_HVG-Subset_dt_simple_100_mMulti-addon,Scanpy-Filter_HVG-Subset_dt_deg_100_mMulti-addon,Scanpy-Filter_HVG-Subset_deg_only_100_mMulti-addon,Scanpy-Filter_Spapros_100_mMulti-addon,mMulti_v1"
    "Scanpy-Filter_HVG-Subset_100_5k_NMF_25-75_v2:Scanpy-Filter_HVG-Subset_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_100_cell-type-specific-filling_5k-addon,Scanpy-Filter_HVG-Subset_nmf_per_celltype_method_a_100_5k-addon,Scanpy-Filter_HVG-Subset_dt_simple_100_5k-addon,Scanpy-Filter_HVG-Subset_dt_deg_100_5k-addon,Scanpy-Filter_HVG-Subset_deg_only_100_5k-addon,Scanpy-Filter_Spapros_100_5k-addon,5k"
    "Xenium-Filter_All-Genes_100_mMulti_NMF_25-75_v2:Xenium-Filter_All-Genes_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_100_cell-type-specific-filling_mMulti-addon,Xenium-Filter_All-Genes_nmf_per_celltype_method_a_100_mMulti-addon,Xenium-Filter_All-Genes_dt_simple_100_mMulti-addon,Xenium-Filter_All-Genes_dt_deg_100_mMulti-addon,Xenium-Filter_All-Genes_deg_only_100_mMulti-addon,Xenium-Filter_Spapros_100_mMulti-addon,mMulti_v1"
    "Xenium-Filter_All-Genes_100_5k_NMF_25-75_v2:Xenium-Filter_All-Genes_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_100_cell-type-specific-filling_5k-addon,Xenium-Filter_All-Genes_nmf_per_celltype_method_a_100_5k-addon,Xenium-Filter_All-Genes_dt_simple_100_5k-addon,Xenium-Filter_All-Genes_dt_deg_100_5k-addon,Xenium-Filter_All-Genes_deg_only_100_5k-addon,Xenium-Filter_Spapros_100_5k-addon,5k"
    "Xenium-Filter_HVG-Subset_100_mMulti_NMF_25-75_v2:Xenium-Filter_HVG-Subset_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_100_cell-type-specific-filling_mMulti-addon,Xenium-Filter_HVG-Subset_nmf_per_celltype_method_a_100_mMulti-addon,Xenium-Filter_HVG-Subset_dt_simple_100_mMulti-addon,Xenium-Filter_HVG-Subset_dt_deg_100_mMulti-addon,Xenium-Filter_HVG-Subset_deg_only_100_mMulti-addon,Xenium-Filter_Spapros_100_mMulti-addon,mMulti_v1"
    "Xenium-Filter_HVG-Subset_100_5k_NMF_25-75_v2:Xenium-Filter_HVG-Subset_dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a_100_cell-type-specific-filling_5k-addon,Xenium-Filter_HVG-Subset_nmf_per_celltype_method_a_100_5k-addon,Xenium-Filter_HVG-Subset_dt_simple_100_5k-addon,Xenium-Filter_HVG-Subset_dt_deg_100_5k-addon,Xenium-Filter_HVG-Subset_deg_only_100_5k-addon,Xenium-Filter_Spapros_100_5k-addon,5k"
)

# =============================================================================
# CATEGORY PLOTTING CONFIGURATION
# =============================================================================
# Format: "ARRAY_NAME:EVALUATION_TYPE:OUTPUT_SUBDIR:PLOT_FLAGS"
#
# ARRAY_NAME      – name of the bash array defined above
# EVALUATION_TYPE – "baseline" or "variability"
# OUTPUT_SUBDIR   – subdirectory under PLOTS_OUTPUT_DIR for this category's output
# PLOT_FLAGS      – comma-separated: clustering, neighborhood, celltype
#                                    mechanistic, celltype_specific
#
# Comment/uncomment entries to control which categories are processed.
CATEGORY_CONFIGS=(
    # Modules v2 – Random forest (no 10x add-on) – Category 1
    "CATEGORY_1_PLOT_GROUPS_v2_RF_Eval:baseline:Baseline/Category-1:clustering,neighborhood,celltype"
    #"CATEGORY_1_PLOT_GROUPS_v2_RF_Eval:variability:Variability/Category-1:mechanistic,celltype_specific"

    # Modules v2 – Random forest (with 10x add-on) – Category 4
    #"CATEGORY_4_PLOT_GROUPS_v2_RF_Eval:baseline:Baseline/Category-4:clustering,neighborhood,celltype"
    #"CATEGORY_4_PLOT_GROUPS_v2_RF_Eval:variability:Variability/Category-4:mechanistic,celltype_specific"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Plot a single category: iterates over the category array and calls
# plot_evaluation.py once per plot group.
plot_category() {
    local category_array_name="$1"
    local evaluation_type="$2"
    local output_subdir="$3"
    local plot_flags="$4"

    # Access the array by name
    local -n category_array="$category_array_name"

    if [[ ${#category_array[@]} -eq 0 ]]; then
        log_message "Skipping $category_array_name (empty or not defined)"
        return
    fi

    # Resolve results directory
    local results_dir
    if [[ "$evaluation_type" == "baseline" ]]; then
        results_dir="$BASELINE_RESULTS_DIR"
    elif [[ "$evaluation_type" == "variability" ]]; then
        results_dir="$VARIABILITY_RESULTS_DIR"
    else
        log_message "✗ Unknown evaluation type: $evaluation_type"
        return 1
    fi

    if [[ ! -d "$results_dir" ]]; then
        log_message "Skipping $category_array_name (results directory not found: $results_dir)"
        return
    fi

    log_message "=== PLOTTING: $category_array_name ==="
    log_message "Evaluation type: $evaluation_type"
    log_message "Output subdirectory: $output_subdir"
    log_message "Plot flags: $plot_flags"
    echo ""

    for plot_group in "${category_array[@]}"; do
        local group_name="${plot_group%%:*}"
        local panels="${plot_group#*:}"

        log_message "Creating plots for group: $group_name"
        log_message "  Panels: $panels"

        # Build base command
        local cmd=(
            "$ENV_PYTHON" "$PLOTTING_SCRIPT"
            --evaluation_type "$evaluation_type"
            --results_dir    "$results_dir"
            --output_dir     "${PLOTS_OUTPUT_DIR}/${output_subdir}/${group_name}"
            --panels         "$panels"
            --group_name     "$group_name"
            --png_dpi        "$PNG_DPI"
        )

        # External names
        if [[ ${#EXTERNAL_NAMES[@]} -gt 0 ]]; then
            local external_csv
            external_csv=$(IFS=,; echo "${EXTERNAL_NAMES[*]}")
            cmd+=(--external_names "$external_csv")
        fi

        # Selectively add plot-type flags based on global switches
        IFS=',' read -ra FLAGS <<< "$plot_flags"
        for flag in "${FLAGS[@]}"; do
            case "$flag" in
                clustering)      [[ "$PLOT_CLUSTERING"       == "true" ]] && cmd+=(--plot_clustering) ;;
                neighborhood)    [[ "$PLOT_NEIGHBORHOOD"     == "true" ]] && cmd+=(--plot_neighborhood) ;;
                celltype)        [[ "$PLOT_CELLTYPE"         == "true" ]] && cmd+=(--plot_celltype) ;;
                mechanistic)     [[ "$PLOT_MECHANISTIC"      == "true" ]] && cmd+=(--plot_mechanistic) ;;
                celltype_specific) [[ "$PLOT_CELLTYPE_SPECIFIC" == "true" ]] && cmd+=(--plot_celltype_specific) ;;
                *) log_message "⚠ Unknown plot flag: $flag" ;;
            esac
        done

        if [[ "$DRY_RUN" == "true" ]]; then
            log_message "[DRY RUN] Would run: ${cmd[*]}"
        else
            log_message "Running: ${cmd[*]}"
            if "${cmd[@]}"; then
                log_message "✓ Successfully created plots for $group_name"
            else
                log_message "✗ Failed to create plots for $group_name"
            fi
        fi
        echo ""
    done
}

# =============================================================================
# VALIDATION
# =============================================================================

log_message "Validating configuration..."

if [[ ! -f "$ENV_PYTHON" ]]; then
    log_message "ERROR: Python interpreter not found: $ENV_PYTHON"
    exit 1
fi

if [[ ! -f "$PLOTTING_SCRIPT" ]]; then
    log_message "ERROR: Plotting script not found: $PLOTTING_SCRIPT"
    log_message "  Expected: Modules_v2/Plotting-module/plot_evaluation.py"
    exit 1
fi

# Gracefully disable plot types when their results directory is absent
if [[ ! -d "$BASELINE_RESULTS_DIR" ]] && \
   [[ "$PLOT_CLUSTERING" == "true" || "$PLOT_NEIGHBORHOOD" == "true" || "$PLOT_CELLTYPE" == "true" ]]; then
    log_message "WARNING: Baseline results directory not found: $BASELINE_RESULTS_DIR"
    log_message "  Disabling baseline plot types for this run."
    PLOT_CLUSTERING=false
    PLOT_NEIGHBORHOOD=false
    PLOT_CELLTYPE=false
fi

if [[ ! -d "$VARIABILITY_RESULTS_DIR" ]] && \
   [[ "$PLOT_MECHANISTIC" == "true" ]]; then
    log_message "WARNING: Variability results directory not found: $VARIABILITY_RESULTS_DIR"
    log_message "  Disabling variability plot types for this run."
    PLOT_MECHANISTIC=false
    PLOT_CELLTYPE_SPECIFIC=false
fi

mkdir -p "$PLOTS_OUTPUT_DIR"

log_message "Configuration validated."
log_message "Output directory: $PLOTS_OUTPUT_DIR"
echo ""

# =============================================================================
# PROCESS CATEGORIES
# =============================================================================

for config in "${CATEGORY_CONFIGS[@]}"; do
    IFS=':' read -r category_name eval_type output_subdir plot_flags <<< "$config"
    plot_category "$category_name" "$eval_type" "$output_subdir" "$plot_flags"
done

# =============================================================================
# SUMMARY
# =============================================================================

log_message "=== PLOTTING PIPELINE COMPLETED ==="
log_message "Plots saved to: $PLOTS_OUTPUT_DIR"
log_message "Finished at $(date)"
