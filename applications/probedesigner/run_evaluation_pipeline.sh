#!/bin/bash
#SBATCH --job-name=Eval_LCA_dt_nmf_100
#SBATCH --output=logs/eval-comparison_LCA_selection-v1-scanpy-filter_%j.out
#SBATCH --error=logs/eval-comparison_LCA_selection-v1-scanpy-filter_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G

# =============================================================================
# Evaluation Pipeline (Modules v2)
# =============================================================================
# Usage: [sbatch] run_evaluation_pipeline.sh
# Configure via variables in the CONFIGURATION section below.
#
# Runs any combination of:
#   preprocess  - Subset panels to selected genes and compute PCA/NMF,
#                 clustering, and KNN graphs
#   evaluate    - Compute baseline and/or variability metrics on
#                 preprocessed panels
#   both        - Preprocessing immediately followed by evaluation
#
# Author: Helene Hemmer
# Date: 2026-03-18
# =============================================================================

set -euo pipefail

echo "=== Evaluation Pipeline (Modules v2) ==="
echo "Starting at $(date)"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Pipeline mode: "preprocess" | "evaluate" | "both"
MODE="both"

# Dry-run: set to "true" to print resolved config without executing
DRY_RUN=false

# --- Paths ---

# Python environment
ENV_PYTHON="/home/gruengroup/helene/helene/.conda/envs/SpaprosProbeDesign/bin/python"

# Path to the v2 evaluation entry point
EVAL_SCRIPT="/home/gruengroup/helene/helene/SpatialProbeDesign_tmp/Code/Modules_v2/Evaluation-module/run_evaluation.py"

# Raw / reference h5ad file (required for MODE=preprocess or MODE=both;
# for MODE=evaluate this should be the full-transcriptome preprocessed file)
INPUT_FILE="/home/gruengroup/helene/helene/SpatialProbeDesign_tmp/Data/LCA/raw/LCA_raw.h5ad"

# Directory containing preprocessed panel .h5ad files
PREPROCESSED_DIR="/home/gruengroup/helene/helene/SpatialProbeDesign_tmp/Experiments/LCA/Test-kfold-vs-simple-split_Selection-pipeline-v1_Reselection/Evaluation-pipeline_Scanpy-Filter/preprocessed"

# Root output directory (Baseline-Evaluation/ and Variability-Evaluation/
# subdirectories will be created automatically by the Python script)
OUTPUT_DIR="/home/gruengroup/helene/helene/SpatialProbeDesign_tmp/Experiments/LCA/Test-kfold-vs-simple-split_Selection-pipeline-v1_Reselection/Evaluation-pipeline_Scanpy-Filter"

# (Optional) Directory containing gene-list CSV files from the selection pipeline.
# Required when MODE=preprocess or MODE=both.
GENE_LISTS_DIR=""
# Example: "/home/gruengroup/helene/SpatialProbeDesign_tmp/Experiments/.../Selected-panels"

# (Optional) Text file listing gene-list CSV paths, one per line.
# Alternative to GENE_LISTS_DIR; only one of the two needs to be set.
GENE_LIST_FILES_TXT="/home/gruengroup/helene/helene/SpatialProbeDesign_tmp/Experiments/LCA/Test-kfold-vs-simple-split_Selection-pipeline-v1_Reselection/Evaluation-pipeline_Scanpy-Filter/gene_list_for_eval.txt"

# --- Analysis parameters ---

# Type of evaluation: "baseline" | "variability" | "both"
EVALUATION_TYPE="variability"

# Cell type annotation column in the h5ad file
CELLTYPE_COLUMN="annot"
# LCA:     "annot"
# Lung:    "celltypes_v2"

# Dimensionality reduction for evaluation metrics: "pca" | "nmf" | "both"
DIMENSIONALITY_REDUCTION="both"

# Dimensionality reduction computed during preprocessing: "pca" | "nmf" | "both"
DIM_REDUCTION_PREPROCESS="both"

# Number of PCA/NMF components (set to the maximum you want to evaluate;
# for factor-range experiments set this to the upper bound)
N_COMPONENTS=5

# Fraction of data held out for test split (0.0–1.0)
TEST_SIZE=0.2

# Random seed for reproducibility
RANDOM_STATE=42

# Split mode: "kfold" (stratified 5-fold) | "simple" (single 80/20 split) | "both" (run each in its own subdirectory)
# When SPLIT_MODE="both", results go to $OUTPUT_DIR/kfold/ and $OUTPUT_DIR/simple/.
# When SPLIT_MODE is a single value, results go to $OUTPUT_DIR/$SPLIT_MODE/.
SPLIT_MODE="both"
N_SPLITS=5

# Number of neighbours for KNN graph construction during preprocessing
N_NEIGHBORS=15

# --- Tangram (optional) ---

# Set to "true" to run the Tangram reconstruction check
INCLUDE_TANGRAM=false
TANGRAM_N_EPOCHS=1000

# --- 10x panel integration ---

# Set to a panel identifier (e.g. "mMulti_v1") or leave as "no-10x-panel"
ADD_10X_PANELS="no-10x-panel"
DATA_DIR=""

# =============================================================================
# DATASET FILTERING PARAMETERS
# =============================================================================
# Leave arrays empty to process ALL matching datasets found in PREPROCESSED_DIR.

# Selection strategies to include (e.g. "dt_nmf" "dt_pca" "deg_only")
STRATEGIES=("dt_nmf")

# Probe-set sizes to include (e.g. 100 200)
PROBESET_SIZES=(100)

# Preprocessing filter methods to include: "scanpy" | "10x" | "no_filter"
FILTER_METHODS=()

# HVG subsetting: "false" (all genes) | "true" (highly variable genes)
HVG_SUBSET_OPTIONS=()

# =============================================================================
# DIMENSIONALITY REDUCTION FILTERING PARAMETERS
# =============================================================================

# Reduction types: "pca" | "nmf"
REDUCTION_TYPES=()

# Analysis types: "per_celltype" | "global"
ANALYSIS_TYPES=()

# Dimred methods: "method_a" | "method_b"
DIMRED_METHODS=()

# =============================================================================
# HYBRID STRATEGY FILTERING PARAMETERS
# =============================================================================

# DT percentages (e.g. 0.25 0.50 0.75)
DT_PERCENTAGES=(0.25)

# Dimred percentages (e.g. 0.75 0.50 0.25)
DIMRED_PERCENTAGES=(0.75)

# Gap-filling strategy control ("true" | "false" | "" for all)
RUN_CELLTYPE_SPECIFIC_FILLING="true"
RUN_GLOBAL_GENE_FILLING=""
RUN_DEG_BASED_FILLING=""

# Preferred gap-filling: "auto" | "celltype_specific" | "global_gene" | "deg_based" | ""
PREFERRED_STRATEGY=""

# =============================================================================
# EXTERNAL PANELS
# =============================================================================
# Simple panel identifiers as returned by extract_strategy_and_display_name()
# (e.g. "Spapros" "mMulti_v1" "NSForest"). Leave empty to include all.
EXTERNAL_NAMES=()

# External panel file paths (parallel arrays with EXTERNAL_NAMES)
EXTERNAL_PANELS=()

# External panel probe-set sizes (parallel array with EXTERNAL_PANELS)
EXTERNAL_PROBESET_SIZES=()

# Filter to only evaluate these external names (subset of EXTERNAL_NAMES)
EXTERNAL_NAMES_FILTER=()

# =============================================================================
# VALIDATION
# =============================================================================

echo "Validating configuration..."

if [[ ! -f "$ENV_PYTHON" ]]; then
    echo "ERROR: Python interpreter not found: $ENV_PYTHON"
    exit 1
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "ERROR: Evaluation script not found: $EVAL_SCRIPT"
    echo "  Expected: Modules_v2/Evaluation-module/run_evaluation.py"
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

if [[ "$MODE" != "preprocess" ]] || [[ "$MODE" == "evaluate" ]] || [[ "$MODE" == "both" ]]; then
    if [[ ! -d "$PREPROCESSED_DIR" ]]; then
        if [[ "$MODE" == "evaluate" ]]; then
            echo "ERROR: Preprocessed directory not found: $PREPROCESSED_DIR"
            echo "  Run with MODE=preprocess first."
            exit 1
        fi
        # For MODE=both, directory may not exist yet; the Python script will create it
        echo "NOTE: Preprocessed directory does not yet exist (will be created by preprocess step): $PREPROCESSED_DIR"
    fi
fi

if [[ "$MODE" == "evaluate" ]]; then
    PREPROCESSED_COUNT=$(ls -1 "${PREPROCESSED_DIR}"/*.h5ad 2>/dev/null | wc -l)
    if [[ $PREPROCESSED_COUNT -eq 0 ]]; then
        echo "ERROR: No .h5ad files found in $PREPROCESSED_DIR"
        echo "  Run with MODE=preprocess first."
        exit 1
    fi
    echo "Found ${PREPROCESSED_COUNT} preprocessed files in ${PREPROCESSED_DIR}"
fi

mkdir -p "$OUTPUT_DIR"

echo "Configuration validated."
echo ""

# =============================================================================
# BUILD COMMAND
# =============================================================================

echo "Building command..."

EVAL_ARGS=(
    "--mode"              "$MODE"
    "--input_file"        "$INPUT_FILE"
    "--preprocessed_dir"  "$PREPROCESSED_DIR"
    "--output_dir"        "$OUTPUT_DIR"
    "--evaluation_type"   "$EVALUATION_TYPE"
    "--celltype_col"      "$CELLTYPE_COLUMN"
    "--dimensionality_reduction" "$DIMENSIONALITY_REDUCTION"
    "--dim_reduction_preprocess" "$DIM_REDUCTION_PREPROCESS"
    "--n_components"      "$N_COMPONENTS"
    "--test_size"         "$TEST_SIZE"
    "--random_state"      "$RANDOM_STATE"
    "--split_mode"        "$SPLIT_MODE"
    "--n_splits"          "$N_SPLITS"
    "--n_neighbors"       "$N_NEIGHBORS"
    "--add_10x_panels"    "$ADD_10X_PANELS"
)

# Optional paths
[[ -n "$GENE_LISTS_DIR"       ]] && EVAL_ARGS+=("--gene_lists_dir"       "$GENE_LISTS_DIR")
[[ -n "$GENE_LIST_FILES_TXT"  ]] && EVAL_ARGS+=("--gene_list_files_txt"  "$GENE_LIST_FILES_TXT")
[[ -n "$DATA_DIR"             ]] && EVAL_ARGS+=("--data_dir"              "$DATA_DIR")

# Tangram
[[ "$INCLUDE_TANGRAM" == "true" ]] && EVAL_ARGS+=("--include_tangram" "--tangram_n_epochs" "$TANGRAM_N_EPOCHS")

# Dataset filters (only added when non-empty)
[[ ${#STRATEGIES[@]}          -gt 0 ]] && EVAL_ARGS+=("--strategies"          "${STRATEGIES[@]}")
[[ ${#PROBESET_SIZES[@]}      -gt 0 ]] && EVAL_ARGS+=("--probeset_sizes"      "${PROBESET_SIZES[@]}")
[[ ${#FILTER_METHODS[@]}      -gt 0 ]] && EVAL_ARGS+=("--filter_methods"      "${FILTER_METHODS[@]}")
[[ ${#HVG_SUBSET_OPTIONS[@]}  -gt 0 ]] && EVAL_ARGS+=("--hvg_subset_options"  "${HVG_SUBSET_OPTIONS[@]}")
[[ ${#REDUCTION_TYPES[@]}     -gt 0 ]] && EVAL_ARGS+=("--reduction_types"     "${REDUCTION_TYPES[@]}")
[[ ${#ANALYSIS_TYPES[@]}      -gt 0 ]] && EVAL_ARGS+=("--analysis_types"      "${ANALYSIS_TYPES[@]}")
[[ ${#DIMRED_METHODS[@]}      -gt 0 ]] && EVAL_ARGS+=("--dimred_methods"      "${DIMRED_METHODS[@]}")
[[ ${#DT_PERCENTAGES[@]}      -gt 0 ]] && EVAL_ARGS+=("--dt_percentages"      "${DT_PERCENTAGES[@]}")
[[ ${#DIMRED_PERCENTAGES[@]}  -gt 0 ]] && EVAL_ARGS+=("--dimred_percentages"  "${DIMRED_PERCENTAGES[@]}")

# Gap-filling filters
[[ -n "$RUN_CELLTYPE_SPECIFIC_FILLING" ]] && EVAL_ARGS+=("--run_celltype_specific_filling" "$RUN_CELLTYPE_SPECIFIC_FILLING")
[[ -n "$RUN_GLOBAL_GENE_FILLING"       ]] && EVAL_ARGS+=("--run_global_gene_filling"       "$RUN_GLOBAL_GENE_FILLING")
[[ -n "$RUN_DEG_BASED_FILLING"         ]] && EVAL_ARGS+=("--run_deg_based_filling"         "$RUN_DEG_BASED_FILLING")
[[ -n "$PREFERRED_STRATEGY"            ]] && EVAL_ARGS+=("--preferred_strategy"            "$PREFERRED_STRATEGY")

# External panels
[[ ${#EXTERNAL_PANELS[@]}          -gt 0 ]] && EVAL_ARGS+=("--external_panels"          "${EXTERNAL_PANELS[@]}")
[[ ${#EXTERNAL_NAMES[@]}           -gt 0 ]] && EVAL_ARGS+=("--external_names"           "${EXTERNAL_NAMES[@]}")
[[ ${#EXTERNAL_PROBESET_SIZES[@]}  -gt 0 ]] && EVAL_ARGS+=("--external_probeset_sizes"  "${EXTERNAL_PROBESET_SIZES[@]}")
[[ ${#EXTERNAL_NAMES_FILTER[@]}    -gt 0 ]] && EVAL_ARGS+=("--external_names_filter"    "${EXTERNAL_NAMES_FILTER[@]}")

# =============================================================================
# DRY-RUN
# =============================================================================

if [[ "$DRY_RUN" == "true" ]]; then
    if [[ "$SPLIT_MODE" == "both" ]]; then
        DRY_MODES=("kfold" "simple")
    else
        DRY_MODES=("$SPLIT_MODE")
    fi
    for DRY_MODE in "${DRY_MODES[@]}"; do
        echo "[DRY RUN] Would execute (split_mode=${DRY_MODE}):"
        echo "  $ENV_PYTHON $EVAL_SCRIPT \\"
        printf '    %s \\\n' "${EVAL_ARGS[@]}"
        echo "    (--output_dir overridden to ${OUTPUT_DIR}/${DRY_MODE})"
        echo "    (--split_mode overridden to ${DRY_MODE})"
        echo ""
    done
    echo "[DRY RUN] No files were modified."
    exit 0
fi

# =============================================================================
# PRINT RESOLVED CONFIGURATION
# =============================================================================

echo "=== Running Evaluation Pipeline ==="
echo "Mode:             $MODE"
echo "Input file:       $INPUT_FILE"
echo "Preprocessed dir: $PREPROCESSED_DIR"
echo "Output dir:       $OUTPUT_DIR"
echo ""
echo "Evaluation settings:"
echo "  Type:                    $EVALUATION_TYPE"
echo "  Cell type column:        $CELLTYPE_COLUMN"
echo "  Dimensionality reduction:$DIMENSIONALITY_REDUCTION"
echo "  N components (NMF):      $N_COMPONENTS"
echo "  Test size:               $TEST_SIZE"
echo "  Random state:            $RANDOM_STATE"
echo "  Split mode:              $SPLIT_MODE"
echo "  N splits (k-fold):       $N_SPLITS"
echo "  Tangram epochs:          $TANGRAM_N_EPOCHS"
echo ""
echo "Dataset filtering:"
[[ ${#STRATEGIES[@]}         -gt 0 ]] && echo "  Strategies:          ${STRATEGIES[*]}"          || echo "  Strategies:          ALL"
[[ ${#PROBESET_SIZES[@]}     -gt 0 ]] && echo "  Probeset sizes:      ${PROBESET_SIZES[*]}"      || echo "  Probeset sizes:      ALL"
[[ ${#FILTER_METHODS[@]}     -gt 0 ]] && echo "  Filter methods:      ${FILTER_METHODS[*]}"      || echo "  Filter methods:      ALL"
[[ ${#HVG_SUBSET_OPTIONS[@]} -gt 0 ]] && echo "  HVG subset options:  ${HVG_SUBSET_OPTIONS[*]}"  || echo "  HVG subset options:  ALL"
[[ ${#REDUCTION_TYPES[@]}    -gt 0 ]] && echo "  Reduction types:     ${REDUCTION_TYPES[*]}"     || echo "  Reduction types:     ALL"
[[ ${#ANALYSIS_TYPES[@]}     -gt 0 ]] && echo "  Analysis types:      ${ANALYSIS_TYPES[*]}"      || echo "  Analysis types:      ALL"
[[ ${#DIMRED_METHODS[@]}     -gt 0 ]] && echo "  Dimred methods:      ${DIMRED_METHODS[*]}"      || echo "  Dimred methods:      ALL"
[[ ${#DT_PERCENTAGES[@]}     -gt 0 ]] && echo "  DT percentages:      ${DT_PERCENTAGES[*]}"      || echo "  DT percentages:      ALL"
[[ ${#DIMRED_PERCENTAGES[@]} -gt 0 ]] && echo "  Dimred percentages:  ${DIMRED_PERCENTAGES[*]}"  || echo "  Dimred percentages:  ALL"
[[ -n "$RUN_CELLTYPE_SPECIFIC_FILLING" ]] && echo "  Celltype-specific filling: $RUN_CELLTYPE_SPECIFIC_FILLING"
[[ -n "$RUN_GLOBAL_GENE_FILLING"       ]] && echo "  Global gene filling:       $RUN_GLOBAL_GENE_FILLING"
[[ -n "$RUN_DEG_BASED_FILLING"         ]] && echo "  DEG-based filling:         $RUN_DEG_BASED_FILLING"
[[ ${#EXTERNAL_NAMES[@]} -gt 0 ]] && echo "  External panels:     ${EXTERNAL_NAMES[*]}"
echo ""

# =============================================================================
# EXECUTE
# =============================================================================

# Determine which split modes to run.
# "both" → run kfold then simple, each in its own subdirectory.
# Single value ("kfold" or "simple") → run once, output to $OUTPUT_DIR/$SPLIT_MODE/.
if [[ "$SPLIT_MODE" == "both" ]]; then
    SPLIT_MODES_TO_RUN=("kfold" "simple")
else
    SPLIT_MODES_TO_RUN=("$SPLIT_MODE")
fi

OVERALL_START=$(date +%s)
FAILED_MODES=()

for CURRENT_SPLIT_MODE in "${SPLIT_MODES_TO_RUN[@]}"; do
    SPLIT_OUTPUT_DIR="${OUTPUT_DIR}/${CURRENT_SPLIT_MODE}"
    mkdir -p "$SPLIT_OUTPUT_DIR"
    mkdir -p "$SPLIT_OUTPUT_DIR/logs"

    echo ""
    echo "--- Split mode: ${CURRENT_SPLIT_MODE} → ${SPLIT_OUTPUT_DIR} ---"

    # Build per-run args: override --split_mode and --output_dir
    RUN_ARGS=("${EVAL_ARGS[@]}")
    # Replace --output_dir value
    for i in "${!RUN_ARGS[@]}"; do
        if [[ "${RUN_ARGS[$i]}" == "--output_dir" ]]; then
            RUN_ARGS[$((i+1))]="$SPLIT_OUTPUT_DIR"
        fi
        if [[ "${RUN_ARGS[$i]}" == "--split_mode" ]]; then
            RUN_ARGS[$((i+1))]="$CURRENT_SPLIT_MODE"
        fi
    done

    START_TIME=$(date +%s)
    if "$ENV_PYTHON" "$EVAL_SCRIPT" "${RUN_ARGS[@]}"; then
        END_TIME=$(date +%s)
        echo "  Completed in $((END_TIME - START_TIME))s → ${SPLIT_OUTPUT_DIR}"
    else
        END_TIME=$(date +%s)
        echo "  FAILED after $((END_TIME - START_TIME))s (split_mode=${CURRENT_SPLIT_MODE})"
        FAILED_MODES+=("$CURRENT_SPLIT_MODE")
    fi
done

OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))
echo ""

if [[ ${#FAILED_MODES[@]} -eq 0 ]]; then
    echo "=== EVALUATION PIPELINE COMPLETED SUCCESSFULLY ==="
    echo "Duration: ${OVERALL_DURATION} seconds"
    for CURRENT_SPLIT_MODE in "${SPLIT_MODES_TO_RUN[@]}"; do
        echo "  ${CURRENT_SPLIT_MODE} results: ${OUTPUT_DIR}/${CURRENT_SPLIT_MODE}"
    done
    exit 0
else
    echo "=== EVALUATION PIPELINE FINISHED WITH ERRORS ==="
    echo "Duration: ${OVERALL_DURATION} seconds"
    echo "Failed split modes: ${FAILED_MODES[*]}"
    echo "Check logs in: $OUTPUT_DIR/<split_mode>/logs/"
    exit 1
fi
