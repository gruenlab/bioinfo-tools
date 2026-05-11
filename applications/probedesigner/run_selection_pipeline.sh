#!/bin/bash
#SBATCH --job-name=Selector
#SBATCH --output=logs/Run-Selection_LCA_no-Xenium-filter_%j.out
#SBATCH --error=logs/Run-Selection_LCA_no-Xenium-filter_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --nodelist=electrode
#SBATCH --mem=80G

# ===================================================================
# COMBINED PREPROCESSING + SELECTION PIPELINE
# ===================================================================
# This script runs preprocessing (optional) followed by gene selection.
# 
# Preprocessing is SKIPPED if preprocessed files already exist.
# This allows you to:
# 1. Run full pipeline from scratch (preprocessing + selection)
# 2. Re-run selection only (if preprocessed files exist)
# 3. Update preprocessing and re-run selection
#
# Author: Helene Hemmer
# Date: 2026-02-08
# ===================================================================

echo "=== Combined Preprocessing + Selection Pipeline ==="
echo "Starting at $(date)"

# ===================================================================
# CONDA ENVIRONMENT ACTIVATION
# ===================================================================

CONDA_ENV="ProbeDesigner"

if [ -f "/home/gruengroup/helene/helene/.conda/etc/profile.d/conda.sh" ]; then
    source "/home/gruengroup/helene/helene/.conda/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ Activated conda environment: $CONDA_ENV"
    else
        echo "Warning: Could not activate conda environment"
        export PATH="/home/gruengroup/helene/helene/.conda/envs/$CONDA_ENV/bin:$PATH"
    fi
else
    echo "Warning: conda.sh not found"
    export PATH="/home/gruengroup/helene/helene/.conda/envs/$CONDA_ENV/bin:$PATH"
fi

echo "Python: $(which python)"
echo ""

# ===================================================================
# CONFIGURATION
# ===================================================================

# Python executable
ENV_PYTHON="python"

# Scripts
PREPROCESS_SCRIPT="/home/gruengroup/helene/helene/SpatialProbeDesign_tmp/Code/Modules_v2/Preprocessing-module/preprocess_for_selection.py"
SELECTION_SCRIPT="/home/gruengroup/helene/helene/SpatialProbeDesign_tmp/Code/Modules_v2/Selection-module/run_selection_pipeline.py"

# Data paths
RAW_DATA="/home/gruengroup/helene/helene/SpatialProbeDesign_tmp/Data/LCA/raw/LCA_raw.h5ad"
PREPROCESSED_DIR="/home/gruengroup/helene/helene/SpatialProbeDesign_tmp/Data/LCA/Selection-pipeline/preprocessed/"
BASE_OUTPUT_DIR="/home/gruengroup/helene/helene/SpatialProbeDesign_tmp/Experiments/LCA/Test-kfold-vs-simple-split_no-Xenium-filter/Selection-pipeline/"

# Analysis parameters
CELLTYPE_COLUMN="annot"
MIN_CELLS_PER_CELLTYPE=10

# Preprocessing parameters
FILTER_METHODS=("scanpy")  # Options: scanpy, no_filter
N_COMPONENTS_PCA=50
N_COMPONENTS_NMF=5
RANDOM_STATE=42

# Directory structure parameters (matches existing architecture)
# Structure: {BASE_OUTPUT_DIR}/{FILTER_METHOD}/{HVG_OPTION}/{STRATEGY_NAME}/{PROBESET_SIZE}/results|data
FILTER_METHOD="Scanpy-Filter"  # Capitalized to match existing structure
HVG_OPTION="All-Genes"          # Options: "All-Genes" or "HVG-Subset"

# Selection parameters
STRATEGIES=('rf_nmf')  # 'deg_only' 'rf_simple' 'rf_deg', 'hvg', 'random', 'dimred_only', 'rf_nmf', 'rf_pca'
PROBESET_SIZES=(100)
REDUCTION_TYPES=("nmf") #"pca"
ANALYSIS_TYPES=("per_celltype")
DIMRED_METHODS=("method_a")

# Filtering parameters
DISABLE_DEFAULT_BLACKLIST=false
CUSTOM_BLACKLIST=()
FORCE_INCLUDE_GENES=()

DEFAULT_MIN_XENIUM_EXPRESSION=0.1
DEFAULT_MAX_XENIUM_EXPRESSION=100.0
RUN_XENIUM_FILTERING=false   # true = apply Xenium expression filter (keep genes with mean 0.1-100 counts/cell); false = no filtering
XENIUM_CELLTYPE_AWARE=false  # true = check expression per assigned celltype; false = check against global mean across all cells
                             # (only applies when RUN_XENIUM_FILTERING=true)

# Combination strategy parameters
RF_PERCENTAGES=(0.25)
DIMRED_PERCENTAGES=(0.75)
RUN_CELLTYPE_SPECIFIC_FILLING=true
RUN_GLOBAL_GENE_FILLING=false
RUN_DEG_BASED_FILLING=false

# Create log directory (use absolute path for SLURM)
SCRIPT_DIR="/home/gruengroup/helene/helene/SpatialProbeDesign_tmp/Code/Shell-scripts"
mkdir -p "$SCRIPT_DIR/logs"

# ===================================================================
# STEP 1: PREPROCESSING (Optional - skip if exists)
# ===================================================================

echo "================================================================================"
echo "STEP 1: PREPROCESSING"
echo "================================================================================"
echo ""

# Map HVG_OPTION to Python parameter format
case "$HVG_OPTION" in
    "All-Genes")
        HVG_PARAM="all_genes"
        ;;
    "HVG-Subset")
        HVG_PARAM="hvg"
        ;;
    *)
        echo "ERROR: Unknown HVG_OPTION: $HVG_OPTION"
        echo "Valid options: 'All-Genes' or 'HVG-Subset'"
        exit 1
        ;;
esac

# Check if preprocessed files exist
PREPROCESSED_FILE="$PREPROCESSED_DIR/${FILTER_METHOD}_${HVG_OPTION}/preprocessed.h5ad"

if [ -f "$PREPROCESSED_FILE" ]; then
    echo "✓ Preprocessed data already exists: $PREPROCESSED_FILE"
    echo "Skipping preprocessing step."
    echo ""
    echo "To force re-preprocessing, delete: $PREPROCESSED_DIR"
    echo ""
else
    echo "Preprocessed data not found. Running preprocessing..."
    echo ""

    "$ENV_PYTHON" "$PREPROCESS_SCRIPT" \
        --input_file "$RAW_DATA" \
        --output_dir "$PREPROCESSED_DIR" \
        --celltype_column "$CELLTYPE_COLUMN" \
        --filter_methods "${FILTER_METHODS[@]}" \
        --hvg_option "$HVG_PARAM" \
        --n_components_pca "$N_COMPONENTS_PCA" \
        --n_components_nmf "$N_COMPONENTS_NMF" \
        --random_state "$RANDOM_STATE"
    
    preprocess_exit_code=$?
    
    if [ $preprocess_exit_code -ne 0 ]; then
        echo "ERROR: Preprocessing failed with exit code $preprocess_exit_code"
        exit $preprocess_exit_code
    fi
    
    echo "✓ Preprocessing complete"
    echo ""
fi

# ===================================================================
# STEP 2: GENE SELECTION
# ===================================================================
# 
# OUTPUT DIRECTORY STRUCTURE:
# {BASE_OUTPUT_DIR}/{FILTER_METHOD}/{HVG_OPTION}/{STRATEGY_NAME}/{PROBESET_SIZE}-genes/results|data
# 
# Example paths:
# - Scanpy-Filter/All-Genes/random/100-genes/results/
# - Scanpy-Filter/All-Genes/dt_nmf_DT0.25_Dimred0.75_per_celltype_method_a/100-genes/results/
# - Scanpy-Filter/All-Genes/dimred_only_nmf_per_celltype_method_a/100-genes/results/
# 
# Strategy naming conventions:
# - Simple strategies: {strategy} (e.g., "random", "deg_only", "hvg")
# - Dimred-only: dimred_only_{reduction_type}_{analysis_type}_{dimred_method}
# - Combination: {strategy}_DT{rf_pct}_Dimred{dimred_pct}_{analysis_type}_{dimred_method}
# ===================================================================

echo "================================================================================"
echo "STEP 2: GENE SELECTION"
echo "================================================================================"
echo ""

selection_count=0
selection_failed=0

for strategy in "${STRATEGIES[@]}"; do
    for probeset_size in "${PROBESET_SIZES[@]}"; do
        
        # Strategy-specific configuration
        if [[ "$strategy" == "dimred_only" ]]; then
            # Run for each reduction type
            for reduction_type in "${REDUCTION_TYPES[@]}"; do
                for analysis_type in "${ANALYSIS_TYPES[@]}"; do
                    for dimred_method in "${DIMRED_METHODS[@]}"; do
                        
                        selection_count=$((selection_count + 1))
                        
                        # Strategy naming: dimred_only_{reduction_type}_{analysis_type}_{dimred_method}
                        strategy_name="dimred_only_${reduction_type}_${analysis_type}_${dimred_method}"
                        output_dir="$BASE_OUTPUT_DIR/$FILTER_METHOD/$HVG_OPTION/$strategy_name/${probeset_size}-genes"
                        exp_name="$strategy_name"
                        
                        echo ""
                        echo "Running: $strategy_name (${probeset_size} genes)"
                        
                        cmd=(
                            "$ENV_PYTHON" "$SELECTION_SCRIPT"
                            --strategy "$strategy"
                            --input_file "$PREPROCESSED_FILE"
                            --output_dir "$output_dir"
                            --experiment_name "$exp_name"
                            --probeset_size "$probeset_size"
                            --celltype_column "$CELLTYPE_COLUMN"
                            --min_cells_per_celltype "$MIN_CELLS_PER_CELLTYPE"
                            --reduction_type "$reduction_type"
                            --analysis_type "$analysis_type"
                            --dimred_method "$dimred_method"
                            --n_components "$N_COMPONENTS_NMF"
                            --random_state "$RANDOM_STATE"
                        )
                        
                        # Add filtering parameters
                        if [ "$RUN_XENIUM_FILTERING" = false ]; then
                            cmd+=(--disable_xenium_filter)
                        fi

                        if [ "$XENIUM_CELLTYPE_AWARE" = false ]; then
                            cmd+=(--disable_xenium_celltype_aware)
                        fi
                        
                        if [ "$DISABLE_DEFAULT_BLACKLIST" = true ]; then
                            cmd+=(--disable_default_blacklist)
                        fi
                        
                        if [ ${#CUSTOM_BLACKLIST[@]} -gt 0 ]; then
                            cmd+=(--blacklist_patterns "${CUSTOM_BLACKLIST[@]}")
                        fi
                        
                        if [ ${#FORCE_INCLUDE_GENES[@]} -gt 0 ]; then
                            cmd+=(--force_include_genes "${FORCE_INCLUDE_GENES[@]}")
                        fi
                        
                        "${cmd[@]}"
                        
                        if [ $? -ne 0 ]; then
                            echo "✗ Failed: $exp_name"
                            selection_failed=$((selection_failed + 1))
                        else
                            echo "✓ Complete: $exp_name"
                        fi
                    done
                done
            done
            
        elif [[ "$strategy" == "rf_nmf" || "$strategy" == "rf_pca" ]]; then
            # Combination strategies
            for rf_pct in "${RF_PERCENTAGES[@]}"; do
                for dimred_pct in "${DIMRED_PERCENTAGES[@]}"; do
                    
                    reduction_type="${strategy#rf_}"  # Extract nmf or pca
                    
                    for analysis_type in "${ANALYSIS_TYPES[@]}"; do
                        for dimred_method in "${DIMRED_METHODS[@]}"; do
                            
                            selection_count=$((selection_count + 1))
                            
                            # Strategy naming for combination: {strategy}_DT{rf_pct}_Dimred{dimred_pct}_{analysis_type}_{dimred_method}
                            # Keep percentages as decimals (0.25 -> 0.25)
                            strategy_name="${strategy}_DT${rf_pct}_Dimred${dimred_pct}_${analysis_type}_${dimred_method}"
                            output_dir="$BASE_OUTPUT_DIR/$FILTER_METHOD/$HVG_OPTION/$strategy_name/${probeset_size}-genes"
                            exp_name="$strategy_name"
                            
                            echo ""
                            echo "Running: $strategy_name (${probeset_size} genes)"
                            
                            cmd=(
                                "$ENV_PYTHON" "$SELECTION_SCRIPT"
                                --strategy "$strategy"
                                --input_file "$PREPROCESSED_FILE"
                                --output_dir "$output_dir"
                                --experiment_name "$exp_name"
                                --probeset_size "$probeset_size"
                                --celltype_column "$CELLTYPE_COLUMN"
                                --min_cells_per_celltype "$MIN_CELLS_PER_CELLTYPE"
                                --reduction_type "$reduction_type"
                                --analysis_type "$analysis_type"
                                --dimred_method "$dimred_method"
                                --n_components "$N_COMPONENTS_NMF"
                                --rf_percentage "$rf_pct"
                                --dimred_percentage "$dimred_pct"
                                --random_state "$RANDOM_STATE"
                            )
                            
                            # Add filtering parameters
                            if [ "$RUN_XENIUM_FILTERING" = false ]; then
                                cmd+=(--disable_xenium_filter)
                            fi

                            if [ "$XENIUM_CELLTYPE_AWARE" = false ]; then
                                cmd+=(--disable_xenium_celltype_aware)
                            fi
                            
                            if [ "$DISABLE_DEFAULT_BLACKLIST" = true ]; then
                                cmd+=(--disable_default_blacklist)
                            fi
                            
                            if [ ${#CUSTOM_BLACKLIST[@]} -gt 0 ]; then
                                cmd+=(--blacklist_patterns "${CUSTOM_BLACKLIST[@]}")
                            fi
                            
                            if [ ${#FORCE_INCLUDE_GENES[@]} -gt 0 ]; then
                                cmd+=(--force_include_genes "${FORCE_INCLUDE_GENES[@]}")
                            fi
                            
                            # Add gap-filling options
                            if [ "$RUN_CELLTYPE_SPECIFIC_FILLING" = false ]; then
                                cmd+=(--disable_celltype_filling)
                            fi
                            if [ "$RUN_GLOBAL_GENE_FILLING" = false ]; then
                                cmd+=(--disable_global_filling)
                            fi
                            if [ "$RUN_DEG_BASED_FILLING" = false ]; then
                                cmd+=(--disable_deg_filling)
                            fi
                            
                            "${cmd[@]}"
                            
                            if [ $? -ne 0 ]; then
                                echo "✗ Failed: $exp_name"
                                selection_failed=$((selection_failed + 1))
                            else
                                echo "✓ Complete: $exp_name"
                            fi
                        done
                    done
                done
            done
        
        else
            # Simple strategies (random, deg_only, hvg, etc.)
            # Strategy naming: just the strategy name
            selection_count=$((selection_count + 1))
            
            strategy_name="$strategy"
            output_dir="$BASE_OUTPUT_DIR/$FILTER_METHOD/$HVG_OPTION/$strategy_name/${probeset_size}-genes"
            exp_name="$strategy_name"
            
            echo ""
            echo "Running: $strategy_name (${probeset_size} genes)"
            
            cmd=(
                "$ENV_PYTHON" "$SELECTION_SCRIPT"
                --strategy "$strategy"
                --input_file "$PREPROCESSED_FILE"
                --output_dir "$output_dir"
                --experiment_name "$exp_name"
                --probeset_size "$probeset_size"
                --celltype_column "$CELLTYPE_COLUMN"
                --min_cells_per_celltype "$MIN_CELLS_PER_CELLTYPE"
                --random_state "$RANDOM_STATE"
            )
            
            # Add filtering parameters
            if [ "$RUN_XENIUM_FILTERING" = false ]; then
                cmd+=(--disable_xenium_filter)
            fi

            if [ "$XENIUM_CELLTYPE_AWARE" = false ]; then
                cmd+=(--disable_xenium_celltype_aware)
            fi
            
            if [ "$DISABLE_DEFAULT_BLACKLIST" = true ]; then
                cmd+=(--disable_default_blacklist)
            fi
            
            if [ ${#CUSTOM_BLACKLIST[@]} -gt 0 ]; then
                cmd+=(--blacklist_patterns "${CUSTOM_BLACKLIST[@]}")
            fi
            
            if [ ${#FORCE_INCLUDE_GENES[@]} -gt 0 ]; then
                cmd+=(--force_include_genes "${FORCE_INCLUDE_GENES[@]}")
            fi
            
            "${cmd[@]}"
            
            if [ $? -ne 0 ]; then
                echo "✗ Failed: $exp_name"
                selection_failed=$((selection_failed + 1))
            else
                echo "✓ Complete: $exp_name"
            fi
        fi
    done
done

# ===================================================================
# SUMMARY
# ===================================================================

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================================"
echo "Total selections run: $selection_count"
echo "Successful: $((selection_count - selection_failed))"
echo "Failed: $selection_failed"
echo "Results saved to: $BASE_OUTPUT_DIR"
echo "Finished at $(date)"

if [ $selection_failed -gt 0 ]; then
    exit 1
else
    exit 0
fi
