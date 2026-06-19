#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Constant config files path (resolved from repo root)
CONFIG_DIR="${PROJECT_ROOT}/experiments/segmentation/allmodels"
LOG_DIR="${PROJECT_ROOT}/logs/slurm_out/segmentation_allmodels_new"
SBATCH_SCRIPT="${SCRIPT_DIR}/launch_single_experiment.sbatch"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory not found: $CONFIG_DIR"
    exit 1
fi

if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "Error: sbatch script not found: $SBATCH_SCRIPT"
    exit 1
fi

# Ensure log directory exists
mkdir -p "$LOG_DIR"

shopt -s nullglob
CONFIG_FILES=("$CONFIG_DIR"/*.yaml "$CONFIG_DIR"/*.yml)

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    echo "Error: No .yaml or .yml files found in: $CONFIG_DIR"
    exit 1
fi

for config_file in "${CONFIG_FILES[@]}"; do

    # Extract experiment name from filename (e.g. debug.yaml -> debug)
    exp_name=$(basename "$config_file")
    exp_name="${exp_name%.*}"
    
    echo "Submitting job for experiment: $exp_name (Config: $config_file)"
    
    # Launch job with overridden name and output
    # Pass config_file as the first argument to the sbatch script
    sbatch \
        --job-name="$exp_name" \
        --output="${LOG_DIR}/${exp_name}_%j.log" \
        --chdir="$PROJECT_ROOT" \
        --export=NEURO_BRAIN_PROJECT_ROOT="$PROJECT_ROOT" \
        "$SBATCH_SCRIPT" "$config_file"
        
    echo "Job submitted for $exp_name"
    #wait 2 seconds between submissions to avoid overloading the scheduler
    echo "Waiting 2 seconds before next submission..."
    sleep 2
done

# Final message
echo "All jobs submitted."

echo "Current user's jobs in the queue:"
echo "--------------------------------"
squeue --me
