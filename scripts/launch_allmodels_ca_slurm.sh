#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_DIR="${PROJECT_ROOT}/experiments/segmentation/allmodels"
LOG_DIR="${PROJECT_ROOT}/logs/slurm_out/segmentation_allmodels"
SBATCH_SCRIPT="${SCRIPT_DIR}/launch_single_experiment.sbatch"

REGIONS=(RCA1 RCA2 RCA3 RCA4)

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory not found: $CONFIG_DIR"
    exit 1
fi

if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "Error: sbatch script not found: $SBATCH_SCRIPT"
    exit 1
fi

for region in "${REGIONS[@]}"; do
    config_file="${CONFIG_DIR}/allmodels-${region}.yaml"
    if [ ! -f "$config_file" ]; then
        echo "Error: Required config file not found: $config_file"
        exit 1
    fi
done

mkdir -p "$LOG_DIR"

echo "Submitting allmodels segmentation jobs from: $PROJECT_ROOT"
echo "Config directory: $CONFIG_DIR"
echo "Log directory: $LOG_DIR"
echo "SBATCH script: $SBATCH_SCRIPT"
echo

for region in "${REGIONS[@]}"; do
    config_file="${CONFIG_DIR}/allmodels-${region}.yaml"
    job_name="seg_${region}_allmodels"
    log_file="${LOG_DIR}/${region}_%j.log"

    echo "Submitting ${region}: $config_file"
    if ! submit_output=$(sbatch "$SBATCH_SCRIPT" "$config_file"); then
        echo "Error: sbatch submission failed for ${region}."
        exit 1
    fi

    echo "${region}: $submit_output"
    sleep 2
done

echo
echo "All 4 jobs submitted. Current user's jobs in the queue:"
echo "-------------------------------------------------------"
squeue --me
