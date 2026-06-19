#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Require a folder argument
if [ -z "$1" ]; then
    echo "Error: Configuration directory not provided."
    echo "Usage: ./scripts/run_yaml_experiments.bash <path_to_yaml_folder>"
    exit 1
fi

INPUT_CONFIG_DIR="$1"

# Resolve input path either from current directory or from project root
if [ -d "$INPUT_CONFIG_DIR" ]; then
    CONFIG_DIR="$(cd "$INPUT_CONFIG_DIR" && pwd)"
elif [ -d "${PROJECT_ROOT}/${INPUT_CONFIG_DIR}" ]; then
    CONFIG_DIR="$(cd "${PROJECT_ROOT}/${INPUT_CONFIG_DIR}" && pwd)"
else
    echo "Error: Directory '$INPUT_CONFIG_DIR' does not exist."
    echo "Checked both '$INPUT_CONFIG_DIR' and '${PROJECT_ROOT}/${INPUT_CONFIG_DIR}'."
    exit 1
fi

cd "$PROJECT_ROOT" || exit 1

# Verify directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Directory '$CONFIG_DIR' does not exist."
    exit 1
fi

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/training_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Enable nullglob so the array is empty if no files match
shopt -s nullglob
YAML_FILES=("$CONFIG_DIR"/*.yaml "$CONFIG_DIR"/*.yml)

# Verify directory contains yaml files
if [ ${#YAML_FILES[@]} -eq 0 ]; then
    echo "Error: No .yaml or .yml files found in $CONFIG_DIR."
    exit 1
fi

echo "Starting training sequence from: $CONFIG_DIR"

for config in "${YAML_FILES[@]}"; do
    echo "Starting experiment with config: $config"
    
    # Execute the script with the current yaml file
    uv run python -m src.density_estimator --config "$config"
    
    # Capture the exit code to determine success/failure without stopping
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS: Configuration $config completed."
    else
        echo "FAILURE: Configuration $config failed with exit code $EXIT_CODE. Moving to next script."
    fi
    echo "----------------------------------------"
done

echo "All scheduled training scripts have been processed."
