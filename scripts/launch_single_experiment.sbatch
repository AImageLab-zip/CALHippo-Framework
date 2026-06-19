#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
# Submit with site-specific options when needed, for example:
#   sbatch --partition=<partition> --account=<account> scripts/launch_single_experiment.sbatch <config>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "$NEURO_BRAIN_PROJECT_ROOT" ]; then
    PROJECT_ROOT="$NEURO_BRAIN_PROJECT_ROOT"
elif [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "${SLURM_SUBMIT_DIR}/src" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [ -d "${SCRIPT_DIR}/../src" ]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "Error: Could not determine project root."
    echo "Set NEURO_BRAIN_PROJECT_ROOT or submit from the repository root."
    exit 1
fi

if [ ! -f "${PROJECT_ROOT}/src/segmentation/multimodel_inference.py" ]; then
    echo "Error: Project root does not look valid: $PROJECT_ROOT"
    exit 1
fi

VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"

if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
else
    echo "Warning: Virtual environment not found at $VENV_ACTIVATE; using current Python."
fi

cd "$PROJECT_ROOT" || exit 1

# Environment variables for clean logging
export TQDM_MININTERVAL=30
export TERM=dumb
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow logs
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1


echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_NAME} started on ${HOSTNAME}"

start_time=$(date +%s)

# Check if arguments are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config_file> [additional_args...]"
    exit 1
fi

CONFIG_FILE=$1
shift # remove config file from args to pass the rest to the python script

# Resolve config path from current path or from project root
if [ ! -f "$CONFIG_FILE" ] && [ -f "${PROJECT_ROOT}/${CONFIG_FILE}" ]; then
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Using Python from: $(which python)"
echo "Script started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Project root: $PROJECT_ROOT"
echo "Config file: $CONFIG_FILE"

# Execute inference
if command -v uv >/dev/null 2>&1; then
    PYTHONPATH=. uv run python ./src/segmentation/multimodel_inference.py \
        --config "$CONFIG_FILE" "$@"
else
    PYTHONPATH=. python ./src/segmentation/multimodel_inference.py \
        --config "$CONFIG_FILE" "$@"
fi

status=$?

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_NAME} finished."
echo "Runtime: ${runtime} seconds ($((runtime / 60)) min)"
exit "$status"
