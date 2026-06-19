from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_REGIONS = ("RCA1", "RCA2", "RCA3", "RCA4")
DEFAULT_SLEEP_SECONDS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and submit SLURM sbatch files for allmodels segmentation "
            "experiments."
        )
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=DEFAULT_REGIONS,
        default=list(DEFAULT_REGIONS),
        help="Regions to submit. Defaults to all four RCA regions.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds to sleep between sbatch submissions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated sbatch files without writing or submitting them.",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Write sbatch files but do not submit them.",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="Optional SLURM partition to write into generated sbatch files.",
    )
    parser.add_argument(
        "--account",
        default=None,
        help="Optional SLURM account to write into generated sbatch files.",
    )
    parser.add_argument(
        "--constraint",
        default=None,
        help="Optional SLURM node constraint to write into generated sbatch files.",
    )
    return parser.parse_args()


def build_sbatch_content(
    *,
    project_root: Path,
    config_file: Path,
    log_dir: Path,
    region: str,
    partition: str | None,
    account: str | None,
    constraint: str | None,
) -> str:
    job_name = f"seg_allmodels_{region}"
    output_file = log_dir / f"{job_name}_%j.out"
    error_file = log_dir / f"{job_name}_%j.err"
    venv_activate = project_root / ".venv" / "bin" / "activate"
    inference_script = project_root / "src" / "segmentation" / "multimodel_inference.py"
    optional_sbatch_lines = []
    if partition:
        optional_sbatch_lines.append(f"#SBATCH --partition={partition}")
    if account:
        optional_sbatch_lines.append(f"#SBATCH --account={account}")
    if constraint:
        optional_sbatch_lines.append(f"#SBATCH --constraint={constraint}")
    optional_sbatch_block = "\n".join(optional_sbatch_lines)

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
{optional_sbatch_block}

set -u

PROJECT_ROOT="{project_root}"
CONFIG_FILE="{config_file}"
VENV_ACTIVATE="{venv_activate}"
INFERENCE_SCRIPT="{inference_script}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$INFERENCE_SCRIPT" ]; then
    echo "Error: Inference script not found: $INFERENCE_SCRIPT"
    exit 1
fi

if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "Error: Virtual environment not found: $VENV_ACTIVATE"
    exit 1
fi

source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT" || exit 1

JOB_LABEL="${{SLURM_JOB_NAME:-{job_name}}}"

export TQDM_MININTERVAL=30
export TERM=dumb
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $JOB_LABEL started on ${{HOSTNAME}}"
echo "Project root: $PROJECT_ROOT"
echo "Config file: $CONFIG_FILE"
echo "Working directory: $(pwd)"
echo "Using Python from: $(which python)"

start_time=$(date +%s)

PYTHONPATH=. python ./src/segmentation/multimodel_inference.py --config "$CONFIG_FILE"
status=$?

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $JOB_LABEL finished with status ${{status}}."
echo "Runtime: ${{runtime}} seconds ($((runtime / 60)) min)"
exit "$status"
"""


def validate_inputs(project_root: Path, config_dir: Path, regions: list[str]) -> None:
    inference_script = project_root / "src" / "segmentation" / "multimodel_inference.py"
    venv_activate = project_root / ".venv" / "bin" / "activate"

    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    if not inference_script.is_file():
        raise FileNotFoundError(f"Inference script not found: {inference_script}")
    if not venv_activate.is_file():
        raise FileNotFoundError(f"Virtual environment not found: {venv_activate}")

    for region in regions:
        config_file = config_dir / f"allmodels-{region}.yaml"
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_file}")


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    config_dir = project_root / "experiments" / "segmentation" / "allmodels"
    sbatch_dir = project_root / "slurm_sbatch_files" / "segmentation_allmodels"
    log_dir = project_root / "logs" / "slurm_out" / "segmentation_allmodels"

    try:
        validate_inputs(project_root, config_dir, args.regions)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Project root: {project_root}")
    print(f"Config directory: {config_dir}")
    print(f"SBATCH directory: {sbatch_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Regions: {', '.join(args.regions)}")
    print()

    if not args.dry_run:
        sbatch_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    for region in args.regions:
        config_file = config_dir / f"allmodels-{region}.yaml"
        sbatch_file = sbatch_dir / f"seg_allmodels_{region}.sbatch"
        content = build_sbatch_content(
            project_root=project_root,
            config_file=config_file,
            log_dir=log_dir,
            region=region,
            partition=args.partition,
            account=args.account,
            constraint=args.constraint,
        )

        if args.dry_run:
            print(f"DRY RUN: {sbatch_file}")
            print(content)
            continue

        sbatch_file.write_text(content)
        print(f"Wrote {sbatch_file}")

        if args.no_submit:
            continue

        result = subprocess.run(
            ["sbatch", str(sbatch_file)],
            check=True,
            text=True,
            capture_output=True,
        )
        print(result.stdout.strip())
        time.sleep(args.sleep_seconds)

    if not args.dry_run and not args.no_submit:
        subprocess.run(["squeue", "--me"], check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
