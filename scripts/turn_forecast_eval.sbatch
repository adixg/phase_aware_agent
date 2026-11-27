#!/bin/bash
# How predictable is what is left of the current turn (GPU work, time with tools, requests)? Runs
# scripts/turn_forecast_eval.py on a CPU node: it needs ~10-15 GB, too much for a laptop.
#
#   sbatch scripts/turn_forecast_eval.sbatch
#   TARGETS=rem_work_s sbatch scripts/turn_forecast_eval.sbatch
#   sbatch scripts/turn_forecast_eval.sbatch --smoke              # args go to the script
#
# Submit from the repo root. It uses a worktree of Mallika's branch (origin/track1/proportional-deadline-online)
# for her forecaster code and builds her rows there once (~5 min). Needs the FreeInference release under
# data/freeinference/release (see analysis/dag/README.md on her branch; download on the login node, which
# has internet). Results: results/summary/turn_forecast_<jobid>.{txt,json}.
#SBATCH --job-name=paa-turnfc
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
REPO="$PWD"
WT="${WT:-$HOME/scratch/paa-track1-wt}"
BRANCH="${BRANCH:-origin/track1/proportional-deadline-online}"
TARGETS="${TARGETS:-rem_work_s,rem_path_s,rem_after}"
mkdir -p logs results/summary

command -v conda >/dev/null || module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-$HOME/scratch/conda-envs/paa}"
python -c 'import numpy, orjson, zstandard' || { echo "the env needs numpy, orjson and zstandard"; exit 1; }

[[ -d data/freeinference/release ]] || { echo "no data/freeinference/release: download it first (her README, step 1)"; exit 1; }
if [[ ! -d "$WT" ]]; then
    git fetch -q origin
    git worktree add -q --detach "$WT" "$BRANCH"
fi
echo "track 1 code: $(git -C "$WT" log -1 --oneline)"
mkdir -p "$WT/data/freeinference"
[[ -e "$WT/data/freeinference/release" ]] || ln -s "$REPO/data/freeinference/release" "$WT/data/freeinference/release"

cd "$WT"
export PYTHONPATH="$WT"
if [[ ! -f data/freeinference/dag_all/online_rows.csv.gz ]]; then
    echo "building her rows (once)"
    python -m analysis.dag.build --week all --out data/freeinference/dag_all
    python -m analysis.dag.online data/freeinference/dag_all
fi

OUT="$REPO/results/summary/turn_forecast_${SLURM_JOB_ID}"
python "$REPO/scripts/turn_forecast_eval.py" --targets "$TARGETS" --out "$OUT.json" "$@" | tee "$OUT.txt"
echo "results: results/summary/turn_forecast_${SLURM_JOB_ID}.{txt,json}"
