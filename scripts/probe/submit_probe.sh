#!/bin/bash
# ---------------------------------------------------------------------------
# submit_probe.sh -- submit the allocation probe and measure how long the
# scheduler actually makes you wait for each GPU type.
#
# Usage, from your scratch directory:
#     ./submit_probe.sh                 # A100, the recommended target
#     ./submit_probe.sh a100            # same
#     ./submit_probe.sh l40s            # fallback if A100 queue is long
#     ./submit_probe.sh h100            # for comparison
#     ./submit_probe.sh all             # submit one of each, race them
#
# This script does NOT block. It submits, records the submission timestamp,
# and then you run ./check_probe.sh to see queue wait + results.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs results .probe-state

GPUTYPE="${1:-a100}"

submit_one() {
    local gpu="$1"
    local stamp
    stamp=$(date +%s)

    echo "--- submitting probe for GPU type: $gpu ---"

    # Pass the GPU type through so the job script can adapt its checks.
    local jobid
    jobid=$(sbatch --parsable \
                   --job-name="probe-${gpu}" \
                   --gres="gpu:${gpu}:1" \
                   --export=ALL,PROBE_GPU="${gpu}" \
                   probe_env.sbatch)

    # Record submission time so we can compute true queue wait later. SLURM's
    # own accounting can be configured off on some clusters, so we keep our
    # own record rather than relying on sacct being available.
    echo "${jobid} ${gpu} ${stamp}" >> .probe-state/submissions.txt
    echo "  job id      : ${jobid}"
    echo "  submitted at: $(date -d "@${stamp}" '+%Y-%m-%d %H:%M:%S')"
    echo "  watch with  : ./check_probe.sh"
    echo
}

case "$GPUTYPE" in
    all)
        for g in a100 l40s h100; do submit_one "$g"; done
        echo "All three submitted. Whichever starts first tells you which GPU"
        echo "type to build the real experiment grid around."
        ;;
    *)
        submit_one "$GPUTYPE"
        ;;
esac

echo "Queue snapshot right now:"
squeue -u "$USER" -o "%.10i %.16j %.10P %.8T %.10M %.12L %R" || true
