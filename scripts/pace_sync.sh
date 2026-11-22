#!/usr/bin/env bash
# Laptop side of running the replay sweeps on PACE ICE: send inputs, submit, fetch results.
# Run from the repo root. REMOTE_REPO is the repo's path on the cluster (quote any ~ so the
# cluster expands it, not your laptop).
#
#   export REMOTE_REPO='~/scratch/phase_aware_agent'
#   scripts/pace_sync.sh push-data            # data/freeinference/compact -> cluster (~77 MB, once)
#   scripts/pace_sync.sh push-code            # analysis/ from this working tree -> cluster (uncommitted changes too)
#   scripts/pace_sync.sh submit               # sbatch the main sweep; SWEEPS=all and any sbatch flags after it
#   scripts/pace_sync.sh status               # squeue for your jobs
#   scripts/pace_sync.sh pull                 # gate results and job logs -> data/freeinference/gate, logs/
#
#   SWEEPS=all scripts/pace_sync.sh submit --time=02:00:00 --cpus-per-task=32
#   DRY_RUN=1 scripts/pace_sync.sh pull       # print the commands instead of running them
#
# PACE_HOST defaults to login-ice.pace.gatech.edu; your username comes from ~/.ssh/config. All
# commands share one SSH connection kept open for 30 minutes, so a two-factor prompt, if you get
# one, comes once and not once per command (works on macOS and Linux; needs OpenSSH).
set -euo pipefail

HOST=${PACE_HOST:-login-ice.pace.gatech.edu}
: "${REMOTE_REPO:?set REMOTE_REPO to the repo path on the cluster, e.g. export REMOTE_REPO='~/scratch/phase_aware_agent'}"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/pace-%C" -o ControlPersist=30m)
RUN=""; [[ -n ${DRY_RUN:-} ]] && RUN=echo
rsh() { $RUN ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }
rsync_() { $RUN rsync -avz --partial -e "ssh ${SSH_OPTS[*]}" "$@"; }

cmd=${1:-}; shift || true
case $cmd in
  push-data)
    [[ -n $RUN ]] || [[ $(ls data/freeinference/compact/*.json 2>/dev/null | wc -l) -ge 14 ]] || \
        { echo "need the 14 files in data/freeinference/compact (run analysis/replay/extract.py first)" >&2; exit 1; }
    rsh "mkdir -p $REMOTE_REPO/data/freeinference/compact"
    rsync_ data/freeinference/compact/ "$HOST:$REMOTE_REPO/data/freeinference/compact/"
    ;;
  push-code)
    # the code the sweep runs; the cluster's git checkout is otherwise left alone
    rsync_ --exclude __pycache__ analysis/replay/ "$HOST:$REMOTE_REPO/analysis/replay/"
    ;;
  submit)
    # sbatch copies the environment of the shell it runs in, so SWEEPS is set on the cluster side
    rsh "cd $REMOTE_REPO && mkdir -p logs && SWEEPS='${SWEEPS:-class_l40s}' sbatch $* analysis/replay/run_gate.sbatch"
    ;;
  status)
    rsh "squeue --me -o '%.10i %.12j %.9P %.8T %.10M %.6D %R'"
    ;;
  pull)
    mkdir -p data/freeinference/gate logs
    rsync_ "$HOST:$REMOTE_REPO/data/freeinference/gate/" data/freeinference/gate/
    rsync_ --include 'paa-gate-*.out' --exclude '*' "$HOST:$REMOTE_REPO/logs/" logs/
    echo "next: python -m analysis.replay.report data/freeinference/gate/class_l40s.jsonl"
    ;;
  *)
    sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2
    exit 2
    ;;
esac
