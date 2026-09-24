#!/bin/bash
# ---------------------------------------------------------------------------
# check_probe.sh -- report queue wait time and result location for every
# probe job you have submitted. Safe to run repeatedly.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")"

STATE=.probe-state/submissions.txt
if [[ ! -f "$STATE" ]]; then
    echo "No probes submitted yet. Run ./submit_probe.sh first."
    exit 0
fi

printf "%-10s %-6s %-9s %-14s %-14s %s\n" \
       JOBID GPU STATE QUEUE_WAIT RUNTIME RESULTS
printf "%-10s %-6s %-9s %-14s %-14s %s\n" \
       ---------- ------ --------- -------------- -------------- -------

hms() {  # seconds -> Hh Mm Ss
    local s=$1
    printf "%dh%02dm%02ds" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

while read -r jobid gpu sub_ts; do
    [[ -z "${jobid:-}" ]] && continue

    # Current state from squeue; if absent, the job has finished or failed.
    st=$(squeue -j "$jobid" -h -o "%T" 2>/dev/null | head -1)
    started_file=".probe-state/${jobid}.started"

    if [[ -z "$st" ]]; then
        st=$(sacct -j "$jobid" -n -o State -X 2>/dev/null | head -1 | tr -d ' ')
        [[ -z "$st" ]] && st="GONE"
    fi

    # Queue wait: prefer the timestamp the job itself wrote the moment it
    # began executing. Fall back to "still waiting, counting from now".
    if [[ -f "$started_file" ]]; then
        start_ts=$(cat "$started_file")
        qwait=$(( start_ts - sub_ts ))
        qstr=$(hms "$qwait")
        now=$(date +%s)
        if [[ "$st" == "RUNNING" ]]; then
            rstr=$(hms $(( now - start_ts )))
        else
            endf=".probe-state/${jobid}.ended"
            if [[ -f "$endf" ]]; then
                rstr=$(hms $(( $(cat "$endf") - start_ts )))
            else
                rstr="-"
            fi
        fi
    else
        qwait=$(( $(date +%s) - sub_ts ))
        qstr="$(hms "$qwait")+"   # trailing + = still accumulating
        rstr="-"
    fi

    resdir="results/probe-${jobid}"
    [[ -d "$resdir" ]] || resdir="(pending)"

    printf "%-10s %-6s %-9s %-14s %-14s %s\n" \
           "$jobid" "$gpu" "$st" "$qstr" "$rstr" "$resdir"
done < "$STATE"

echo
echo "Reason a pending job is waiting (blank = starting soon):"
squeue -u "$USER" -o "  %.10i  %R" -h 2>/dev/null || true
echo
echo "Read a finished probe with:"
echo "  cat results/probe-<JOBID>/SUMMARY.txt"
