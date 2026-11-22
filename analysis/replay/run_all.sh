#!/bin/zsh
# The oracle-gate sweeps, from the repo root. Needs a Python with numpy, orjson and zstandard
# (PY=..., default python). Deadlines cover whole sessions and users reply after a fixed 5 s;
# UNIT=turn REPLY=recorded OUT=data/freeinference/gate gives the first runs, with one deadline
# per turn and the trace's own pauses. Results land in OUT (gitignored).
#
#   hf download harvardMadsys/freeinference_agentic_trace --type dataset --local-dir data/freeinference \
#       --include 'release/*/traces/*'
#   analysis/replay/run_all.sh
#   python -m analysis.replay.report data/freeinference/gate_session_reply5/*.jsonl
#   python -m analysis.replay.breakdown data/freeinference/gate_session_reply5
set -e
export PYTHONPATH=.
PY=${PY:-python}
UNIT=${UNIT:-session}
REPLY=${REPLY:-fixed:5}
OUT=${OUT:-data/freeinference/gate_session_reply5}
mkdir -p data/freeinference/compact $OUT

# WEEK__DAY.json: a day at a week boundary is in both weeks' releases, with different sessions
for f in data/freeinference/release/*/traces/*.jsonl.zst; do
  c=data/freeinference/compact/$(basename $(dirname $(dirname $f)))__$(basename $f .jsonl.zst).json
  [[ -f $c ]] || echo "$f $c"
done | xargs -P 3 -n 2 $PY -W ignore analysis/replay/extract.py

# LOADS=0.7,0.9,1.1 KV=0,1000000,2000000,4000000,8000000,16000000 is the full grid; 0 = unlimited memory
LOADS=${LOADS:-0.9}
KV=${KV:-0}
GRID=(--loads $LOADS --kv $KV --unit $UNIT --reply $REPLY --workers ${WORKERS:-8})
$PY analysis/replay/run_gate.py --out $OUT/class_l40s.jsonl --profiles l40s --tight 0.5,1,2 $GRID
$PY analysis/replay/run_gate.py --out $OUT/prop_l40s.jsonl --profiles l40s --mode proportional --tight 1.2,1.5,2,3 $GRID
# a CPU tier only matters once something is evicted
[[ $KV == 0 ]] || $PY analysis/replay/run_gate.py --out $OUT/class_offload.jsonl --profiles l40s_offload --tight 0.5,1,2 $GRID
$PY analysis/replay/run_gate.py --out $OUT/class_trace.jsonl --profiles trace --tight 0.5,1,2 $GRID
$PY analysis/replay/run_gate.py --out $OUT/class_l40s_slots16.jsonl --profiles l40s --slots 16 --tight 0.5,1,2 $GRID
