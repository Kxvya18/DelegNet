#!/usr/bin/env bash
# Paper experiments, most decisive first. Needs built graphs (scripts/run_pipeline.sh <chain>).
#   scripts/run_experiments.sh                     # pretrain + all protocols + analyses + report
#   scripts/run_experiments.sh temporal family     # resume: only these protocols, then analyses + report
# Each stage runs under a watchdog: if its log is silent for STALL_MIN minutes (e.g. a hung GPU
# command buffer on macOS) the stage is killed and restarted; finished folds/methods are checkpointed.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .env.local ] && source .env.local          # machine-specific data paths (not in git)
CFG=${CFG:-configs/paper.yaml}
STALL_MIN=${STALL_MIN:-20}
MAX_RESTARTS=${MAX_RESTARTS:-5}
export PYTHONPATH=.
mkdir -p logs
mark() { echo "$(date '+%H:%M:%S') $*" | tee -a logs/experiments.log; }

watched() {   # watched <logfile> <run.py args...>
  local logf=$1; shift
  for attempt in $(seq 0 "$MAX_RESTARTS"); do
    touch "$logf"            # stall timer starts now, not at the log's last write from an earlier run
    .venv/bin/python -W ignore run.py "$@" --config "$CFG" >> "$logf" 2>&1 &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      sleep 60
      if [ -n "$(find "$logf" -mmin +"$STALL_MIN" 2>/dev/null)" ]; then
        mark "stall in '$*' (attempt $attempt) - restarting"
        kill -9 "$pid" 2>/dev/null
        break                  # a killed child stays a zombie until waited on, so leave the loop
      fi
    done
    wait "$pid" && return 0
    grep -q "Traceback" <(tail -40 "$logf") && ! grep -q "stall" <(tail -1 logs/experiments.log) && return 1
  done
  return 1
}

if [ $# -eq 0 ]; then
  # full run: results are regenerated from scratch; the code encoder and tuning are reused if present
  set -- zeroday zeroday_ablation temporal family crosschain
  RUNS=$(.venv/bin/python -c "from delegnet.config import load_config; print(load_config('$CFG')['run_root'])")
  rm -rf "$RUNS"/{zeroday,zeroday_ablation,temporal,family,crosschain,diagnostics,report}
  if [ ! -f "$RUNS/code_encoder.pt" ]; then
    mark "pretrain"; watched logs/pretrain.log pretrain || { mark "FAILED pretrain"; exit 1; }
  fi
  if [ ! -f "$RUNS/tuning/best.json" ]; then
    mark "tune"; watched logs/tune.log tune || { mark "FAILED tune"; exit 1; }
  fi
fi
for proto in "$@"; do
  mark "experiment $proto"
  watched "logs/experiment_$proto.log" experiment --protocol "$proto" || mark "FAILED $proto"
done
mark "diagnose"; watched logs/diagnose.log diagnose || mark "FAILED diagnose"
mark "analyses"; watched logs/analyse.log analyse || mark "FAILED analyses"
mark "report";   watched logs/report.log report  || mark "FAILED report"
mark "EXPERIMENTS_DONE"
