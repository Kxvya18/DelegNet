#!/usr/bin/env bash
# Paper experiments, most decisive first. Needs built graphs (scripts/run_pipeline.sh <chain>).
#   scripts/run_experiments.sh                     # pretrain + all protocols + analyses + report
#   scripts/run_experiments.sh temporal family     # resume: only these protocols, then analyses + report
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/paper.yaml}
export PYTHONPATH=.
PY=".venv/bin/python -W ignore run.py"
mkdir -p logs
mark() { echo "$(date '+%H:%M:%S') $*" | tee -a logs/experiments.log; }
if [ $# -eq 0 ]; then
  set -- zeroday zeroday_ablation temporal family crosschain
  mark "pretrain"; $PY pretrain --config "$CFG" > logs/pretrain.log 2>&1 || { mark "PRETRAIN FAILED"; exit 1; }
fi
for proto in "$@"; do
  mark "experiment $proto"
  $PY experiment --config "$CFG" --protocol "$proto" > "logs/experiment_$proto.log" 2>&1 || mark "FAILED $proto"
done
mark "analyses"; $PY analyse --config "$CFG" > logs/analyse.log 2>&1 || mark "FAILED analyses"
mark "report";   $PY report  --config "$CFG" > logs/report.log 2>&1 || mark "FAILED report"
mark "EXPERIMENTS_DONE"
