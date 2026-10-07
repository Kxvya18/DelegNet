#!/usr/bin/env bash
# All paper experiments, most decisive first. Needs built graphs (scripts/run_pipeline.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/paper.yaml}
export PYTHONPATH=.
PY=".venv/bin/python -W ignore run.py"
$PY pretrain   --config "$CFG"
$PY experiment --config "$CFG" --protocol zeroday
$PY experiment --config "$CFG" --protocol temporal
$PY experiment --config "$CFG" --protocol family
until [ -f data/full/bsc/graph.pkl ] && [ -f data/full/base/graph.pkl ]; do sleep 120; done   # target chains built
$PY experiment --config "$CFG" --protocol crosschain
$PY analyse    --config "$CFG"
$PY report     --config "$CFG"
echo "EXPERIMENTS_DONE"
