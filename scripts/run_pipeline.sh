#!/usr/bin/env bash
# Full pipeline for one chain: collection -> labels -> graph. Resumable; safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
CHAIN=${1:-ethereum}
CFG=${CFG:-configs/paper.yaml}
export PYTHONPATH=.
PY=.venv/bin/python
$PY -W ignore run.py collect --config "$CFG" --chain "$CHAIN" --steps code,activity,hop2
$PY -W ignore run.py label   --config "$CFG" --chain "$CHAIN"
$PY -W ignore run.py build   --config "$CFG" --chain "$CHAIN"
echo "PIPELINE_DONE $CHAIN"
