#!/usr/bin/env bash
# Public scammer-address lists used only to cross-check our behavioural labels.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p labels
curl -sfL -o labels/scamsniffer_address.json \
  https://raw.githubusercontent.com/scamsniffer/scam-database/main/blacklist/address.json
python - <<'PY'
import json, pandas as pd
d = json.load(open("labels/scamsniffer_address.json"))
pd.DataFrame({"address": [a.lower() for a in d], "source": "scamsniffer", "name": ""}) \
  .drop_duplicates().to_csv("labels/scam_addresses.csv", index=False)
print(len(d), "addresses -> labels/scam_addresses.csv")
PY
