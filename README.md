# DelegNet

Early detection of malicious EIP-7702 delegations with heterogeneous temporal graph learning,
bytecode representation learning and a pattern-recognition module (temporal attack motifs,
prototype-based explanations, open-set detection of new attack families). Built on real
on-chain data from Ethereum, Base, BSC and Arbitrum.

## Layout

| Path | What it does |
|---|---|
| `delegnet/collect.py` | Real-chain collection: type-4 tx scan + authority recovery, delegate code, native/internal transfers (`trace_filter`), ERC-20 logs, tx senders, 2-hop expansion |
| `delegnet/labels.py` | Outcome-based labels (drains into shared attacker sinks vs owners still in control) + curated/public list cross-checks |
| `delegnet/graph.py` | Temporal heterogeneous graph, leak-free subgraph sampler, code families, campaigns |
| `delegnet/patterns/` | Pattern Recognition Module: `motifs.py` (temporal attack motifs), `code_idioms.py` (bytecode canonicalisation + idioms), `prototypes.py` (prototype layer, open-set score) |
| `delegnet/model.py` | DelegNet and all graph baselines/ablations as flag variants |
| `delegnet/baselines.py` | Hash blocklist, opcode-kNN, XGBoost |
| `delegnet/train.py` | Sample building (cached), training, prototype projection, contrastive code pre-training |
| `delegnet/experiments.py` | Protocols (temporal, family-disjoint, cross-chain + few-shot) and analyses (early warning, robustness, open-set, campaigns, prototypes, explanations) |
| `delegnet/report.py` | LaTeX tables and figures |
| `tests/` | Correctness checks (no future leakage, mutation invariance, attention maths) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
scripts/fetch_labels.sh
```

Free public RPC endpoints are configured in `configs/paper.yaml`. A provider key is optional and
only speeds things up / enables token transfers: `export ETH_RPC=...` and set `token_logs: true`.

## Pipeline

```bash
python run.py collect --chain ethereum      # scan, code, activity, hop2 (resumable)
python run.py label   --chain ethereum
python run.py build   --chain ethereum
python run.py pretrain
python run.py experiment --protocol temporal
python run.py experiment --protocol family
python run.py experiment --protocol crosschain
python run.py analyse
python run.py report
pytest -q tests
```

Data (`data/`) and results (`runs/`) are not in git; they are reproducible from the pipeline and
will be released separately as the dataset artifact.

## Working together

- `main` is protected: never commit to it directly.
- One branch per change: `git checkout -b feature/<short-name>` (or `fix/...`, `exp/...`).
- Push the branch and open a pull request into `main`; at least one other person reviews.
- Keep experiment settings in `configs/`, not hard-coded, so results stay reproducible.
- Run `pytest -q tests` before opening a PR.
