# DelegNet

Detection of malicious EIP-7702 delegations from the delegate contract's code, on real on-chain
data from Ethereum. An EIP-7702 authorization hands an account's execution to a delegate
contract; drainer kits re-deploy the same sweeper logic under fresh addresses and lightly
obfuscated bytecode, so exact-hash blocklists miss them.

DelegNet's **Pattern Recognition Module** canonicalises the delegate's bytecode (removes junk
instructions, folds equivalent opcodes, strips padded trailers) and keeps a memory of the
canonical opcode 4-gram sets of every labelled training contract. A delegation is scored by its
nearest known patterns: `0.5 + (best Jaccard to a malicious contract - best Jaccard to a benign
one) / 2`. Because it needs only the code, it can screen a delegation before it is signed.

## Paper names and code names

| Paper | Code (`method` in `runs/` outputs) | Where |
|---|---|---|
| DelegNet (Pattern Recognition Module) | `OpcodeKNN` | `delegnet/baselines.py`, canonicalisation in `delegnet/patterns/code_idioms.py` |
| Fuzzy hash | `OpcodeKNN-raw` (same memory on raw opcodes) | `delegnet/baselines.py` |
| Blocklist | `HashBlocklist` (exact bytecode match) | `delegnet/baselines.py` |
| XGBoost | `XGBoost` (graph, motif, authorization and idiom features) | `delegnet/baselines.py` |
| TGAT, GraphSAGE | `TGAT`, `GraphSAGE` | `delegnet/model.py` |

`results/paper_names.py` holds this mapping for every generated figure and table.

## Layout

| Path | What it does |
|---|---|
| `delegnet/collect.py` | Real-chain collection: type-4 tx scan + authority recovery, delegate code, native/internal transfers (`trace_filter`), ERC-20 logs, tx senders, 2-hop expansion |
| `delegnet/labels.py` | Outcome-based labels (drains into shared attacker sinks vs owners still in control) + curated/public list cross-checks |
| `delegnet/graph.py` | Temporal heterogeneous graph, leak-free subgraph sampler, code families |
| `delegnet/evm.py` | Disassembler and semantics-preserving bytecode mutation (code-disguise attacks) |
| `delegnet/patterns/` | `code_idioms.py` (canonicalisation + bytecode idioms), `motifs.py` (temporal motifs used by the graph/tabular baselines) |
| `delegnet/baselines.py` | DelegNet (`OpcodeKNN`), fuzzy hash, blocklist, XGBoost |
| `delegnet/model.py`, `delegnet/train.py` | TGAT / GraphSAGE baselines, sample building (cached), training |
| `delegnet/tuning.py` | Equal-budget hyperparameter search on validation families |
| `delegnet/experiments.py` | Zero-day protocol (leave-family-out), temporal / family / cross-chain protocols, capability analyses (monitoring, code disguise, evasion suite, pre-signature screening) |
| `delegnet/evasion.py` | Adaptive-attacker transformations (self-submit, fresh sponsor, fresh contract, code disguise, combined) |
| `delegnet/diagnostics.py` | History buckets, contract-level AUROC, significance, train/test gaps, label shuffle |
| `delegnet/report.py` | Summary tables of a run |
| `results/make_figures.py` | Paper figures and tables (`results/figures`, `results/tables`) |
| `results/supporting_numbers.py` | Remaining paper numbers: pre-signature false-alarm bound, token exposure, significance |
| `results/presignature_scores.py` | Signing-time scores of the saved models (cached for the pre-signature figure) |
| `tests/` | Correctness checks (no future leakage, mutation and canonicalisation, pattern scoring, attention maths) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
scripts/fetch_labels.sh
```

Free public RPC endpoints are configured in `configs/paper.yaml`. A provider key is optional and
only speeds things up / enables token transfers: `export ETH_RPC=...` and set `token_logs: true`.

Keep collected data **outside** cloud-synced folders (OneDrive, iCloud): the pipeline writes tens of
thousands of files. Point to it without editing the config, e.g. in `.env.local`:

```bash
export DELEGNET_DATA_ROOT=~/DelegNet-local/full
export DELEGNET_SCAN_ROOT=~/DelegNet-local/scan
```

## Pipeline

```bash
scripts/run_pipeline.sh ethereum        # collect -> label -> build graph
scripts/run_experiments.sh              # tune, all protocols, diagnostics, analyses, report
scripts/run_experiments.sh zeroday      # resume from specific protocols
```

Or step by step:

```bash
python run.py collect --chain ethereum --steps scan,sample,code,activity,hop2   # resumable
python run.py label   --chain ethereum
python run.py build   --chain ethereum
python run.py tune                                    # TGAT, GraphSAGE, XGBoost; validation families only
python run.py experiment --protocol zeroday           # headline: unseen contract families, 3 seeds
python run.py diagnose                                # history buckets, significance, leakage checks
python run.py analyse                                 # monitoring, code disguise, evasion suite, pre-signature
python run.py report
pytest -q tests
```

`experiment --protocol temporal | family | crosschain` run the additional split protocols.

## Reproducing the paper's numbers

Every figure and table in the paper comes from `runs/` through two scripts (run from the repo root):

```bash
source .env.local
PYTHONPATH=. python results/make_figures.py         # figures + most tables
PYTHONPATH=. python results/supporting_numbers.py   # false_alarm_bound, token_exposure, significance_delegnet
```

Data (`data/`) and results (`runs/`) are not in git; they are reproducible from the pipeline and
will be released separately as the dataset artifact. The manuscript is kept outside the repository.

## Working together

- `main` is protected: never commit to it directly.
- One branch per change: `git checkout -b feature/<short-name>` (or `fix/...`, `exp/...`).
- Push the branch and open a pull request into `main`; it is reviewed before merging.
- Keep experiment settings in `configs/`, not hard-coded, so results stay reproducible.
- Run `pytest -q tests` before opening a PR.
