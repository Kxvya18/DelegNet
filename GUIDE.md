# DelegNet: the easy guide

Start here. It explains what the project is, how far it has got, and exactly how to join in.

---

## 1. What is this project? (the 1-minute version)

In May 2025 Ethereum added a feature called **EIP-7702**. With **one signature**, a normal wallet can
hand control of itself to a piece of code (a "smart contract").

That's useful (smarter wallets, batched payments), but scammers abuse it:

- they trick people, or use stolen keys, into signing that handover to **their** code;
- their code then **empties the wallet**, often within seconds, every time money arrives.

**Our project, DelegNet**, is an AI system that looks at these handovers and says
**"this one is a scam"**, ideally *before* the money is stolen.

We are writing it up as a research paper for an **IEEE Transactions** journal.

### How it decides (in plain words)

It looks at three kinds of clues:

| Clue | What it means |
|---|---|
| **Who is connected to whom** | Money flows between wallets form a network (a "graph"). Scammers leave tell-tale shapes, e.g. one attacker signing for 73 victims in a single transaction. |
| **What happened over time** | The order and speed of events. "Money arrives, then gets swept to the attacker 12 seconds later" is a classic pattern. |
| **What the code looks like** | The contract the wallet was handed to. Scam contracts reuse the same tricks, even when disguised. |

On top of that sits a **pattern recognition module** that:

1. spots known attack sequences;
2. explains each alert as "this looks like known scam pattern #4, here's the real example";
3. raises a flag when an attack looks like **nothing seen before**.

---

## 2. What is done so far

| Status | Item |
|---|---|
| ✅ | Code that downloads **real** blockchain data and finds every handover |
| ✅ | Automatic scam labelling from what actually happened to the wallets afterwards |
| ✅ | The network/time model, the code reader and the pattern recognition module |
| ✅ | Simpler comparison methods (needed to show ours is better) |
| ✅ | Tests that prove the system never "peeks into the future" (that would be cheating) |
| ✅ | Charts and tables for the paper (generated automatically) |
| ⏳ | Downloading the full Ethereum data (running now) |
| ⏳ | Training and results on the full data |
| ⏳ | Other blockchains: Base, BSC, Arbitrum |
| ⬜ | Writing the paper text |

**First real finding:** in just the first days after the upgrade, one scam contract had taken over
**968 wallets**, and the attacker signed for **73 victims in one transaction**.

---

## 3. The parts of the code (what each file does)

```
DelegNet/
├── run.py                   ← the one command you run for every step
├── configs/paper.yaml       ← all settings (dates, servers, model size...)
├── delegnet/
│   ├── collect.py           ← downloads blockchain data
│   ├── labels.py            ← decides which handovers were scams (from what happened afterwards)
│   ├── graph.py             ← builds the "who sent what to whom, when" network
│   ├── patterns/
│   │   ├── motifs.py        ← spots attack sequences in time
│   │   ├── code_idioms.py   ← reads contract code, sees through disguises
│   │   └── prototypes.py    ← "looks like known scam #4" explanations, new-attack alarm
│   ├── model.py             ← the AI model (DelegNet) and its simpler variants
│   ├── baselines.py         ← simple comparison methods (blocklist, code matching, XGBoost)
│   ├── train.py             ← trains the models
│   ├── experiments.py       ← runs all the experiments for the paper
│   ├── evaluate.py          ← scores: accuracy, how early we warn, money saved
│   ├── report.py            ← makes the paper's tables and charts
│   └── evm.py               ← low-level reader/writer for contract bytecode
├── tests/test_core.py       ← safety checks (run before every pull request)
└── scripts/fetch_labels.sh  ← downloads a public scam-address list
```

**Not in GitHub on purpose:** the downloaded data (`data/`) and results (`runs/`). They are big and
anyone can rebuild them with the steps below.

---

## 4. Getting set up (one time, about 15 minutes)

You need: a GitHub account, **Python 3.11 or 3.12**, and **git**.

**Step 1: accept the invite.** Check your email for the GitHub invitation to `Kxvya18/DelegNet`
and click **Accept**.

**Step 2: download the project.**

```bash
git clone https://github.com/Kxvya18/DelegNet.git
cd DelegNet
```

**Step 3: create a private Python environment and install everything.**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
scripts/fetch_labels.sh            # optional: public scam list
```

**Step 4: check everything works** (needs some data first; see section 5).

```bash
pytest -q tests
```

---

## 5. Running the project

Every step is one command. They run in this order:

```bash
python run.py collect --chain ethereum      # 1. download data (slow, can be stopped and resumed)
python run.py label   --chain ethereum      # 2. mark scams / safe ones
python run.py build   --chain ethereum      # 3. build the network
python run.py pretrain                      # 4. teach the code reader
python run.py experiment --protocol temporal    # 5. main experiment
python run.py experiment --protocol family      #    unseen scam families
python run.py experiment --protocol crosschain  #    other blockchains
python run.py analyse                       # 6. early warning, robustness, explanations
python run.py report                        # 7. tables + charts land in runs/report/
```

Tips:
- Downloading uses **free** public servers, so it is slow. You don't have to download yourself;
  ask for the data folder if you only want to run experiments.
- If you get a free key from Alchemy or Etherscan later, it only makes downloading faster.

---

## 6. How we work together (branches and pull requests)

**Golden rule: nobody changes `main` directly.** `main` is the clean, working version.
GitHub blocks direct pushes to it.

Each piece of work goes like this:

**1. Get the latest version.**
```bash
git checkout main
git pull
```

**2. Make your own branch** (a private copy for your change).
```bash
git checkout -b feature/what-you-are-doing
```
Name ideas: `feature/...` for something new, `fix/...` for a bug, `exp/...` for an experiment,
`docs/...` for writing.

**3. Make your changes, then save them.**
```bash
pytest -q tests                  # make sure nothing broke
git add -A
git commit -m "Short description of what you changed"
```

**4. Upload your branch.**
```bash
git push -u origin feature/what-you-are-doing
```

**5. Raise a pull request (PR).** A PR is "please review my change and add it to main".
- Go to https://github.com/Kxvya18/DelegNet.
- GitHub shows a yellow bar: **"Compare & pull request"**. Click it.
- Write what you changed and why, then click **Create pull request**.
- Someone looks at it, comments or approves, then clicks **Merge**. Done!

**6. Start your next piece of work from step 1 again.**

---

## 7. What you can help with

Pick anything, tell the team, and open a branch for it:

- **Check labels by hand:** open `data/<chain>/label_audit.csv`, look up the top contracts on
  Etherscan, and note whether our "scam / safe" call is right. Reviewers love this.
- **Get a free Etherscan API key** (etherscan.io → sign up → API keys) so we can add token
  (USDC/USDT) theft to the data.
- **Read related papers** and add them to the literature list for the paper.
- **Write paper sections:** introduction, related work, dataset description.
- **Run experiments** on your machine to speed things up (different seeds or chains).
- **Make figures nicer** for the paper (`delegnet/report.py`).

---

## 8. Mini dictionary

| Word | Meaning |
|---|---|
| **Wallet / EOA** | A normal crypto account controlled by a private key |
| **Smart contract** | A program that lives on the blockchain |
| **Delegation / handover** | EIP-7702: a wallet hands control to a contract with one signature |
| **Sweeper / drainer** | Scam contracts that empty wallets automatically |
| **Sponsor** | Whoever submitted the handover transaction (often the attacker) |
| **Sink** | The attacker's wallet that collects stolen money |
| **Graph** | The network of wallets and money flows |
| **Branch** | Your own copy of the code to work on safely |
| **Pull request (PR)** | A request to add your branch's changes to `main` |
| **Merge** | Accepting a PR into `main` |
