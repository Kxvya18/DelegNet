"""Paper names of the methods in the code. The paper's DelegNet is the canonical code-pattern
recognition module, which the code calls OpcodeKNN; every other method keeps its baseline name."""
NAMES = {"OpcodeKNN": "DelegNet", "OpcodeKNN-raw": "Fuzzy hash", "HashBlocklist": "Blocklist",
         "XGBoost": "XGBoost", "GraphSAGE": "GraphSAGE", "TGAT": "TGAT"}


def name(method: str) -> str:
    base = method.replace(" (ensemble)", "")
    return NAMES.get(base, base) + (" (ens.)" if "(ensemble)" in method else "")


def paper_only(df, col="method"):
    """Rows of the methods the paper reports (older runs/ outputs may hold other, unreported models)."""
    return df[df[col].str.replace(" (ensemble)", "", regex=False).isin(NAMES)]
