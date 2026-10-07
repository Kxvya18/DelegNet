"""XGBoost runs in a separate spawned process: its OpenMP runtime clashes with the one
torch links against, which segfaults when both live in the same process."""
import numpy as np


def fit(X, y, Xv, yv, seed):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8,
                          colsample_bytree=0.8, eval_metric="aucpr", early_stopping_rounds=100,
                          scale_pos_weight=max((y == 0).sum(), 1) / max((y == 1).sum(), 1),
                          random_state=seed, n_jobs=8, tree_method="hist")
    m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    return bytes(m.get_booster().save_raw("ubj"))


def predict(raw, X, leaves=False):
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(bytearray(raw))
    d = xgb.DMatrix(X)
    return b.predict(d, pred_leaf=True) if leaves else b.predict(d)


def call(fn, *args):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(fn, *args).result()


__all__ = ["fit", "predict", "call", "np"]
